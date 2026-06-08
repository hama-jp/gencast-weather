"""GenCast Mini inference — direct call, no pmap/rollout complexity."""
import sys, time
import haiku as hk
import jax, jax.numpy as jnp
import numpy as np
import xarray

sys.path.insert(0, '/home/hama/agents/weather/graphcast')
from graphcast import checkpoint, gencast, denoiser, normalization, nan_cleaning, data_utils

print("JAX devices:", jax.devices())
DATA_DIR = '/home/hama/agents/weather/data'

print("Loading params...")
with open(f'{DATA_DIR}/params/GenCast_1p0deg_Mini.npz', 'rb') as f:
    ckpt = checkpoint.load(f, gencast.CheckPoint)
params = ckpt.params
state  = {}
task_config          = ckpt.task_config
sampler_config       = ckpt.sampler_config
noise_config         = ckpt.noise_config
noise_encoder_config = ckpt.noise_encoder_config
import dataclasses
# Replace splash_mha (TPU-only) with mha for GPU compatibility
_arch = dataclasses.asdict(ckpt.denoiser_architecture_config)
_arch['sparse_transformer_config']['attention_type'] = 'mha'
denoiser_arch_cfg = denoiser.DenoiserArchitectureConfig(
    **{k: (denoiser.SparseTransformerConfig(**v) if k == 'sparse_transformer_config' else v)
       for k, v in _arch.items()})
print(f"  input_duration={task_config.input_duration}  noise_levels={sampler_config.num_noise_levels}  attn=mha(gpu)")

print("Loading stats...")
diffs_stddev    = xarray.open_dataset(f'{DATA_DIR}/stats/diffs_stddev_by_level.nc', engine='netcdf4').load()
mean_by_level   = xarray.open_dataset(f'{DATA_DIR}/stats/mean_by_level.nc', engine='netcdf4').load()
stddev_by_level = xarray.open_dataset(f'{DATA_DIR}/stats/stddev_by_level.nc', engine='netcdf4').load()
min_by_level    = xarray.open_dataset(f'{DATA_DIR}/stats/min_by_level.nc', engine='netcdf4').load()

print("Loading sample data...")
example_batch = xarray.open_dataset(f'{DATA_DIR}/dataset/era5_1p0deg_1step.nc', engine='netcdf4').load()

eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch,
    target_lead_times=slice("12h", "12h"),  # single 12h step
    **dataclasses.asdict(task_config))
print(f"  inputs:  {dict(eval_inputs.sizes)}")
print(f"  targets: {dict(eval_targets.sizes)}")

def construct_predictor():
    p = gencast.GenCast(
        sampler_config=sampler_config,
        task_config=task_config,
        denoiser_architecture_config=denoiser_arch_cfg,
        noise_config=noise_config,
        noise_encoder_config=noise_encoder_config,
    )
    p = normalization.InputsAndResiduals(
        p, diffs_stddev_by_level=diffs_stddev,
        mean_by_level=mean_by_level, stddev_by_level=stddev_by_level,
    )
    p = nan_cleaning.NaNCleaner(
        predictor=p, reintroduce_nans=True,
        fill_value=min_by_level, var_to_clean='sea_surface_temperature',
    )
    return p

@hk.transform_with_state
def run_forward(inputs, targets_template, forcings):
    return construct_predictor()(inputs, targets_template=targets_template, forcings=forcings)

# JIT-compile
run_jit = jax.jit(run_forward.apply)

print("\nRunning 1 inference step (JIT compile on first call — may take a few minutes)...")
rng = jax.random.PRNGKey(42)
targets_template = eval_targets * np.nan

t0 = time.time()
(predictions, _state) = run_jit(params, state, rng, eval_inputs, targets_template, eval_forcings)
# Force materialization
jax.block_until_ready(predictions)
t1 = time.time()

print(f"Inference time (incl. JIT): {t1-t0:.1f}s")
print(f"Prediction vars: {list(predictions.data_vars)[:5]} ...")
print(f"Sample u_component_of_wind: {float(predictions['u_component_of_wind'].values.flat[0]):.3f}")
print("\nSUCCESS — GenCast inference running on GPU!")
