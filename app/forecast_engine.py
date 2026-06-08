"""GenCast inference engine — loads model once, runs on demand."""
import sys, dataclasses, time
import haiku as hk
import jax, jax.numpy as jnp
import numpy as np
import xarray

sys.path.insert(0, '/home/hama/agents/weather/graphcast')
from graphcast import checkpoint, gencast, denoiser, normalization, nan_cleaning, data_utils

DATA_DIR = '/home/hama/agents/weather/data'
_model_loaded = False
_run_jit = None
_params = None
_state = None
_task_config = None
_diffs_stddev = _mean_by_level = _stddev_by_level = _min_by_level = None


def load_model():
    global _model_loaded, _run_jit, _params, _state, _task_config
    global _diffs_stddev, _mean_by_level, _stddev_by_level, _min_by_level

    if _model_loaded:
        return

    print("Loading GenCast model...", flush=True)
    with open(f'{DATA_DIR}/params/GenCast_1p0deg_Mini.npz', 'rb') as f:
        ckpt = checkpoint.load(f, gencast.CheckPoint)

    _params = ckpt.params
    _state  = {}
    _task_config = ckpt.task_config

    _arch = dataclasses.asdict(ckpt.denoiser_architecture_config)
    _arch['sparse_transformer_config']['attention_type'] = 'mha'
    denoiser_arch_cfg = denoiser.DenoiserArchitectureConfig(
        **{k: (denoiser.SparseTransformerConfig(**v) if k == 'sparse_transformer_config' else v)
           for k, v in _arch.items()})

    _diffs_stddev    = xarray.open_dataset(f'{DATA_DIR}/stats/diffs_stddev_by_level.nc', engine='netcdf4').load()
    _mean_by_level   = xarray.open_dataset(f'{DATA_DIR}/stats/mean_by_level.nc', engine='netcdf4').load()
    _stddev_by_level = xarray.open_dataset(f'{DATA_DIR}/stats/stddev_by_level.nc', engine='netcdf4').load()
    _min_by_level    = xarray.open_dataset(f'{DATA_DIR}/stats/min_by_level.nc', engine='netcdf4').load()

    def construct_predictor():
        p = gencast.GenCast(
            sampler_config=ckpt.sampler_config, task_config=_task_config,
            denoiser_architecture_config=denoiser_arch_cfg,
            noise_config=ckpt.noise_config, noise_encoder_config=ckpt.noise_encoder_config,
        )
        p = normalization.InputsAndResiduals(p, diffs_stddev_by_level=_diffs_stddev,
            mean_by_level=_mean_by_level, stddev_by_level=_stddev_by_level)
        return nan_cleaning.NaNCleaner(predictor=p, reintroduce_nans=True,
            fill_value=_min_by_level, var_to_clean='sea_surface_temperature')

    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
        return construct_predictor()(inputs, targets_template=targets_template, forcings=forcings)

    _run_jit = jax.jit(run_forward.apply)
    _model_loaded = True
    print("Model ready.", flush=True)


def run_forecast(num_members: int = 5, progress_cb=None) -> dict:
    """Run GenCast and return JSON-serializable forecast dict."""
    load_model()

    example_batch = xarray.open_dataset(
        f'{DATA_DIR}/dataset/era5_1p0deg_1step.nc', engine='netcdf4').load()

    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
        example_batch, target_lead_times=slice("12h", "12h"),
        **dataclasses.asdict(_task_config))

    lats = eval_inputs['lat'].values.tolist()
    lons = eval_inputs['lon'].values.tolist()

    members_data = []
    t0 = time.time()

    for seed in range(num_members):
        rng = jax.random.PRNGKey(seed)
        (pred, _) = _run_jit(_params, _state, rng,
                              eval_inputs, eval_targets * np.nan, eval_forcings)
        jax.block_until_ready(pred)

        # Downsample 181×360 → 91×180 for bandwidth
        def downsample(arr):
            return arr[::2, ::2].tolist()

        members_data.append({
            't2m':  downsample(pred['2m_temperature'].values[0, 0] - 273.15),
            'mslp': downsample(pred['mean_sea_level_pressure'].values[0, 0] / 100),
            'u10':  downsample(pred['10m_u_component_of_wind'].values[0, 0]),
            'v10':  downsample(pred['10m_v_component_of_wind'].values[0, 0]),
        })

        elapsed = time.time() - t0
        if progress_cb:
            progress_cb(seed + 1, num_members, elapsed)
        print(f"  member {seed+1}/{num_members} ({elapsed:.1f}s)", flush=True)

    # Ensemble stats
    t2m_stack  = np.array([m['t2m']  for m in members_data])
    mslp_stack = np.array([m['mslp'] for m in members_data])
    u10_stack  = np.array([m['u10']  for m in members_data])
    v10_stack  = np.array([m['v10']  for m in members_data])

    # Downsampled grid coordinates
    lats_ds = lats[::2]
    lons_ds = lons[::2]

    return {
        'meta': {
            'model': 'GenCast 1p0deg Mini',
            'source': 'ERA5 2019-03-29',
            'lead_time': '+12h',
            'num_members': num_members,
            'elapsed_sec': round(time.time() - t0, 1),
        },
        'grid': {'lats': lats_ds, 'lons': lons_ds},
        'ensemble_mean': {
            't2m':  t2m_stack.mean(axis=0).tolist(),
            'mslp': mslp_stack.mean(axis=0).tolist(),
            'u10':  u10_stack.mean(axis=0).tolist(),
            'v10':  v10_stack.mean(axis=0).tolist(),
        },
        'ensemble_spread': {
            't2m':  t2m_stack.std(axis=0).tolist(),
            'mslp': mslp_stack.std(axis=0).tolist(),
        },
        'members': members_data,
    }
