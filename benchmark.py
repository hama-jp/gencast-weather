#!/usr/bin/env python3
"""
Benchmark GenCast inference time across models.
Usage:
  python benchmark.py 1p0deg_mini    # default
  python benchmark.py 1p0deg
  python benchmark.py 0p25deg
  python benchmark.py 0p25deg_ops
"""
import os, sys, time, dataclasses
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax, jax.numpy as jnp
import numpy as np
import haiku as hk
import xarray

sys.path.insert(0, '/home/hama/agents/weather/graphcast')
from graphcast import checkpoint, gencast, denoiser, normalization, nan_cleaning, data_utils
from google.cloud import storage

DATA_DIR = '/home/hama/agents/weather/data'

MODELS = {
    '1p0deg_mini': {
        'ckpt_gcs': 'gencast/params/GenCast 1p0deg Mini <2019.npz',
        'ckpt_local': f'{DATA_DIR}/params/GenCast_1p0deg_Mini.npz',
        'data_gcs': 'gencast/dataset/source-era5_date-2019-03-29_res-1.0_levels-13_steps-01.nc',
        'data_local': f'{DATA_DIR}/dataset/era5_1p0deg_1step.nc',
    },
    '1p0deg': {
        'ckpt_gcs': 'gencast/params/GenCast 1p0deg <2019.npz',
        'ckpt_local': f'{DATA_DIR}/params/GenCast_1p0deg.npz',
        'data_gcs': 'gencast/dataset/source-era5_date-2019-03-29_res-1.0_levels-13_steps-01.nc',
        'data_local': f'{DATA_DIR}/dataset/era5_1p0deg_1step.nc',
    },
    '0p25deg': {
        'ckpt_gcs': 'gencast/params/GenCast 0p25deg <2019.npz',
        'ckpt_local': f'{DATA_DIR}/params/GenCast_0p25deg.npz',
        'data_gcs': 'gencast/dataset/source-era5_date-2019-03-29_res-0.25_levels-13_steps-01.nc',
        'data_local': f'{DATA_DIR}/dataset/era5_0p25deg_1step.nc',
    },
    '0p25deg_ops': {
        'ckpt_gcs': 'gencast/params/GenCast 0p25deg Operational <2022.npz',
        'ckpt_local': f'{DATA_DIR}/params/GenCast_0p25deg_Operational.npz',
        'data_gcs': 'gencast/dataset/source-hres_date-2022-03-29_res-0.25_levels-13_steps-01.nc',
        'data_local': f'{DATA_DIR}/dataset/hres_0p25deg_1step.nc',
    },
}


def download_if_missing(gcs_path, local_path):
    if os.path.exists(local_path):
        print(f'  already have: {os.path.basename(local_path)}')
        return
    size_mb = _gcs_size(gcs_path) / 1e6
    print(f'  downloading {os.path.basename(local_path)} ({size_mb:.0f}MB)...')
    t0 = time.time()
    client = storage.Client.create_anonymous_client()
    bucket = client.get_bucket('dm_graphcast')
    bucket.blob(gcs_path).download_to_filename(local_path)
    print(f'  done in {time.time()-t0:.0f}s')


def _gcs_size(path):
    client = storage.Client.create_anonymous_client()
    bucket = client.get_bucket('dm_graphcast')
    return bucket.blob(path).size or 0


def run_benchmark(model_key):
    cfg = MODELS[model_key]
    print(f'\n{"="*60}')
    print(f'Model: {model_key}')
    print(f'{"="*60}')

    # Download if needed
    download_if_missing(cfg['ckpt_gcs'], cfg['ckpt_local'])
    download_if_missing(cfg['data_gcs'], cfg['data_local'])

    # Load checkpoint
    print('Loading checkpoint...')
    t0 = time.time()
    with open(cfg['ckpt_local'], 'rb') as f:
        ckpt = checkpoint.load(f, gencast.CheckPoint)
    print(f'  loaded in {time.time()-t0:.1f}s')

    task_config = ckpt.task_config
    print(f'  resolution: {task_config.input_variables[:2]}...')

    # Patch splash_mha → mha for GPU
    _arch = dataclasses.asdict(ckpt.denoiser_architecture_config)
    _arch['sparse_transformer_config']['attention_type'] = 'mha'
    denoiser_arch_cfg = denoiser.DenoiserArchitectureConfig(
        **{k: (denoiser.SparseTransformerConfig(**v) if k == 'sparse_transformer_config' else v)
           for k, v in _arch.items()})

    # Print architecture info
    sp = _arch['sparse_transformer_config']
    print(f'  layers: {sp["num_layers"]}, heads: {sp["num_heads"]}, '
          f'noise_levels: {ckpt.sampler_config.num_noise_levels}')

    # Load stats
    stats_dir = f'{DATA_DIR}/stats'
    diffs_stddev    = xarray.open_dataset(f'{stats_dir}/diffs_stddev_by_level.nc').load()
    mean_by_level   = xarray.open_dataset(f'{stats_dir}/mean_by_level.nc').load()
    stddev_by_level = xarray.open_dataset(f'{stats_dir}/stddev_by_level.nc').load()
    min_by_level    = xarray.open_dataset(f'{stats_dir}/min_by_level.nc').load()

    def construct_predictor():
        p = gencast.GenCast(
            sampler_config=ckpt.sampler_config, task_config=task_config,
            denoiser_architecture_config=denoiser_arch_cfg,
            noise_config=ckpt.noise_config, noise_encoder_config=ckpt.noise_encoder_config,
        )
        p = normalization.InputsAndResiduals(p, diffs_stddev_by_level=diffs_stddev,
            mean_by_level=mean_by_level, stddev_by_level=stddev_by_level)
        return nan_cleaning.NaNCleaner(predictor=p, reintroduce_nans=True,
            fill_value=min_by_level, var_to_clean='sea_surface_temperature')

    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
        return construct_predictor()(inputs, targets_template=targets_template, forcings=forcings)

    run_jit = jax.jit(run_forward.apply)
    params = ckpt.params
    state  = {}

    # Load data
    print('Loading ERA5 data...')
    example_batch = xarray.open_dataset(cfg['data_local'], engine='netcdf4').load()
    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
        example_batch, target_lead_times=slice("12h", "12h"),
        **dataclasses.asdict(task_config))

    grid_shape = (len(eval_inputs.lat), len(eval_inputs.lon))
    print(f'  grid: {grid_shape[0]} x {grid_shape[1]} = {grid_shape[0]*grid_shape[1]:,} points')

    # Warmup / JIT compile
    print('JIT compiling (first inference)...')
    rng = jax.random.PRNGKey(0)
    t_jit = time.time()
    try:
        (pred, _) = run_jit(params, state, rng, eval_inputs, eval_targets * np.nan, eval_forcings)
        jax.block_until_ready(pred)
        jit_time = time.time() - t_jit
        print(f'  JIT + inference: {jit_time:.1f}s')
    except Exception as e:
        print(f'  FAILED: {e}')
        return

    # Second run (pure inference, no JIT overhead)
    print('Second inference (no JIT overhead)...')
    rng2 = jax.random.PRNGKey(1)
    t_inf = time.time()
    (pred2, _) = run_jit(params, state, rng2, eval_inputs, eval_targets * np.nan, eval_forcings)
    jax.block_until_ready(pred2)
    inf_time = time.time() - t_inf
    print(f'  inference only: {inf_time:.1f}s')

    print(f'\nSummary [{model_key}]:')
    print(f'  JIT+inference: {jit_time:.1f}s')
    print(f'  inference/step: {inf_time:.1f}s')
    print(f'  5-member forecast: ~{5*inf_time:.0f}s ({5*inf_time/60:.1f}min)')
    return {'model': model_key, 'jit_sec': jit_time, 'step_sec': inf_time, 'grid': grid_shape}


if __name__ == '__main__':
    model = sys.argv[1] if len(sys.argv) > 1 else '1p0deg_mini'
    if model not in MODELS:
        print(f'Unknown model. Choose from: {list(MODELS.keys())}')
        sys.exit(1)
    run_benchmark(model)
