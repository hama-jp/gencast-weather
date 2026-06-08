"""Visualize GenCast prediction output."""
import sys, time, dataclasses
import haiku as hk
import jax, jax.numpy as jnp
import numpy as np
import xarray
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, '/home/hama/agents/weather/graphcast')
from graphcast import checkpoint, gencast, denoiser, normalization, nan_cleaning, data_utils

DATA_DIR = '/home/hama/agents/weather/data'

print("Loading model...")
with open(f'{DATA_DIR}/params/GenCast_1p0deg_Mini.npz', 'rb') as f:
    ckpt = checkpoint.load(f, gencast.CheckPoint)
params = ckpt.params
state  = {}
task_config          = ckpt.task_config
sampler_config       = ckpt.sampler_config
noise_config         = ckpt.noise_config
noise_encoder_config = ckpt.noise_encoder_config

import dataclasses as dc
_arch = dc.asdict(ckpt.denoiser_architecture_config)
_arch['sparse_transformer_config']['attention_type'] = 'mha'
denoiser_arch_cfg = denoiser.DenoiserArchitectureConfig(
    **{k: (denoiser.SparseTransformerConfig(**v) if k == 'sparse_transformer_config' else v)
       for k, v in _arch.items()})

diffs_stddev    = xarray.open_dataset(f'{DATA_DIR}/stats/diffs_stddev_by_level.nc', engine='netcdf4').load()
mean_by_level   = xarray.open_dataset(f'{DATA_DIR}/stats/mean_by_level.nc', engine='netcdf4').load()
stddev_by_level = xarray.open_dataset(f'{DATA_DIR}/stats/stddev_by_level.nc', engine='netcdf4').load()
min_by_level    = xarray.open_dataset(f'{DATA_DIR}/stats/min_by_level.nc', engine='netcdf4').load()
example_batch   = xarray.open_dataset(f'{DATA_DIR}/dataset/era5_1p0deg_1step.nc', engine='netcdf4').load()

eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times=slice("12h", "12h"),
    **dc.asdict(task_config))

def construct_predictor():
    p = gencast.GenCast(
        sampler_config=sampler_config, task_config=task_config,
        denoiser_architecture_config=denoiser_arch_cfg,
        noise_config=noise_config, noise_encoder_config=noise_encoder_config,
    )
    p = normalization.InputsAndResiduals(p, diffs_stddev_by_level=diffs_stddev,
        mean_by_level=mean_by_level, stddev_by_level=stddev_by_level)
    p = nan_cleaning.NaNCleaner(predictor=p, reintroduce_nans=True,
        fill_value=min_by_level, var_to_clean='sea_surface_temperature')
    return p

@hk.transform_with_state
def run_forward(inputs, targets_template, forcings):
    return construct_predictor()(inputs, targets_template=targets_template, forcings=forcings)

run_jit = jax.jit(run_forward.apply)

# Run 3 ensemble members for spread visualization
print("Running 3 ensemble members (for uncertainty spread)...")
ensemble_preds = []
for seed in range(3):
    rng = jax.random.PRNGKey(seed)
    (pred, _) = run_jit(params, state, rng, eval_inputs, eval_targets * np.nan, eval_forcings)
    ensemble_preds.append(pred)
    print(f"  member {seed+1}/3 done")

# Stack ensemble
t2m_members = np.stack([
    float(m['2m_temperature'].values[0, 0, :, :].mean()) for m in ensemble_preds
])
print(f"  T2m ensemble mean: {t2m_members.mean():.2f} K")

# --- Visualization ---
fig, axes = plt.subplots(2, 2, figsize=(16, 9), facecolor='#0a0f1e')
fig.suptitle('GenCast 1°  |  ERA5 2019-03-29  |  +12h Forecast',
             fontsize=16, color='white', fontweight='bold', y=0.98)

lats = eval_inputs['lat'].values
lons = eval_inputs['lon'].values

def plot_field(ax, data, title, cmap, vmin=None, vmax=None, unit=''):
    ax.set_facecolor('#0a0f1e')
    im = ax.imshow(data, origin='upper', cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[lons[0], lons[-1], lats[-1], lats[0]], aspect='auto')
    ax.set_title(title, color='white', fontsize=12, pad=6)
    ax.tick_params(colors='#666')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.ax.yaxis.set_tick_params(color='#aaa')
    cb.set_label(unit, color='#aaa', fontsize=9)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='#aaa')
    return im

pred = ensemble_preds[0]
spread = np.stack([m['2m_temperature'].values[0, 0] for m in ensemble_preds]).std(axis=0)

# 1) 2m Temperature
t2m = pred['2m_temperature'].values[0, 0] - 273.15  # K → °C
plot_field(axes[0,0], t2m, '2m Temperature (+12h)', 'RdBu_r',
           vmin=t2m.min(), vmax=t2m.max(), unit='°C')

# 2) Mean Sea Level Pressure
mslp = pred['mean_sea_level_pressure'].values[0, 0] / 100  # Pa → hPa
plot_field(axes[0,1], mslp, 'Mean Sea Level Pressure (+12h)', 'viridis',
           vmin=960, vmax=1040, unit='hPa')

# 3) 10m Wind Speed
u10 = pred['10m_u_component_of_wind'].values[0, 0]
v10 = pred['10m_v_component_of_wind'].values[0, 0]
wspd = np.sqrt(u10**2 + v10**2)
plot_field(axes[1,0], wspd, '10m Wind Speed (+12h)', 'plasma',
           vmin=0, vmax=25, unit='m/s')

# 4) Ensemble spread (T2m std across 3 members)
im = axes[1,1].imshow(spread, origin='upper', cmap='YlOrRd',
                       extent=[lons[0], lons[-1], lats[-1], lats[0]], aspect='auto')
axes[1,1].set_facecolor('#0a0f1e')
axes[1,1].set_title('T2m Ensemble Spread (3 members)', color='white', fontsize=12, pad=6)
axes[1,1].tick_params(colors='#666')
for spine in axes[1,1].spines.values():
    spine.set_edgecolor('#333')
cb = plt.colorbar(im, ax=axes[1,1], fraction=0.03, pad=0.02)
cb.set_label('K (std)', color='#aaa', fontsize=9)
plt.setp(cb.ax.yaxis.get_ticklabels(), color='#aaa')

# Annotation
fig.text(0.5, 0.01, 'GenCast 1p0deg Mini  •  JAX 0.10.1 + GPU  •  attention=mha',
         ha='center', color='#666', fontsize=9)

plt.tight_layout(rect=[0, 0.02, 1, 0.97])
out = '/tmp/gencast_preview.png'
plt.savefig(out, dpi=120, bbox_inches='tight', facecolor='#0a0f1e')
print(f"Saved: {out}")
