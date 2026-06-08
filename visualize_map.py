"""GenCast prediction overlaid on world map — preview of website look."""
import sys, dataclasses
import haiku as hk
import jax, jax.numpy as jnp
import numpy as np
import xarray
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, '/home/hama/agents/weather/graphcast')
from graphcast import checkpoint, gencast, denoiser, normalization, nan_cleaning, data_utils

DATA_DIR = '/home/hama/agents/weather/data'

print("Loading model + running 5 ensemble members...")
with open(f'{DATA_DIR}/params/GenCast_1p0deg_Mini.npz', 'rb') as f:
    ckpt = checkpoint.load(f, gencast.CheckPoint)
params = ckpt.params
state  = {}

_arch = dataclasses.asdict(ckpt.denoiser_architecture_config)
_arch['sparse_transformer_config']['attention_type'] = 'mha'
denoiser_arch_cfg = denoiser.DenoiserArchitectureConfig(
    **{k: (denoiser.SparseTransformerConfig(**v) if k == 'sparse_transformer_config' else v)
       for k, v in _arch.items()})

diffs_stddev    = xarray.open_dataset(f'{DATA_DIR}/stats/diffs_stddev_by_level.nc', engine='netcdf4').load()
mean_by_level   = xarray.open_dataset(f'{DATA_DIR}/stats/mean_by_level.nc', engine='netcdf4').load()
stddev_by_level = xarray.open_dataset(f'{DATA_DIR}/stats/stddev_by_level.nc', engine='netcdf4').load()
min_by_level    = xarray.open_dataset(f'{DATA_DIR}/stats/min_by_level.nc', engine='netcdf4').load()
example_batch   = xarray.open_dataset(f'{DATA_DIR}/dataset/era5_1p0deg_1step.nc', engine='netcdf4').load()

task_config = ckpt.task_config
eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times=slice("12h", "12h"),
    **dataclasses.asdict(task_config))

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

members = []
for seed in range(5):
    rng = jax.random.PRNGKey(seed)
    (pred, _) = run_jit(params, state, rng, eval_inputs, eval_targets * np.nan, eval_forcings)
    members.append(pred)
    print(f"  member {seed+1}/5")

lats = eval_inputs['lat'].values   # 181 points -90..90
lons = eval_inputs['lon'].values   # 360 points 0..359

# Compute ensemble mean and spread
mslp_all = np.stack([m['mean_sea_level_pressure'].values[0,0]/100 for m in members])  # hPa
t2m_all  = np.stack([m['2m_temperature'].values[0,0] - 273.15 for m in members])       # °C
u10_all  = np.stack([m['10m_u_component_of_wind'].values[0,0] for m in members])
v10_all  = np.stack([m['10m_v_component_of_wind'].values[0,0] for m in members])

mslp_mean = mslp_all.mean(axis=0)
t2m_mean  = t2m_all.mean(axis=0)
t2m_spread= t2m_all.std(axis=0)
wspd_mean = np.sqrt(u10_all.mean(axis=0)**2 + v10_all.mean(axis=0)**2)

# Load a simple world coastline via matplotlib boundaries
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch

# ---------- FIGURE 1: Temperature + MSLP contours (website preview) ----------
fig, ax = plt.subplots(figsize=(18, 9), facecolor='#0d1b2a')
ax.set_facecolor('#0d1b2a')

# Shift lons 0-360 → -180..180 for cleaner display
lon_shift = np.where(lons > 180, lons - 360, lons)
sort_idx  = np.argsort(lon_shift)
lon_s     = lon_shift[sort_idx]
t2m_s     = t2m_mean[:, sort_idx]
mslp_s    = mslp_mean[:, sort_idx]

# Temperature heatmap
cmap_temp = LinearSegmentedColormap.from_list('temp', [
    '#0a1628', '#1a3a5c', '#0d5986', '#1b9aaa', '#56cfb2',
    '#f7f7f7', '#f4a261', '#e76f51', '#9b2226', '#641220'])
im = ax.imshow(t2m_s, origin='upper', cmap=cmap_temp, vmin=-50, vmax=45,
               extent=[lon_s[0], lon_s[-1], lats[-1], lats[0]], aspect='auto', alpha=0.85)

# MSLP contours
lon_grid, lat_grid = np.meshgrid(lon_s, lats)
cs = ax.contour(lon_grid, lat_grid, mslp_s, levels=np.arange(960, 1045, 8),
                colors='white', linewidths=0.5, alpha=0.4)
ax.clabel(cs, fmt='%d', fontsize=6, colors='white', inline=True)

# Draw simple coastlines from matplotlib's built-in data
try:
    import cartopy.feature as cfeature
    import cartopy.crs as ccrs
    # cartopy not available, skip
except ImportError:
    pass

# Draw lat/lon grid
for lat in range(-60, 91, 30):
    ax.axhline(lat, color='white', lw=0.2, alpha=0.2)
for lon in range(-180, 181, 60):
    ax.axvline(lon, color='white', lw=0.2, alpha=0.2)
    ax.text(lon, -85, f'{lon}°', color='#888', fontsize=7, ha='center')
for lat in [-60, -30, 0, 30, 60, 90]:
    ax.text(-178, lat, f'{lat}°', color='#888', fontsize=7, va='center')

ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.axis('off')

# Colorbar
cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.015, pad=0.01)
cbar.set_label('2m Temperature (°C)', color='white', fontsize=10)
cbar.ax.yaxis.set_tick_params(color='#aaa')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#aaa', fontsize=8)

# Title & labels
ax.set_title('GenCast アンサンブル予測  (5 members)  |  +12h  |  ERA5 2019-03-29',
             color='white', fontsize=14, fontweight='bold', pad=10)

# Overlay legend
legend_text = [
    '■ 塗り: 地上2m気温',
    '○ 白線: 海面気圧 (hPa)',
    '● 5メンバー平均',
]
for i, txt in enumerate(legend_text):
    ax.text(0.01, 0.08 - i*0.04, txt, transform=ax.transAxes,
            color='#ccc', fontsize=9,
            path_effects=[pe.withStroke(linewidth=2, foreground='#0d1b2a')])

# Website UI mockup label
ax.text(0.5, 0.97, '🌀 Weather Forecast — powered by Google DeepMind GenCast',
        transform=ax.transAxes, color='#56cfb2', fontsize=11,
        ha='center', va='top', fontweight='bold',
        path_effects=[pe.withStroke(linewidth=3, foreground='#0d1b2a')])

plt.tight_layout(pad=0)
out1 = '/tmp/gencast_map.png'
plt.savefig(out1, dpi=150, bbox_inches='tight', facecolor='#0d1b2a')
print(f"Saved: {out1}")

# ---------- FIGURE 2: Ensemble spread (uncertainty) around Japan ----------
fig2, ax2 = plt.subplots(figsize=(10, 8), facecolor='#0d1b2a')
ax2.set_facecolor('#0d1b2a')

# Focus on Western Pacific (typhoon region)
lat_mask = (lats >= 10) & (lats <= 50)
lon_mask  = (lon_s >= 110) & (lon_s <= 160)
t2m_sub   = t2m_spread[np.ix_(lat_mask, lon_mask)]
mslp_sub  = mslp_s[np.ix_(lat_mask, lon_mask)]
u_sub     = u10_all.mean(axis=0)[:, sort_idx][np.ix_(lat_mask, lon_mask)]
v_sub     = v10_all.mean(axis=0)[:, sort_idx][np.ix_(lat_mask, lon_mask)]

lats_sub = lats[lat_mask]
lons_sub = lon_s[lon_mask]
lg, la = np.meshgrid(lons_sub, lats_sub)

cmap_spread = LinearSegmentedColormap.from_list('spread', [
    '#0d1b2a', '#1a3a5c', '#0d5986', '#f4a261', '#e76f51', '#9b2226'])
im2 = ax2.imshow(t2m_sub, origin='upper', cmap=cmap_spread, vmin=0, vmax=4,
                 extent=[lons_sub[0], lons_sub[-1], lats_sub[-1], lats_sub[0]], aspect='auto')

# MSLP contours
cs2 = ax2.contour(lg, la, mslp_sub, levels=np.arange(960, 1045, 4),
                  colors='white', linewidths=0.7, alpha=0.5)
ax2.clabel(cs2, fmt='%d', fontsize=7, colors='white', inline=True)

# Wind vectors (subsampled)
step = 5
ax2.quiver(lg[::step, ::step], la[::step, ::step],
           u_sub[::step, ::step], v_sub[::step, ::step],
           color='white', alpha=0.6, scale=200, width=0.003)

ax2.set_xlim(110, 160)
ax2.set_ylim(10, 50)
ax2.set_title('Western Pacific  |  T2m Ensemble Spread + Wind Vectors + MSLP',
              color='white', fontsize=12, fontweight='bold')
ax2.tick_params(colors='#aaa', labelsize=9)
for spine in ax2.spines.values():
    spine.set_edgecolor('#333')

cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.04, pad=0.02)
cbar2.set_label('T2m Spread (K std, 5 members)', color='white', fontsize=9)
plt.setp(cbar2.ax.yaxis.get_ticklabels(), color='#aaa')

# Japan region box
from matplotlib.patches import Rectangle
rect = Rectangle((129, 30), 16, 15, linewidth=2, edgecolor='#56cfb2',
                 facecolor='none', linestyle='--')
ax2.add_patch(rect)
ax2.text(137, 46, 'Japan', color='#56cfb2', fontsize=10, ha='center',
         fontweight='bold',
         path_effects=[pe.withStroke(linewidth=3, foreground='#0d1b2a')])

plt.tight_layout()
out2 = '/tmp/gencast_pacific.png'
plt.savefig(out2, dpi=150, bbox_inches='tight', facecolor='#0d1b2a')
print(f"Saved: {out2}")
