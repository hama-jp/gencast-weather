#!/usr/bin/env python3
"""
Fetch ERA5 reanalysis via CDS API for the most recent available date
(~5-day latency) and save as GenCast-compatible NetCDF.

Usage:
    python scripts/fetch_era5.py             # auto date
    python scripts/fetch_era5.py 2026-06-01  # specific date (t2 = that date 00 UTC)
"""
import sys
import cdsapi
import numpy as np
import xarray as xr
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data'
OUT      = DATA_DIR / 'dataset' / 'era5_latest.nc'
SAMPLE   = DATA_DIR / 'dataset' / 'era5_1p0deg_1step.nc'

ERA5_LAG_DAYS = 5

SURFACE_VARS = [
    '2m_temperature', 'sea_surface_temperature', 'mean_sea_level_pressure',
    '10m_u_component_of_wind', '10m_v_component_of_wind', 'total_precipitation',
]
PRESSURE_VARS = [
    'u_component_of_wind', 'v_component_of_wind', 'specific_humidity',
    'temperature', 'vertical_velocity', 'geopotential',
]
LEVELS = ['50','100','150','200','250','300','400','500','600','700','850','925','1000']


def get_times(ref=None):
    if ref:
        t2 = datetime.strptime(ref, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    else:
        now = datetime.now(timezone.utc)
        t = now - timedelta(days=ERA5_LAG_DAYS)
        h = 0 if t.hour < 12 else 12
        t2 = t.replace(hour=h, minute=0, second=0, microsecond=0)
    t1 = t2 - timedelta(hours=12)
    t0 = t2 - timedelta(hours=24)
    return t0, t1, t2


def download(c, dataset, req, path):
    if path.exists():
        print(f'  cached: {path.name}')
        return
    print(f'  requesting {path.name} ...')
    c.retrieve(dataset, req, str(path))
    print(f'  saved ({path.stat().st_size/1e6:.1f} MB)')


def _get_time_coord(ds):
    """Return whichever time coordinate exists (valid_time or time)."""
    for name in ('valid_time', 'time'):
        if name in ds.coords:
            return name
    raise KeyError(f'No time coord in {list(ds.coords)}')


def _normalize(ds):
    """Rename CDS coords → GenCast coords, flip lat, roll lon 0-360."""
    tc = _get_time_coord(ds)
    renames = {tc: 'time'}
    if 'latitude'  in ds.dims: renames['latitude']  = 'lat'
    if 'longitude' in ds.dims: renames['longitude'] = 'lon'
    if 'pressure_level' in ds.dims: renames['pressure_level'] = 'level'
    ds = ds.rename(renames)

    # latitude: ensure -90 → 90 (ascending)
    if float(ds.lat[0]) > float(ds.lat[-1]):
        ds = ds.isel(lat=slice(None, None, -1))

    # longitude: -180..179 → 0..359
    ds = ds.assign_coords(lon=(ds.lon % 360))
    ds = ds.sortby('lon')

    return ds


def _rename_vars(ds, mapping):
    """Rename only variables that exist in ds."""
    return ds.rename({k: v for k, v in mapping.items() if k in ds})


def assemble(times, sfc_path, pl_path):
    t0, t1, t2 = times

    sfc = _normalize(xr.open_dataset(sfc_path, engine='netcdf4'))
    pl  = _normalize(xr.open_dataset(pl_path,  engine='netcdf4'))

    # Select exact 3 time steps (CDS may include extras if crossing day boundary)
    want = [np.datetime64(t.replace(tzinfo=None), 'ns') for t in [t0, t1, t2]]
    try:
        sfc = sfc.sel(time=want)
        pl  = pl.sel(time=want)
    except KeyError:
        # Show what we actually got
        print(f'  available times: {sfc.time.values}')
        print(f'  wanted:          {want}')
        raise

    # Rename surface variables (CDS short names → GenCast names)
    sfc = _rename_vars(sfc, {
        't2m': '2m_temperature',
        'sst':  'sea_surface_temperature',
        'msl':  'mean_sea_level_pressure',
        'u10':  '10m_u_component_of_wind',
        'v10':  '10m_v_component_of_wind',
        'tp':   'total_precipitation_12hr',
    })

    # Rename pressure-level variables
    pl = _rename_vars(pl, {
        'u': 'u_component_of_wind',
        'v': 'v_component_of_wind',
        'q': 'specific_humidity',
        't': 'temperature',
        'w': 'vertical_velocity',
        'z': 'geopotential',
    })

    # Add batch dimension
    sfc = sfc.expand_dims('batch', axis=0)
    pl  = pl.expand_dims('batch', axis=0)

    # Replace absolute time coord with timedelta (GenCast convention)
    td = [np.timedelta64(0, 'ns'),
          np.timedelta64(12 * 3600 * 10**9, 'ns'),
          np.timedelta64(24 * 3600 * 10**9, 'ns')]
    sfc = sfc.assign_coords(time=td)
    pl  = pl.assign_coords(time=td)

    ds = xr.merge([sfc, pl])

    # Load static fields from sample (they don't change with time)
    sample = xr.open_dataset(SAMPLE, engine='netcdf4')
    for var in ('land_sea_mask', 'geopotential_at_surface'):
        if var in sample.data_vars:
            ds[var] = sample[var]

    # Add datetime coordinate (required by add_derived_vars for solar forcing)
    dt_arr = np.array([[
        np.datetime64(t0.replace(tzinfo=None), 'ns'),
        np.datetime64(t1.replace(tzinfo=None), 'ns'),
        np.datetime64(t2.replace(tzinfo=None), 'ns'),
    ]])
    ds = ds.assign_coords(datetime=xr.DataArray(dt_arr, dims=['batch', 'time']))

    return ds


def main():
    ref = sys.argv[1] if len(sys.argv) > 1 else None
    times = get_times(ref)
    t0, t1, t2 = times
    print(f'ERA5 target times: {t0:%Y-%m-%d %H:%M}  {t1:%Y-%m-%d %H:%M}  {t2:%Y-%m-%d %H:%M} UTC')

    c = cdsapi.Client()

    years  = sorted(set(str(t.year)        for t in times))
    months = sorted(set(f'{t.month:02d}'   for t in times))
    days   = sorted(set(f'{t.day:02d}'     for t in times))
    hours  = sorted(set(f'{t.hour:02d}:00' for t in times))

    tag = t2.strftime('%Y%m%d%H')
    sfc_tmp = DATA_DIR / 'dataset' / f'_era5_sfc_{tag}.nc'
    pl_tmp  = DATA_DIR / 'dataset' / f'_era5_pl_{tag}.nc'

    download(c, 'reanalysis-era5-single-levels', {
        'product_type': 'reanalysis',
        'variable': SURFACE_VARS,
        'year': years, 'month': months, 'day': days, 'time': hours,
        'data_format': 'netcdf', 'download_format': 'unarchived',
        'grid': ['1.0', '1.0'],
    }, sfc_tmp)

    download(c, 'reanalysis-era5-pressure-levels', {
        'product_type': 'reanalysis',
        'variable': PRESSURE_VARS,
        'pressure_level': LEVELS,
        'year': years, 'month': months, 'day': days, 'time': hours,
        'data_format': 'netcdf', 'download_format': 'unarchived',
        'grid': ['1.0', '1.0'],
    }, pl_tmp)

    print('Assembling...')
    ds = assemble(times, sfc_tmp, pl_tmp)

    ds.to_netcdf(str(OUT))
    size_mb = OUT.stat().st_size / 1e6
    print(f'Saved: {OUT}  ({size_mb:.0f} MB)')
    print(f'Variables: {sorted(ds.data_vars)}')
    print(f'Dims: {dict(ds.dims)}')
    print('Done.')


if __name__ == '__main__':
    main()
