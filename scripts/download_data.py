"""Download GenCast model weights and sample ERA5 data from Google Cloud Storage."""
import os
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data'
(DATA_DIR / 'params').mkdir(parents=True, exist_ok=True)
(DATA_DIR / 'stats').mkdir(parents=True, exist_ok=True)
(DATA_DIR / 'dataset').mkdir(parents=True, exist_ok=True)

try:
    from google.cloud import storage
except ImportError:
    print("Installing google-cloud-storage...")
    os.system("pip install google-cloud-storage")
    from google.cloud import storage

client = storage.Client.create_anonymous_client()
bucket = client.get_bucket('dm_graphcast')

files = [
    ('gencast/params/GenCast 1p0deg Mini <2019.npz', 'params/GenCast_1p0deg_Mini.npz'),
    ('gencast/stats/diffs_stddev_by_level.nc',        'stats/diffs_stddev_by_level.nc'),
    ('gencast/stats/mean_by_level.nc',                'stats/mean_by_level.nc'),
    ('gencast/stats/stddev_by_level.nc',              'stats/stddev_by_level.nc'),
    ('gencast/stats/min_by_level.nc',                 'stats/min_by_level.nc'),
    ('gencast/dataset/source-era5_date-2019-03-29_res-1.0_levels-13_steps-01.nc',
                                                      'dataset/era5_1p0deg_1step.nc'),
]

for gcs_path, local_rel in files:
    local = DATA_DIR / local_rel
    if local.exists():
        print(f'  already exists: {local_rel}')
        continue
    print(f'Downloading {gcs_path} ...')
    bucket.blob(gcs_path).download_to_filename(str(local))
    print(f'  -> {local_rel} ({local.stat().st_size / 1e6:.1f} MB)')

print('\nAll files ready.')
