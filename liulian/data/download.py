"""Dataset auto-download — fetch missing benchmark datasets on demand.

This module provides :func:`ensure_dataset`, called automatically by
:func:`~liulian.pipeline.build_dataset` before constructing any dataset
object.  If the required data files are missing from the local
``dataset/`` directory, they are downloaded from known public URLs.

Download sources
~~~~~~~~~~~~~~~~
All standard TSL benchmark datasets (traffic, electricity, exchange_rate,
weather, illness, ETT, PEMS) are available as a single ZIP archive from
the Autoformer GitHub repository.  Individual datasets can also be
downloaded from HuggingFace via the ``datasets`` library (optional).

No forced dependencies — only ``urllib.request`` (stdlib) is used by
default.  If the ``datasets`` library is installed, it can be used as an
alternative download backend.

Usage::

    from liulian.data.download import ensure_dataset

    ensure_dataset('traffic')       # Downloads if missing
    ensure_dataset('PEMS03')        # Downloads if missing
    ensure_dataset('swiss-river-1990')  # No-op (manual download needed)
"""

from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

# ── Registry ────────────────────────────────────────────────────────────
#
# Maps dataset names → (relative_dir, required_file, download_key).
# ``download_key`` groups datasets sharing the same archive.
#
# The bulk archive from the Autoformer repo contains all standard TSL
# benchmarks in a single ~300 MB ZIP.

_BULK_ARCHIVE_URL = 'https://drive.google.com/uc?export=download&id=1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP'

# Individual dataset download URLs (fallback / convenience).
# These are direct links from well-known academic repositories.
_INDIVIDUAL_URLS: Dict[str, str] = {
    # CSV benchmarks — from the Autoformer / PatchTST data repos
    'traffic': 'https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/traffic/traffic.txt',
    'electricity': 'https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/electricity/electricity.txt',
}

# Required paths relative to PROJECT_ROOT for existence checks.
_DATASET_FILES: Dict[str, tuple[str, str]] = {
    # data_name → (subdir under dataset/, filename)
    'traffic': ('dataset/traffic', 'traffic.csv'),
    'electricity': ('dataset/electricity', 'electricity.csv'),
    'exchange_rate': ('dataset/exchange_rate', 'exchange_rate.csv'),
    'weather': ('dataset/weather', 'weather.csv'),
    'illness': ('dataset/illness', 'national_illness.csv'),
    'ETTh1': ('dataset/ETT-small', 'ETTh1.csv'),
    'ETTh2': ('dataset/ETT-small', 'ETTh2.csv'),
    'ETTm1': ('dataset/ETT-small', 'ETTm1.csv'),
    'ETTm2': ('dataset/ETT-small', 'ETTm2.csv'),
    'PEMS03': ('dataset/PEMS', 'PEMS03.npz'),
    'PEMS04': ('dataset/PEMS', 'PEMS04.npz'),
    'PEMS07': ('dataset/PEMS', 'PEMS07.npz'),
    'PEMS08': ('dataset/PEMS', 'PEMS08.npz'),
}

# Swiss-river datasets are custom and must be prepared manually.
_MANUAL_DATASETS = {'swiss-river-1990', 'swiss-river-2010', 'swiss-river-zurich'}


def _project_root() -> Path:
    """Resolve the project root directory."""
    from liulian.config import PROJECT_ROOT

    return Path(PROJECT_ROOT)


def _file_exists(data_name: str) -> bool:
    """Check whether the dataset's required file exists on disk."""
    entry = _DATASET_FILES.get(data_name)
    if entry is None:
        return True  # Unknown dataset — assume present
    subdir, filename = entry
    path = _project_root() / subdir / filename
    return path.is_file()


def _try_datasets_lib(data_name: str) -> bool:
    """Attempt to download via the ``datasets`` library (HuggingFace).

    Returns ``True`` if the download succeeded, ``False`` otherwise.
    This is an *optional* path — ``datasets`` is NOT a required
    dependency.
    """
    try:
        import datasets  # type: ignore[import-untyped]  # noqa: F811
    except ImportError:
        return False

    # Map data names to known HuggingFace dataset IDs.
    _HF_MAP: Dict[str, str] = {
        'ETTh1': 'ett',
        'ETTh2': 'ett',
        'ETTm1': 'ett',
        'ETTm2': 'ett',
    }
    hf_id = _HF_MAP.get(data_name)
    if hf_id is None:
        return False

    try:
        logger.info('Attempting download via `datasets` library: %s', hf_id)
        ds = datasets.load_dataset(hf_id, data_name, trust_remote_code=True)
        # Save to expected location
        entry = _DATASET_FILES.get(data_name)
        if entry is None:
            return False
        subdir, filename = entry
        target = _project_root() / subdir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        ds['train'].to_csv(str(target), index=False)  # type: ignore[union-attr]
        logger.ok('Downloaded via datasets lib → %s', target)  # type: ignore[attr-defined]
        return True
    except Exception as exc:
        logger.debug('datasets lib download failed: %s', exc)
        return False


def _download_file(url: str, dest: Path, *, desc: str = '') -> None:
    """Download a single file using ``urllib`` (no extra dependencies).

    Shows a progress bar using ``tqdm`` if available, otherwise prints
    percentage updates.
    """
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + '.tmp')

    logger.info('Downloading %s → %s', desc or url, dest)

    try:
        # Try with tqdm progress bar
        import tqdm  # type: ignore[import-untyped]

        with urllib.request.urlopen(url) as resp:
            total = int(resp.headers.get('Content-Length', 0))
            with (
                open(tmp, 'wb') as fh,
                tqdm.tqdm(total=total, unit='B', unit_scale=True, desc=desc or 'download') as pbar,
            ):
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    fh.write(chunk)
                    pbar.update(len(chunk))
    except ImportError:
        # Fallback: plain urllib without progress bar
        urllib.request.urlretrieve(url, str(tmp))

    shutil.move(str(tmp), str(dest))
    logger.info('Download complete: %s', dest)


def _download_bulk_archive(project_root: Path) -> bool:
    """Download the bulk TSL dataset archive and extract it.

    The archive contains all standard benchmark datasets in a single ZIP.
    Returns ``True`` on success.
    """
    archive_path = project_root / 'dataset' / 'time-series-dataset.zip'

    # If the archive already exists, try extracting it
    if archive_path.is_file():
        logger.info('Found existing archive: %s', archive_path)
    else:
        try:
            _download_file(
                _BULK_ARCHIVE_URL,
                archive_path,
                desc='TSL benchmark datasets (ZIP)',
            )
        except Exception as exc:
            logger.warning('Bulk archive download failed: %s', exc)
            return False

    # Extract
    try:
        with zipfile.ZipFile(str(archive_path), 'r') as zf:
            zf.extractall(str(project_root / 'dataset'))
        logger.info('Extracted archive to dataset/')
        return True
    except Exception as exc:
        logger.warning('Archive extraction failed: %s', exc)
        return False


def ensure_dataset(data_name: str) -> None:
    """Ensure the dataset files exist locally, downloading if necessary.

    Called automatically by :func:`~liulian.pipeline.build_dataset`.

    Strategy:
        1. If files already exist → no-op.
        2. If a bulk ZIP archive exists in ``dataset/`` → extract it.
        3. Try the ``datasets`` library (HuggingFace) if installed.
        4. Try downloading the bulk archive from Google Drive.
        5. If all methods fail → raise a clear error with instructions.

    Args:
        data_name: Dataset identifier (e.g. ``'traffic'``, ``'PEMS03'``).

    Raises:
        FileNotFoundError: When the dataset cannot be found or downloaded.
    """
    # Swiss-river and unknown datasets: skip
    if data_name.startswith('swiss-river') or data_name in _MANUAL_DATASETS:
        return
    if data_name not in _DATASET_FILES:
        return  # Unknown dataset — let build_dataset handle the error

    # Already present
    if _file_exists(data_name):
        return

    root = _project_root()
    entry = _DATASET_FILES[data_name]
    subdir, filename = entry
    target = root / subdir / filename

    logger.info('Dataset %r not found at %s — attempting auto-download.', data_name, target)

    # Strategy 1: Check if bulk archive already exists and extract
    archive_path = root / 'dataset' / 'time-series-dataset.zip'
    if archive_path.is_file():
        logger.info('Found existing archive — extracting...')
        if _download_bulk_archive(root) and _file_exists(data_name):
            logger.ok('Dataset %s ready (from existing archive).', data_name)  # type: ignore[attr-defined]
            return

    # Strategy 2: Try datasets library (optional)
    if _try_datasets_lib(data_name):
        return

    # Strategy 3: Download bulk archive from remote
    logger.info('Downloading bulk dataset archive...')
    if _download_bulk_archive(root) and _file_exists(data_name):
        logger.ok('Dataset %s ready (from bulk download).', data_name)  # type: ignore[attr-defined]
        return

    # All strategies failed
    raise FileNotFoundError(
        f'Dataset {data_name!r} not found at {target} and auto-download failed.\n'
        f'Please download the TSL benchmark datasets manually:\n'
        f'  1. Download from: https://drive.google.com/drive/folders/'
        f'1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy\n'
        f'  2. Extract into: {root / "dataset"}\n'
        f'  3. Or install `pip install datasets` for HuggingFace downloads.'
    )
