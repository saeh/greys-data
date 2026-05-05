"""Data module for greys-data.

Provides utilities for loading generated ML caches.
"""

import pickle
from pathlib import Path
from datetime import datetime
from typing import Any

CACHE_DIR = Path(__file__).parent / "ml"
MAPPINGS_PATH = CACHE_DIR / "mappings.pkl"


def _load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Cache file not found: {path}")

    with open(path, "rb") as f:
        return pickle.load(f)


def cache_dir_for_date(version_date: str | None = None) -> Path:
    if version_date is None:
        version_date = datetime.utcnow().strftime("%Y-%m-%d")
    return CACHE_DIR / version_date


def _latest_cache_dir() -> Path | None:
    if not CACHE_DIR.exists():
        return None

    date_dirs = [p for p in CACHE_DIR.iterdir() if p.is_dir()]
    if not date_dirs:
        return None

    return max(date_dirs, key=lambda p: p.name)


def load_mappings() -> dict[str, Any]:
    """Load persistent mappings (dog_name_to_id, track_name_to_id)."""
    return _load_cache(MAPPINGS_PATH)


def latest_samples_path() -> Path:
    cache_dir = _latest_cache_dir()
    if cache_dir is None:
        raise FileNotFoundError("No dated sample cache directories found under data/ml")

    cache_path = cache_dir / "samples.pkl"
    if not cache_path.exists():
        raise FileNotFoundError(f"Sample cache file not found: {cache_path}")

    return cache_path


def load_train_cache() -> dict[str, Any]:
    """Load the latest training samples from data/ml/YYYY-MM-DD/samples.pkl.
    
    Mappings are loaded separately via load_mappings().
    """
    return _load_cache(latest_samples_path())


def load_inference_cache() -> dict[str, Any]:
    """Load the latest inference samples from data/ml/YYYY-MM-DD/samples.pkl.
    
    Mappings are loaded separately via load_mappings().
    """
    return _load_cache(latest_samples_path())


def load_latest_cache(mode: str = "inference") -> dict[str, Any]:
    """Load the latest samples for the requested mode.
    
    Note: This only loads samples. Call load_mappings() separately for mappings.
    """
    if mode in {"train", "inference"}:
        return _load_cache(latest_samples_path())
    raise ValueError("mode must be 'train' or 'inference'")


def cache_path(mode: str = "inference", version_date: str | None = None) -> Path:
    """Return the samples cache path for the requested mode and optional version date."""
    if mode not in {"train", "inference"}:
        raise ValueError("mode must be 'train' or 'inference'")
    return cache_dir_for_date(version_date) / "samples.pkl"


__all__ = [
    "load_mappings",
    "load_train_cache",
    "load_inference_cache",
    "load_latest_cache",
    "cache_path",
    "cache_dir_for_date",
]

