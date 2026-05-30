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


def _find_latest_file(filename: str) -> Path:
    if not CACHE_DIR.exists():
        raise FileNotFoundError(f"No cache directory: {CACHE_DIR}")

    candidates = []
    for date_dir in CACHE_DIR.iterdir():
        if date_dir.is_dir():
            file_path = date_dir / filename
            if file_path.exists():
                candidates.append(file_path)

    if not candidates:
        raise FileNotFoundError(f"No {filename} files found under {CACHE_DIR}")

    # Return the one with the latest modification time
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_mappings() -> dict[str, Any]:
    """Load persistent mappings (dog_name_to_id, track_name_to_id)."""
    return _load_cache(MAPPINGS_PATH)


def load_train_cache() -> dict[str, Any]:
    """Load the latest training samples from the most recent train.pkl file.
    
    Mappings are loaded separately via load_mappings().
    """
    return _load_cache(_find_latest_file("train.pkl"))


def load_inference_cache() -> dict[str, Any]:
    """Load the latest inference samples from the most recent inference.pkl file.
    
    Mappings are loaded separately via load_mappings().
    """
    return _load_cache(_find_latest_file("inference.pkl"))


def load_latest_cache(mode: str = "inference") -> dict[str, Any]:
    """Load the latest samples for the requested mode.
    
    Note: This only loads samples. Call load_mappings() separately for mappings.
    """
    filename = "train.pkl" if mode == "train" else "inference.pkl"
    return _load_cache(_find_latest_file(filename))


def cache_path(mode: str = "inference", version_date: str | None = None) -> Path:
    """Return the cache path for the requested mode and optional version date."""
    if mode not in {"train", "inference"}:
        raise ValueError("mode must be 'train' or 'inference'")
    filename = "train.pkl" if mode == "train" else "inference.pkl"
    return cache_dir_for_date(version_date) / filename


__all__ = [
    "load_mappings",
    "load_train_cache",
    "load_inference_cache",
    "load_latest_cache",
    "cache_path",
    "cache_dir_for_date",
]

