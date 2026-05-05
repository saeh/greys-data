"""Data generation orchestration."""

import pickle
import argparse
import cProfile
import pstats
import polars as pl
from datetime import datetime, timedelta
from io import StringIO
import logging
from pathlib import Path

from data.lib.preprocessing import (
    load_parquet_data,
    build_dog_name_mapping,
    build_track_name_mapping,
    build_dog_histories,
    build_samples,
)

logger = logging.getLogger(__name__)

PARQUET_PATH = Path("data/clean/train.parquet")
CACHE_ROOT = Path("data/ml")
MAPPINGS_PATH = CACHE_ROOT / "mappings.pkl"


def cache_dir_for_date(version_date: str | None = None) -> Path:
    if version_date is None:
        version_date = datetime.utcnow().strftime("%Y-%m-%d")
    return CACHE_ROOT / version_date


def samples_cache_path(version_date: str | None = None) -> Path:
    return cache_dir_for_date(version_date) / "samples.pkl"


def load_mappings() -> tuple[dict, dict]:
    """Load dog_name_to_id and track_name_to_id from persistent mappings file."""
    if not MAPPINGS_PATH.exists():
        raise FileNotFoundError(
            f"Mappings file not found at {MAPPINGS_PATH}. Train mode must be run first."
        )
    with open(MAPPINGS_PATH, "rb") as f:
        data = pickle.load(f)
        return data["dog_name_to_id"], data["track_name_to_id"]


def save_mappings(dog_name_to_id: dict, track_name_to_id: dict) -> None:
    """Save dog_name_to_id and track_name_to_id to persistent mappings file."""
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    with open(MAPPINGS_PATH, "wb") as f:
        pickle.dump(
            {
                "dog_name_to_id": dog_name_to_id,
                "track_name_to_id": track_name_to_id,
            },
            f,
        )


def _ensure_cache_dir(version_date: str | None = None) -> Path:
    cache_dir = cache_dir_for_date(version_date)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def run_generate(mode: str = "inference", profile: bool = False, version_date: str | None = None) -> Path:
    """Run the specified generation mode and persist the result."""
    if mode == "train":
        build_func = build_all
    elif mode == "inference":
        build_func = lambda: build_today(version_date)
    else:
        raise ValueError("mode must be either 'train' or 'inference'")

    if profile:
        profiler = cProfile.Profile()
        profiler.enable()
        samples, dog_name_to_id, track_name_to_id = build_func()
        profiler.disable()

        s = StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
        ps.print_stats(30)
        print("\n" + s.getvalue())
    else:
        samples, dog_name_to_id, track_name_to_id = build_func()

    # Save mappings (train mode) and samples (both modes)
    if mode == "train":
        save_mappings(dog_name_to_id, track_name_to_id)
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        cache_path = CACHE_ROOT / "train.pkl"
        with open(cache_path, "wb") as f:
            pickle.dump({"samples": samples}, f)
    else:  # inference
        cache_dir = _ensure_cache_dir(version_date)
        cache_path = cache_dir / "inference.pkl"
        with open(cache_path, "wb") as f:
            pickle.dump({"samples": samples}, f)

    return cache_path


def generate_training_data(profile: bool = False) -> Path:
    """Generate training data from the cleaned parquet dataset."""
    return run_generate("train", profile)


def generate_inference_data(profile: bool = False) -> Path:
    """Generate daily inference data from the cleaned parquet dataset."""
    return run_generate("inference", profile)


def build_today(version_date: str | None = None):
    """Inference Path"""
    print(f"  Loading {PARQUET_PATH}... Processing Today")
    df_full = load_parquet_data(PARQUET_PATH)
    if version_date is None:
        dt_str = (datetime.utcnow() + timedelta(hours=10)).strftime("%Y-%m-%d")
    else:
        dt_str = version_date
    df_today = df_full.filter(pl.col("date") == dt_str)
    df_full = df_full.filter(pl.col("date") <= dt_str)

    dogs_today = set(df_today["dogname"].unique().to_list())

    df_today_history = df_full.filter(pl.col("dogname").is_in(dogs_today)).sort("date")
    print(
        f"  Full DF: {df_full.shape[0]}. Today Only History: {df_today_history.shape[0]}. Dogs: {len(dogs_today)}"
    )

    print("  Loading persistent mappings...")
    dog_name_to_id, track_name_to_id = load_mappings()

    print("  Building dog histories...")
    df_today_history = build_dog_histories(df_today_history, max_len=20)
    df_today = df_today_history.filter(pl.col("date") == dt_str)

    print(f"  Building samples ({len(df_today)} rows)...")
    samples = build_samples(
        df_today,
        max_seq_len=20,
        dog_name_to_id=dog_name_to_id,
        track_name_to_id=track_name_to_id,
    )

    return samples, dog_name_to_id, track_name_to_id


def build_all():
    """Training Path"""
    print(f"  Loading {PARQUET_PATH}...")
    df_full = load_parquet_data(PARQUET_PATH)

    print("  Building dog name mapping...")
    dog_name_to_id = build_dog_name_mapping(df_full)

    print("  Building track name mapping...")
    track_name_to_id = build_track_name_mapping(df_full)

    print("  Building dog histories...")
    df_full = build_dog_histories(df_full, max_len=20)

    # TODO: Add check in inference mode to see if any NEW dogs

    print(f"  Building samples ({len(df_full)} rows)...")
    samples = build_samples(
        df_full,
        max_seq_len=20,
        dog_name_to_id=dog_name_to_id,
        track_name_to_id=track_name_to_id,
    )

    return samples, dog_name_to_id, track_name_to_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess and cache dog racing data")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    parser.add_argument(
        "--mode",
        choices=["train", "inference"],
        default="inference",
        help="Mode: train or inference",
    )
    args = parser.parse_args()

    cache_path = run_generate(args.mode, args.profile)
    print(f"Caching completed to {cache_path}")


if __name__ == "__main__":
    main()
