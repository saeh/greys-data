"""Tests for data fetching and generation."""

import pytest
from pathlib import Path
from datetime import date
import polars as pl
from data import load_latest_data, get_latest_version, list_versions


def test_data_module_imports():
    """Test that data module imports correctly."""
    from data import (
        load_latest_data,
        load_data_by_date,
        get_latest_version,
        list_versions,
    )
    assert callable(load_latest_data)
    assert callable(load_data_by_date)
    assert callable(get_latest_version)
    assert callable(list_versions)


def test_config_imports():
    """Test that config module imports correctly."""
    from config import DATASETS, get_dataset, DATA_OUTPUT_DIR
    assert "form" in DATASETS
    assert "results" in DATASETS
    assert callable(get_dataset)
    assert isinstance(DATA_OUTPUT_DIR, Path)


def test_aws_client_imports():
    """Test that AWS client imports correctly."""
    from data.aws import AthenaS3Client
    assert callable(AthenaS3Client)


def test_scheduler_imports():
    """Test that scheduler module imports correctly."""
    from scheduler import DailyScheduler
    assert callable(DailyScheduler)


def test_generate_imports():
    """Test that generate module imports correctly."""
    from data.generate import generate_daily_data
    assert callable(generate_daily_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
