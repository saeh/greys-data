"""Settings and configuration for greys-data."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "data" / "ml"
DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Athena Configuration
ATHENA_DATABASE = os.getenv("ATHENA_DATABASE", "default")
ATHENA_OUTPUT_LOCATION = os.getenv("ATHENA_OUTPUT_LOCATION")  # s3://bucket/path/

# S3 Configuration
S3_BUCKET = os.getenv("S3_BUCKET")

# Scheduling
DAILY_RUN_TIME = os.getenv("DAILY_RUN_TIME", "02:00")  # 24-hour format

__all__ = [
    "PROJECT_ROOT",
    "DATA_OUTPUT_DIR",
    "AWS_REGION",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "ATHENA_DATABASE",
    "ATHENA_OUTPUT_LOCATION",
    "S3_BUCKET",
    "DAILY_RUN_TIME",
]
