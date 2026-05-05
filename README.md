# greys-data

Data generation pipeline for greys projects. Fetches market data from AWS Athena, processes it, and outputs versioned datasets for consumption by model training and prediction services.

## Project Structure

```
greys-data/
├── config/                 # Configuration and dataset definitions
│   ├── settings.py        # AWS and app settings
│   └── datasets.py        # Dataset configurations
├── data/                  # Core data processing
│   ├── __init__.py        # Consumer API (for sibling projects)
│   ├── fetch.py           # Athena/S3 data fetching
│   ├── clean.py           # Data validation and cleaning
│   ├── generate.py        # Orchestration
│   ├── aws/
│   │   └── client.py      # AWS Athena/S3 client
│   └── lib/               # Preprocessing utilities
├── outputs/
│   └── data/ml/           # Versioned outputs (date-organized)
├── scheduler/             # Daily scheduling
├── scripts/               # Utility scripts
├── tests/                 # Unit tests
├── main.py                # CLI entry point
└── pyproject.toml         # Dependencies and project metadata
```

## Setup

### 1. Install Dependencies

```bash
pip install -e .
```

### 2. Configure AWS Credentials

Copy `.env.example` to `.env` and fill in your AWS credentials:

```bash
cp .env.example .env
# Edit .env with your AWS credentials
```

Required environment variables:
- `AWS_REGION` - AWS region (default: us-east-1)
- `AWS_ACCESS_KEY_ID` - AWS access key
- `AWS_SECRET_ACCESS_KEY` - AWS secret key
- `ATHENA_DATABASE` - Athena database name (default: default)
- `ATHENA_OUTPUT_LOCATION` - S3 path for Athena results (e.g., `s3://bucket/path/`)
- `DAILY_RUN_TIME` - Time for daily runs in HH:MM format (default: 02:00)

### 3. Update Dataset Queries

Edit `config/datasets.py` to set your actual Athena queries for the `current` and `past` datasets.

## Usage

### CLI Commands

#### Generate data immediately

```bash
python main.py generate
# Or for a specific date:
python main.py generate --date 2026-05-04
```

#### Schedule daily generation

```bash
# Schedule for 2:00 AM and start scheduler
python main.py schedule --time 02:00

# Schedule without starting (use cron or systemd instead)
python main.py schedule --time 02:00 --no-start
```

#### Check status

```bash
python main.py status
```

### Consuming Data (From Sibling Projects)

#### Option 1: Direct Import API

```python
from pathlib import Path

# Add parent directory to path (or install as package)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "greys-data"))

from data import load_latest_data

# Load today's data
df_current = load_latest_data("current")
df_past = load_latest_data("past")

# Work with DataFrames...
```

#### Option 2: File Path

```python
from pathlib import Path
from datetime import date
import polars as pl

greys_data_dir = Path(__file__).parent.parent / "greys-data"
data_dir = greys_data_dir / "outputs" / "data" / "ml" / str(date.today())

df_current = pl.read_parquet(data_dir / "dataset_current.parquet")
df_past = pl.read_parquet(data_dir / "dataset_past.parquet")
```

### Data Versioning

Data is automatically versioned by date in `outputs/data/ml/`:

```
outputs/data/ml/
├── 2026-05-04/
│   ├── dataset_current.parquet
│   └── dataset_past.parquet
├── 2026-05-05/
│   ├── dataset_current.parquet
│   └── dataset_past.parquet
└── ...
```

This allows for:
- Historical analysis: `data.load_data_by_date("current", "2026-05-03")`
- Version tracking: `data.list_versions()` returns all available dates
- Easy rollback: Just specify the date you need

## Architecture

### Data Flow

```
Athena Queries
     ↓
AWS S3 Results
     ↓
Fetch (data/fetch.py)
     ↓
Validate (data/clean.py)
     ↓
Save Versioned (data/fetch.py)
     ↓
outputs/data/ml/{date}/
```

### Daily Execution

The scheduler runs at the configured time (default: 2:00 AM) and:

1. **Fetches** both datasets from Athena
2. **Validates** data quality
3. **Saves** to versioned output directory
4. **Logs** results

Failures are logged for investigation.

### Scheduling Options

#### 1. APScheduler (Built-in)

```bash
python main.py schedule --time 02:00
```

Runs in foreground. Use `nohup` or systemd for background execution.

#### 2. Cron

Add to crontab:

```bash
0 2 * * * cd /path/to/greys-data && python -c "from data import generate_daily_data; generate_daily_data()" >> /var/log/greys-data.log 2>&1
```

#### 3. Systemd Timer

Create `/etc/systemd/system/greys-data.service`:

```ini
[Unit]
Description=Greys Data Generation
After=network.target

[Service]
Type=oneshot
WorkingDirectory=/path/to/greys-data
ExecStart=/path/to/greys-data/.venv/bin/python main.py generate
User=youruser
```

Create `/etc/systemd/system/greys-data.timer`:

```ini
[Unit]
Description=Greys Data Generation Timer
Requires=greys-data.service

[Timer]
OnCalendar=*-*-* 02:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

Enable:

```bash
sudo systemctl enable greys-data.timer
sudo systemctl start greys-data.timer
```

## Development

### Running Tests

```bash
pytest tests/
```

### Project Layout Philosophy

- **Single Responsibility**: Each module has one clear purpose
- **Consumer-First**: `data/__init__.py` is the public API
- **Versioned Outputs**: Easy to track changes and rollback
- **Explicit Configuration**: All settings in `config/` module
- **Reusable Components**: AWS client, fetch, validate, save can be used independently

## Troubleshooting

### Data Not Generated

1. Check AWS credentials in `.env`
2. Verify Athena queries in `config/datasets.py`
3. Ensure S3 output location exists and is writable
4. Check logs: `python main.py generate --verbose`

### Scheduler Not Running

1. Check if `main.py schedule` process is still running
2. Verify time format (24-hour)
3. Check logs for errors

### Import Errors from Sibling Projects

Ensure path setup:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "greys-data"))
```

Or install greys-data as an editable package in each sibling project.

## Contributing

When adding new features:

1. Add configuration to `config/` if needed
2. Add processing logic to appropriate `data/` module
3. Update `data/__init__.py` API if consumer-facing
4. Add tests to `tests/`
5. Update README documentation

## License

[Add your license here]
