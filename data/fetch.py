"""Fetch data from Athena/S3 for both datasets."""

import polars as pl
from datetime import datetime
from pathlib import Path
import time
import boto3
from data.aws import AthenaS3Client
from config.datasets import DATASETS, get_dataset
from config import DATA_OUTPUT_DIR

# Legacy constants for compatibility
DATABASE = "doger"
REGION = "ap-southeast-2"
OUTPUT_BUCKET = "aws-athena-query-results-522980072010-ap-southeast-2"

CURRENT_YEAR_WHERE = """
where date_format(date_parse( fp.date ,'%d %b %y'),'%Y-%m-%d') >= concat(date_format(current_date,'%Y'),'-01-01')
"""

PAST_YEARS_WHERE = """
where date_format(date_parse( fp.date ,'%d %b %y'),'%Y-%m-%d') < concat(date_format(current_date,'%Y'),'-01-01')
"""

RESULTS_CURRENT_YEAR_WHERE = """
where date_format(datetime_local,'%Y-%m-%d') >= concat(date_format(current_date,'%Y'),'-01-01')
"""

RESULTS_PAST_YEARS_WHERE = """
where date_format(datetime_local,'%Y-%m-%d') < concat(date_format(current_date,'%Y'),'-01-01')
"""


athena = boto3.client("athena", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)


def run_query(query: str) -> str:
    response = athena.start_query_execution(
        QueryString=query,
        ResultConfiguration={"OutputLocation": f"s3://{OUTPUT_BUCKET}"},
    )
    query_id = response["QueryExecutionId"]

    while True:
        status = athena.get_query_execution(QueryExecutionId=query_id)
        state = status["QueryExecution"]["Status"]["State"]
        if state == "SUCCEEDED":
            break
        elif state == "FAILED":
            raise Exception(
                f"Query failed: {status['QueryExecution']['Status']['StateChangeReason']}"
            )
        time.sleep(0.5)

    result = athena.get_query_execution(QueryExecutionId=query_id)
    return result["QueryExecution"]["ResultConfiguration"]["OutputLocation"]


def download_s3(output_url: str, output_path: Path) -> Path:
    output_path = Path(output_path)
    key = output_url.split(f"s3://{OUTPUT_BUCKET}/")[-1]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        s3.download_fileobj(OUTPUT_BUCKET, key, f)

    return output_path


def query_to_csv(query: str, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    print("Running query...")
    output_url = run_query(query)
    print("Downloading from S3...")
    return download_s3(output_url, output_path)


def fetch_form_data(past: bool = False, output_path: str | None = None) -> Path:
    dataset_config = get_dataset("form")
    where_clause = PAST_YEARS_WHERE if past else CURRENT_YEAR_WHERE
    query = f"{dataset_config.query} {where_clause} order by datetime_utc, racenum, racebox"

    suffix = "past" if past else "current"
    output_path = output_path or f"data/aws/{dataset_config.name}_{suffix}.csv"
    return query_to_csv(query, Path(output_path))


def fetch_results(past: bool = False, output_path: str | None = None) -> Path:
    dataset_config = get_dataset("results")
    where_clause = RESULTS_PAST_YEARS_WHERE if past else RESULTS_CURRENT_YEAR_WHERE
    query = f"{dataset_config.query} {where_clause} order by datetime_local, racenum"

    suffix = "past" if past else "current"
    output_path = output_path or f"data/aws/{dataset_config.name}_{suffix}.csv"
    return query_to_csv(query, Path(output_path))


def fetch_all_data(past: bool = False) -> dict[str, Path]:
    """Fetch both form and results datasets and return the output file paths."""
    return {
        "form": fetch_form_data(past=past),
        "results": fetch_results(past=past),
    }


def fetch_dataset(dataset_name: str) -> pl.DataFrame:
    """
    Fetch a single dataset from Athena.

    Args:
        dataset_name: Name of dataset ('form' or 'results')

    Returns:
        Polars DataFrame with fetched data

    Raises:
        ValueError: If dataset not found
    """
    dataset_config = get_dataset(dataset_name)
    client = AthenaS3Client()

    print(f"Fetching {dataset_name} dataset...")
    df = client.query_and_fetch(dataset_config.query)
    print(f"Successfully fetched {len(df)} rows for {dataset_name}")

    return df


def fetch_all_datasets() -> dict[str, pl.DataFrame]:
    """
    Fetch both datasets from Athena.

    Returns:
        Dictionary with dataset names as keys and DataFrames as values
    """
    results = {}
    for dataset_name in DATASETS.keys():
        results[dataset_name] = fetch_dataset(dataset_name)
    return results


def save_datasets(datasets: dict[str, pl.DataFrame], version_date: str | None = None) -> dict[str, Path]:
    """
    Save datasets to versioned output directory.

    Args:
        datasets: Dictionary of dataset names to DataFrames
        version_date: Date string for versioning (YYYY-MM-DD). Defaults to today.

    Returns:
        Dictionary of dataset names to output file paths
    """
    if version_date is None:
        version_date = datetime.now().strftime("%Y-%m-%d")

    version_dir = DATA_OUTPUT_DIR / version_date
    version_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {}
    for dataset_name, df in datasets.items():
        dataset_config = get_dataset(dataset_name)
        output_path = version_dir / dataset_config.output_file

        df.write_parquet(output_path)
        output_paths[dataset_name] = output_path
        print(f"Saved {dataset_name} to {output_path}")

    return output_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch dog racing data from Athena")
    parser.add_argument("--dataset", choices=["form", "results"], default="form")
    parser.add_argument(
        "--past",
        action="store_true",
        help="Fetch past years data (default: current year)",
    )
    args = parser.parse_args()

    suffix = "past" if args.past else "current"
    output_path = f"data/aws/{args.dataset}_{suffix}.csv"

    print(f"Fetching {args.dataset} data ({suffix})...")

    if args.dataset == "form":
        path = fetch_form_data(past=args.past, output_path=output_path)
    elif args.dataset == "results":
        path = fetch_results(past=args.past, output_path=output_path)
    else:
        raise ValueError("dataset must be 'form' or 'results'")

    print(f"Saved to {path}")
