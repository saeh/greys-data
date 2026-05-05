"""CLI entry point for greys-data."""

import typer
import logging
from pathlib import Path

from data.fetch import fetch_form_data, fetch_results, fetch_all_data
from data.clean import clean as clean_data
from data.generate import generate_inference_data, run_generate
from scheduler import DailyScheduler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

app = typer.Typer(help="Data generation pipeline for greys projects")


@app.command()
def fetch(
    dataset: str = typer.Option("form", "--dataset", help="Dataset to fetch: form or results"),
    past: bool = typer.Option(False, "--past", help="Fetch past years data instead of current year"),
) -> None:
    """Fetch raw Athena data for a dataset."""
    try:
        if dataset == "form":
            path = fetch_form_data(past=past)
        elif dataset == "results":
            path = fetch_results(past=past)
        else:
            raise typer.BadParameter("dataset must be form or results")

        typer.echo(f"✓ Fetched {dataset} data to {path}")
    except Exception as e:
        typer.echo(f"✗ Fetch failed: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def fetch_all(
    past: bool = typer.Option(False, "--past", help="Fetch past years data instead of current year"),
) -> None:
    """Fetch both form and results datasets."""
    try:
        paths = fetch_all_data(past=past)
        typer.echo("✓ Fetched both datasets:")
        for name, path in paths.items():
            typer.echo(f"  - {name}: {path}")
    except Exception as e:
        typer.echo(f"✗ Fetch-all failed: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def clean() -> None:
    """Clean raw Athena data and build train.parquet."""
    try:
        clean_data()
        typer.echo("✓ Clean completed and data/clean/train.parquet was written")
    except Exception as e:
        typer.echo(f"✗ Clean failed: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def generate(
    mode: str = typer.Option("inference", "--mode", help="Mode: inference or train"),
    profile: bool = typer.Option(False, "--profile", help="Enable profiling"),
) -> None:
    """Generate ML data from cleaned dataset."""
    try:
        typer.echo(f"Generating {mode} data...")
        cache_path = run_generate(mode, profile)
        typer.echo(f"✓ Generated {mode} data to {cache_path}")
    except Exception as e:
        typer.echo(f"✗ Generate failed: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def daily(
    profile: bool = typer.Option(False, "--profile", help="Enable profiling for the inference generation step"),
) -> None:
    """Run the full daily pipeline: fetch current raw data, clean, and generate inference output."""
    try:
        typer.echo("Running daily pipeline: fetch current form+results, clean, generate inference")
        paths = fetch_all_data(past=False)
        for name, path in paths.items():
            typer.echo(f"  fetched {name}: {path}")

        clean_data()
        typer.echo("  cleaned raw data")

        cache_path = generate_inference_data(profile)
        typer.echo(f"✓ Daily inference data generated to {cache_path}")
    except Exception as e:
        typer.echo(f"✗ Daily pipeline failed: {e}", err=True, exc_info=True)
        raise typer.Exit(1)


@app.command()
def schedule(
    time: str = typer.Option("02:00", "--time", help="Time to run daily (HH:MM, 24-hour format)"),
    start: bool = typer.Option(True, "--start/--no-start", help="Start scheduler immediately"),
) -> None:
    """Schedule daily data generation."""
    try:
        scheduler = DailyScheduler()
        scheduler.schedule_daily(time)

        if start:
            typer.echo(f"Scheduling daily data generation at {time}...")
            scheduler.start()
            typer.echo("✓ Scheduler started. Press Ctrl+C to stop.")
        else:
            typer.echo(f"✓ Scheduled for {time}. Run with --start to activate.")

    except Exception as e:
        typer.echo(f"✗ Scheduling failed: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def status() -> None:
    """Show data generation status for the current pipeline."""
    project_root = Path(__file__).resolve().parent
    cache_base = project_root / "data" / "ml"
    clean_path = project_root / "data" / "clean" / "train.parquet"
    mappings_path = cache_base / "mappings.pkl"

    typer.echo(f"Cache base directory: {cache_base}")
    typer.echo(f"Clean dataset: {clean_path}")
    typer.echo()

    # Show persistent mappings
    if mappings_path.exists():
        size_mb = mappings_path.stat().st_size / (1024 * 1024)
        typer.echo(f"✓ Persistent mappings: {mappings_path.name} ({size_mb:.2f} MB)")
    else:
        typer.echo("✗ No persistent mappings file. Run 'generate --mode train' first.")
    typer.echo()

    # Show dated sample caches
    if not cache_base.exists():
        typer.echo("No dated sample cache directories found yet.")
        return

    dated_dirs = sorted([p for p in cache_base.iterdir() if p.is_dir()])
    if not dated_dirs:
        typer.echo("No dated sample cache directories found yet.")
        return

    typer.echo(f"Dated sample caches ({len(dated_dirs)}):")
    for dated_dir in dated_dirs:
        sample_files = sorted(dated_dir.glob("samples.pkl"))
        typer.echo(f"  - {dated_dir.name} ({len(sample_files)} file)")
        for file in sample_files:
            size_mb = file.stat().st_size / (1024 * 1024)
            typer.echo(f"      * {file.name} ({size_mb:.2f} MB)")

    latest_dir = dated_dirs[-1]
    typer.echo(f"Latest sample cache version: {latest_dir.name}")


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
