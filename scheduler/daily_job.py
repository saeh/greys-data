"""Scheduler for daily data generation using APScheduler."""

import logging
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from data.fetch import fetch_all_data
from data.clean import clean as clean_data
from data.generate import generate_inference_data

logger = logging.getLogger(__name__)


class DailyScheduler:
    """Manages daily data generation scheduling."""

    def __init__(self):
        """Initialize the scheduler."""
        self.scheduler = BackgroundScheduler()
        self.job = None

    def schedule_daily(self, time: str = "02:00") -> None:
        """
        Schedule daily data generation.

        Args:
            time: Time in HH:MM format (24-hour). Default: 02:00
        """
        try:
            hour, minute = map(int, time.split(":"))
            if not (0 <= hour < 24 and 0 <= minute < 60):
                raise ValueError("Invalid time format")
        except (ValueError, IndexError):
            raise ValueError(f"Invalid time format: {time}. Use HH:MM (24-hour format)")

        # Remove existing job if any
        if self.job:
            self.scheduler.remove_job(self.job.id)

        # Schedule the job
        self.job = self.scheduler.add_job(
            func=self._run_generation,
            trigger=CronTrigger(hour=hour, minute=minute),
            id="daily_data_generation",
            name="Daily data generation",
            misfire_grace_time=3600,  # Allow 1 hour for missed runs
        )

        logger.info(f"Scheduled daily data generation at {time}")

    def start(self) -> None:
        """Start the scheduler."""
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("Scheduler started")

    def stop(self) -> None:
        """Stop the scheduler."""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=True)
            logger.info("Scheduler stopped")

    def _run_generation(self) -> None:
        """Internal method to run the full daily pipeline."""
        try:
            logger.info("Starting scheduled daily pipeline...")
            paths = fetch_all_data(past=False)
            logger.info(f"Fetched raw datasets: {paths}")

            clean_data()
            logger.info("Cleaned raw data into data/clean/train.parquet")

            cache_path = generate_inference_data()
            logger.info(f"Generated inference data to {cache_path}")
        except Exception as e:
            logger.error(f"Scheduled data generation failed: {e}", exc_info=True)


__all__ = ["DailyScheduler"]
