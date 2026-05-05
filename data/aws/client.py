"""AWS Athena and S3 client for data fetching."""

import boto3
from botocore.exceptions import ClientError
import polars as pl
from pathlib import Path
import time
from config.settings import (
    AWS_REGION,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    ATHENA_DATABASE,
    ATHENA_OUTPUT_LOCATION,
)


class AthenaS3Client:
    """Client for querying Athena and downloading results from S3."""

    def __init__(self):
        """Initialize AWS clients."""
        self.athena = boto3.client(
            "athena",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )
        self.s3 = boto3.client(
            "s3",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )

    def execute_query(self, query: str, database: str = ATHENA_DATABASE) -> str:
        """
        Execute Athena query and return execution ID.

        Args:
            query: SQL query to execute
            database: Athena database name

        Returns:
            Query execution ID

        Raises:
            ClientError: If query execution fails
        """
        try:
            response = self.athena.start_query_execution(
                QueryString=query,
                QueryExecutionContext={"Database": database},
                ResultConfiguration={"OutputLocation": ATHENA_OUTPUT_LOCATION},
            )
            return response["QueryExecutionId"]
        except ClientError as e:
            raise RuntimeError(f"Failed to execute query: {e}")

    def wait_for_query_completion(self, query_execution_id: str, max_wait: int = 300) -> dict:
        """
        Wait for Athena query to complete.

        Args:
            query_execution_id: Execution ID from start_query_execution
            max_wait: Maximum wait time in seconds

        Returns:
            Query execution details

        Raises:
            TimeoutError: If query doesn't complete within max_wait
        """
        elapsed = 0
        interval = 2  # Check every 2 seconds

        while elapsed < max_wait:
            response = self.athena.get_query_execution(QueryExecutionId=query_execution_id)
            status = response["QueryExecution"]["Status"]["State"]

            if status == "SUCCEEDED":
                return response["QueryExecution"]
            elif status in ("FAILED", "CANCELLED"):
                reason = response["QueryExecution"]["Status"].get("StateChangeReason", "Unknown")
                raise RuntimeError(f"Query {status}: {reason}")

            time.sleep(interval)
            elapsed += interval

        raise TimeoutError(f"Query {query_execution_id} did not complete within {max_wait} seconds")

    def fetch_query_results(self, query_execution_id: str) -> pl.DataFrame:
        """
        Fetch query results from S3 and return as Polars DataFrame.

        Args:
            query_execution_id: Execution ID of completed query

        Returns:
            Polars DataFrame with query results
        """
        response = self.athena.get_query_execution(QueryExecutionId=query_execution_id)
        output_location = response["QueryExecution"]["ResultConfiguration"]["OutputLocation"]

        # Parse S3 path
        if output_location.startswith("s3://"):
            output_location = output_location[5:]
        bucket, key = output_location.split("/", 1)

        try:
            obj = self.s3.get_object(Bucket=bucket, Key=key)
            df = pl.read_csv(obj["Body"])
            return df
        except ClientError as e:
            raise RuntimeError(f"Failed to fetch query results: {e}")

    def query_and_fetch(self, query: str, database: str = ATHENA_DATABASE) -> pl.DataFrame:
        """
        Execute query and fetch results in one operation.

        Args:
            query: SQL query to execute
            database: Athena database name

        Returns:
            Polars DataFrame with query results
        """
        query_id = self.execute_query(query, database)
        self.wait_for_query_completion(query_id)
        return self.fetch_query_results(query_id)


__all__ = ["AthenaS3Client"]
