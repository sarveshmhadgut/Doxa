import sys
import boto3
import pandas as pd
from io import StringIO
from src.logger import logging
from src.exception import MyException


class S3Operations:
    """
    Wrapper around basic S3 operations required by the pipeline.

    Currently supports:
        - Fetching a CSV file from S3 into a pandas DataFrame.
    """

    def __init__(
        self,
        bucket_name: str,
        aws_access_key: str,
        aws_secret_key: str,
        aws_region: str = "us-east-1",
    ) -> None:
        """
        Initialize the S3 client with provided credentials and bucket.

        Args:
            bucket_name (str): Name of the S3 bucket.
            aws_access_key (str): AWS access key ID.
            aws_secret_key (str): AWS secret access key.
            aws_region (str): AWS region for the S3 client. Defaults to "us-east-1".
        """
        try:
            self.bucket_name: str = bucket_name
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region,
            )

        except Exception as e:
            raise MyException(e, sys) from e

    def fetch_csv_from_s3(self, file_key: str) -> pd.DataFrame:
        """
        Fetch a CSV file from S3 and load it into a pandas DataFrame.

        Args:
            file_key (str): Object key (path) of the CSV file in the S3 bucket.

        Returns:
            pd.DataFrame: DataFrame containing the loaded CSV data.

        Raises:
            MyException: If the S3 read or CSV parsing fails.
        """
        try:
            logging.info("Fetching file from S3 bucket...")

            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_key)
            body_bytes: bytes = obj["Body"].read()
            csv_text: str = body_bytes.decode("utf-8")

            df: pd.DataFrame = pd.read_csv(StringIO(csv_text))
            logging.info("Fetched and loaded CSV from S3 ")

            return df

        except Exception as e:
            raise MyException(e, sys) from e
