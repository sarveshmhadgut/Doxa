import os
import sys
import pandas as pd
from halo import Halo
from src.logger import logging
from dotenv import load_dotenv
from typing import Optional, Tuple
from src.exception import MyException
from src.configuration import aws_connection
from sklearn.model_selection import train_test_split
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifacts
from src.utils.main_utils import save_df_as_csv, read_yaml_file
from src.constants import (
    DATASET_BUCKET_NAME,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_REGION,
    AWS_CSV_FILENAME,
)

load_dotenv()
pd.set_option("future.no_silent_downcasting", True)


class DataIngestion:
    """
    Orchestrates data ingestion for the IMDb sentiment analysis project.

    Responsibilities:
        - Load ingestion hyperparameters from params.yaml (if config not passed explicitly).
        - Fetch dataset from S3.
        - Apply primitive preprocessing (duplicates removal, label filtering).
        - Split into train/test sets.
        - Persist splits to disk and return artifact metadata.
    """

    def __init__(
        self, data_ingestion_config: Optional[DataIngestionConfig] = None
    ) -> None:
        """
        Initialize DataIngestion with configuration.

        If no config object is provided, the configuration is read from params.yaml
        under the 'data_ingestion' section.

        Args:
            data_ingestion_config (Optional[DataIngestionConfig]):
                Preconstructed configuration object. If None, will be built
                from params.yaml.
        """
        try:
            if data_ingestion_config is None:
                params = read_yaml_file(filepath="params.yaml")
                data_ingestion_params = params.get("data_ingestion", {})

                self.data_ingestion_config: DataIngestionConfig = DataIngestionConfig(
                    test_size=float(data_ingestion_params["test_size"]),
                    random_state=int(data_ingestion_params["random_state"]),
                )
            else:
                self.data_ingestion_config: DataIngestionConfig = data_ingestion_config

        except Exception as e:
            raise MyException(e, sys) from e

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform primitive preprocessing on the raw DataFrame.

        Steps:
            - Drop duplicate rows in-place.
            - Filter to only rows where 'sentiment' is in {'positive', 'negative'}.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Processed DataFrame containing only positive/negative sentiment.

        Raises:
            MyException: If preprocessing fails at any step.
        """
        try:
            df.drop_duplicates(inplace=True)

            if "sentiment" not in df.columns:
                raise ValueError("Column 'sentiment' not found in input DataFrame.")

            df: pd.DataFrame = df.loc[
                df["sentiment"].isin(["positive", "negative"])
            ].copy()
            return df

        except Exception as e:
            raise MyException(e, sys) from e

    def _train_test_splitting(
        self,
        df: pd.DataFrame,
        test_size: float,
        random_state: int,
        stratify: Optional[pd.Series] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the DataFrame into training and testing sets.

        Args:
            df (pd.DataFrame): Processed DataFrame to split.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Random seed for reproducibility.
            stratify (Optional[pd.Series]):
                Column/array used for stratification. Typically df['sentiment'].

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                A tuple containing (train_data, test_data).

        Raises:
            MyException: If splitting fails.
        """
        try:
            train_data, test_data = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify,
            )

            return train_data, test_data

        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        """
        Execute the full data ingestion workflow:

            1. Load environment variables.
            2. Initialize S3 client via aws_connection.s3_operations.
            3. Fetch the CSV dataset from S3.
            4. Run primitive preprocessing.
            5. Split into train/test sets.
            6. Save splits as CSVs to configured locations.
            7. Build and return DataIngestionArtifacts.

        Returns:
            DataIngestionArtifacts: Artifact object with paths to train/test CSV files.

        Raises:
            MyException: If any of the ingestion steps fail.
        """
        try:
            logging.info("Starting Data Ingestion...")

            bucket_name: str = os.getenv(key=DATASET_BUCKET_NAME)
            aws_access_key: str = os.getenv(key=AWS_ACCESS_KEY_ID)
            aws_secret_key: str = os.getenv(key=AWS_SECRET_ACCESS_KEY)
            aws_region: str = os.getenv(key=AWS_REGION)

            if not bucket_name:
                raise ValueError("DATASET_BUCKET_NAME is not set in environment.")
            if not aws_access_key or not aws_secret_key:
                raise ValueError("AWS credentials are not fully set in environment.")
            if not aws_region:
                raise ValueError("AWS_REGION is not set in environment.")

            logging.info("Initializing S3 connection...")
            s3 = aws_connection.S3Operations(
                bucket_name=bucket_name,
                aws_access_key=aws_access_key,
                aws_secret_key=aws_secret_key,
                aws_region=aws_region,
            )

            logging.info("Fetching dataset from S3...")
            with Halo(
                text="Fetching dataset from S3...",
                spinner="dots",
            ):
                df: pd.DataFrame = s3.fetch_csv_from_s3(file_key=AWS_CSV_FILENAME)

            logging.info("Performing primitive preprocessing...")
            processed_df: pd.DataFrame = self._preprocess_data(df=df)

            logging.info("Splitting into training & testing data...")

            stratify_series: Optional[pd.Series] = (
                processed_df["sentiment"]
                if "sentiment" in processed_df.columns
                else None
            )

            train_data, test_data = self._train_test_splitting(
                df=processed_df,
                test_size=self.data_ingestion_config.test_size,
                random_state=self.data_ingestion_config.random_state,
                stratify=stratify_series,
            )

            logging.info("Dumping training & testing data...")
            save_df_as_csv(
                df=train_data,
                filepath=self.data_ingestion_config.raw_train_filepath,
                index=False,
            )
            save_df_as_csv(
                df=test_data,
                filepath=self.data_ingestion_config.raw_test_filepath,
                index=False,
            )

            data_ingestion_artifacts = DataIngestionArtifacts(
                raw_train_filepath=self.data_ingestion_config.raw_train_filepath,
                raw_test_filepath=self.data_ingestion_config.raw_test_filepath,
            )

            logging.info(
                f"Raw train filepath: {data_ingestion_artifacts.raw_train_filepath}, Raw test filepath: {data_ingestion_artifacts.raw_test_filepath}",
            )

            logging.info("Data Ingestion complete.")
            return data_ingestion_artifacts

        except Exception as e:
            raise MyException(e, sys) from e


def main() -> DataIngestionArtifacts:
    """
    Entry point for running data ingestion as a standalone script.

    Returns:
        DataIngestionArtifacts: Artifact object containing paths to train/test data.

    Raises:
        MyException: If the data ingestion pipeline fails.
    """
    try:
        data_ingestor = DataIngestion()
        data_ingestion_artifacts: DataIngestionArtifacts = (
            data_ingestor.initiate_data_ingestion()
        )
        return data_ingestion_artifacts

    except MyException:
        raise

    except Exception as e:
        raise MyException(e, sys) from e


if __name__ == "__main__":
    main()
