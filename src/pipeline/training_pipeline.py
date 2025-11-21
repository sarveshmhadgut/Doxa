import os
import sys
from typing import Optional

from src.logger import logging
from src.exception import MyException
from src.data.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifacts

terminal_width: int = os.get_terminal_size().columns if os.isatty(1) else 80


class TrainPipeline:
    """
    Orchestrates execution of the end-to-end training pipeline.

    Currently implemented stages:
        - Data Ingestion
    """

    def __init__(
        self, data_ingestion_config: Optional[DataIngestionConfig] = None
    ) -> None:
        """
        Initialize the TrainPipeline with optional stage configurations.

        Args:
            data_ingestion_config (Optional[DataIngestionConfig]):
                Optional pre-built data ingestion configuration. If None,
                the DataIngestion step will construct its own config from params.yaml.
        """
        try:
            self.data_ingestion_config: Optional[DataIngestionConfig] = (
                data_ingestion_config
            )

        except Exception as e:
            raise MyException(e, sys) from e

    def start_data_ingestion(self) -> DataIngestionArtifacts:
        """
        Run the data ingestion stage of the pipeline.

        Returns:
            DataIngestionArtifacts: Paths to the ingested training and testing datasets.
        """
        try:
            logging.info("Starting Data Ingestion stage...")
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )

            data_ingestion_artifacts: DataIngestionArtifacts = (
                data_ingestion.initiate_data_ingestion()
            )

            logging.info(
                f"Data Ingestion stage completed. "
                f"Train path: {data_ingestion_artifacts.raw_train_filepath}, "
                f"Test path: {data_ingestion_artifacts.raw_test_filepath}"
            )
            return data_ingestion_artifacts

        except Exception as e:
            raise MyException(e, sys) from e

    def run_pipeline(self) -> None:
        """
        Execute the full training pipeline in sequence.
        """
        try:
            print("=" * terminal_width)
            logging.info("Executing training pipeline...")
            print("Executing training pipeline...")
            print("-" * terminal_width)

            # Stage 1: Data Ingestion
            logging.info("Executing Data Ingestion stage...")
            print("Stage 1: Data Ingestion")

            data_ingestion_artifacts: DataIngestionArtifacts = (
                self.start_data_ingestion()
            )

            logging.info("Data Ingestion completed successfully.")
            print(data_ingestion_artifacts)
            print("-" * terminal_width)

            logging.info("Training pipeline executed successfully.")
            print("Training pipeline completed successfully.")
            print("=" * terminal_width)

        except Exception as e:
            print("=" * terminal_width)
            print("Training pipeline failed!")
            print("=" * terminal_width)
            raise MyException(e, sys) from e
