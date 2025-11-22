import os
from typing import Final
from dataclasses import dataclass, field

from src.constants import (
    PIPELINE_NAME,
    DATA_DIRNAME,
    DATA_INGESTION_DIRNAME,
    RAW_TRAIN_FILENAME,
    RAW_TEST_FILENAME,
    PROCESSED_DATA_DIRNAME,
    PROCESSED_TRAIN_FILENAME,
    PROCESSED_TEST_FILENAME,
)


@dataclass
class TrainingPipelineConfig:
    """
    Configuration for the main training pipeline.

    Attributes:
        pipeline_name (str): Logical name of the pipeline (used in logging, metadata).
        data_dirpath (str): Root directory where all pipeline data artifacts are stored.
    """

    pipeline_name: str = field(default=PIPELINE_NAME)
    data_dirpath: str = field(default_factory=lambda: os.path.join(DATA_DIRNAME))


training_pipeline_config: Final[TrainingPipelineConfig] = TrainingPipelineConfig()


@dataclass
class DataIngestionConfig:
    """
    Configuration for the data ingestion stage.

    Attributes:
        raw_data_dirpath (str): Directory where raw train/test CSVs will be stored.
        raw_train_filepath (str): File path for the raw training CSV.
        raw_test_filepath (str): File path for the raw testing CSV.
        test_size (float): Proportion of data to use for the test split.
        random_state (int): Seed used for reproducible train/test splitting.
    """

    raw_data_dirpath: str = field(init=False)
    raw_train_filepath: str = field(init=False)
    raw_test_filepath: str = field(init=False)

    test_size: float
    random_state: int

    def __post_init__(self) -> None:
        """
        Initialize derived file system paths for the ingestion outputs and
        ensure the raw data directory exists.
        """
        self.raw_data_dirpath = os.path.join(
            training_pipeline_config.data_dirpath,
            DATA_INGESTION_DIRNAME,
        )
        os.makedirs(self.raw_data_dirpath, exist_ok=True)

        self.raw_train_filepath = os.path.join(
            self.raw_data_dirpath,
            RAW_TRAIN_FILENAME,
        )
        self.raw_test_filepath = os.path.join(
            self.raw_data_dirpath,
            RAW_TEST_FILENAME,
        )


@dataclass
class DataPreprocessingConfig:
    """
    Configuration for the data preprocessing stage.

    Attributes:
        raw_train_filepath (str): Path to the raw training CSV produced by ingestion.
        raw_test_filepath (str): Path to the raw testing CSV produced by ingestion.
        processed_data_dirpath (str): Directory where processed train/test CSVs will be stored.
        processed_train_filepath (str): File path for the processed training CSV.
        processed_test_filepath (str): File path for the processed testing CSV.
        features (list[str]): List of column names that should undergo text preprocessing.
    """

    raw_train_filepath: str = field(init=False)
    raw_test_filepath: str = field(init=False)

    processed_data_dirpath: str = field(init=False)
    processed_train_filepath: str = field(init=False)
    processed_test_filepath: str = field(init=False)

    features: list[str]

    def __post_init__(self) -> None:
        """
        Initialize file system paths for raw and processed data, and ensure
        the processed data directory exists.
        """
        self.raw_train_filepath = os.path.join(
            training_pipeline_config.data_dirpath,
            DATA_INGESTION_DIRNAME,
            RAW_TRAIN_FILENAME,
        )
        self.raw_test_filepath = os.path.join(
            training_pipeline_config.data_dirpath,
            DATA_INGESTION_DIRNAME,
            RAW_TEST_FILENAME,
        )

        self.processed_data_dirpath = os.path.join(
            training_pipeline_config.data_dirpath,
            PROCESSED_DATA_DIRNAME,
        )
        os.makedirs(self.processed_data_dirpath, exist_ok=True)

        self.processed_train_filepath = os.path.join(
            self.processed_data_dirpath,
            PROCESSED_TRAIN_FILENAME,
        )
        self.processed_test_filepath = os.path.join(
            self.processed_data_dirpath,
            PROCESSED_TEST_FILENAME,
        )
