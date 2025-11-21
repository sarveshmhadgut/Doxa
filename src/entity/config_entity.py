import os
from typing import Final
from dataclasses import dataclass, field

from src.constants import (
    PIPELINE_NAME,
    DATA_DIRNAME,
    DATA_INGESTION_DIRNAME,
    RAW_TRAIN_FILENAME,
    RAW_TEST_FILENAME,
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
