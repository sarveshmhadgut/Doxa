import os
from typing import Final
from dataclasses import dataclass, field

from src.constants import (
    PIPELINE_NAME,
    DATA_DIRNAME,
    DATA_INGESTION_DIRNAME,
    RAW_TRAIN_FILENAME,
    RAW_TEST_FILENAME,
    INTERIM_DATA_DIRNAME,
    INTERIM_TRAIN_FILENAME,
    INTERIM_TEST_FILENAME,
    PROCESSED_DATA_DIRNAME,
    PROCESSED_TRAIN_FILENAME,
    PROCESSED_TEST_FILENAME,
    MODELS_DIRNAME,
    VECTORIZER_FILENAME,
    MODEL_FILENAME,
    REPORTS_DIRNAME,
    METRICS_FILENAME,
    MODEL_INFO_FILENAME,
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
    models_dirpath: str = field(default_factory=lambda: os.path.join(MODELS_DIRNAME))
    reports_dirname: str = field(default_factory=lambda: os.path.join(REPORTS_DIRNAME))


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
        interim_data_dirpath (str): Directory where processed train/test CSVs will be stored.
        interim_train_filepath (str): File path for the processed training CSV.
        interim_test_filepath (str): File path for the processed testing CSV.
        features (list[str]): List of column names that should undergo text preprocessing.
    """

    raw_train_filepath: str = field(init=False)
    raw_test_filepath: str = field(init=False)

    interim_data_dirpath: str = field(init=False)
    interim_train_filepath: str = field(init=False)
    interim_test_filepath: str = field(init=False)

    features: list[str]
    target: str

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

        self.interim_data_dirpath = os.path.join(
            training_pipeline_config.data_dirpath,
            INTERIM_DATA_DIRNAME,
        )
        os.makedirs(self.interim_data_dirpath, exist_ok=True)

        self.interim_train_filepath = os.path.join(
            self.interim_data_dirpath,
            INTERIM_TRAIN_FILENAME,
        )
        self.interim_test_filepath = os.path.join(
            self.interim_data_dirpath,
            INTERIM_TEST_FILENAME,
        )


@dataclass
class FeatureEngineeringConfig:
    """
    Configuration for the feature engineering / vectorization stage.

    Attributes:
        interim_train_filepath (str): Path to the interim training CSV (preprocessed text).
        interim_test_filepath (str): Path to the interim testing CSV (preprocessed text).
        processed_data_dirpath (str): Directory where final processed train/test feature
            matrices metadata (e.g., CSVs or references) will be stored.
        processed_train_filepath (str): File path for the processed training data
            (features + target) representation.
        processed_test_filepath (str): File path for the processed testing data
            (features + target) representation.
        vectorizer_filepath (str): File path for the saved vectorizer object.
        max_features (int): Maximum number of features for the vectorizer
            (e.g., max vocabulary size for bag-of-words).
        feature (str): Name of the single text feature column to vectorize.
        target (str): Name of the target column.
    """

    interim_train_filepath: str = field(init=False)
    interim_test_filepath: str = field(init=False)

    processed_data_dirpath: str = field(init=False)
    processed_train_filepath: str = field(init=False)
    processed_test_filepath: str = field(init=False)

    vectorizer_filepath: str = field(init=False)

    max_features: int
    feature: str
    target: str

    def __post_init__(self) -> None:
        """
        Initialize file system paths for interim inputs, processed outputs,
        and the vectorizer object, and ensure necessary directories exist.
        """
        self.interim_train_filepath = os.path.join(
            training_pipeline_config.data_dirpath,
            INTERIM_DATA_DIRNAME,
            INTERIM_TRAIN_FILENAME,
        )
        self.interim_test_filepath = os.path.join(
            training_pipeline_config.data_dirpath,
            INTERIM_DATA_DIRNAME,
            INTERIM_TEST_FILENAME,
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

        self.vectorizer_filepath = os.path.join(
            training_pipeline_config.models_dirpath,
            VECTORIZER_FILENAME,
        )


@dataclass
class ModelTrainingConfig:
    """
    Configuration for the model training stage.

    Attributes:
        processed_train_filepath (str): Path to the processed training CSV produced
            by the feature engineering stage.
        model_filepath (str): Path where the trained model will be saved.
        target (str): Name of the target column used for training.
        c (float): Inverse regularization strength (C parameter) for Logistic Regression.
        solver (str): Solver to use in Logistic Regression.
        penalty (str): Norm used in the penalization.
        max_iter (int): Maximum number of iterations for the solver.
    """

    processed_train_filepath: str = field(init=False)
    model_filepath: str = field(init=False)

    target: str
    c: float
    solver: str
    penalty: str
    max_iter: int

    def __post_init__(self) -> None:
        """
        Initialize file system paths for processed training data and the model file.
        """
        self.processed_train_filepath = os.path.join(
            training_pipeline_config.data_dirpath,
            PROCESSED_DATA_DIRNAME,
            PROCESSED_TRAIN_FILENAME,
        )

        self.model_filepath = os.path.join(
            training_pipeline_config.models_dirpath,
            MODEL_FILENAME,
        )


@dataclass
class ModelEvaluationConfig:
    """
    Configuration for the model evaluation stage.

    Attributes:
        processed_test_filepath (str): Path to the processed test data (features + target).
        model_filepath (str): Path to the trained model file.
        metrics_filepath (str): Path where the final evaluation metrics JSON will be saved.
        model_info_filepath (str): Path where the model metadata JSON (e.g., run_id) will be saved.
        target (str): Name of the target column.
    """

    processed_test_filepath: str = field(init=False)
    model_filepath: str = field(init=False)
    metrics_filepath: str = field(init=False)
    model_info_filepath: str = field(init=False)

    target: str

    def __post_init__(self):
        """
        Initialize file system paths for inputs and report outputs,
        and ensure the necessary reports directory exists.
        """

        self.processed_test_filepath = os.path.join(
            training_pipeline_config.data_dirpath,
            PROCESSED_DATA_DIRNAME,
            PROCESSED_TEST_FILENAME,
        )

        self.model_filepath = os.path.join(
            training_pipeline_config.models_dirpath, MODEL_FILENAME
        )

        self.metrics_filepath = os.path.join(
            training_pipeline_config.reports_dirname, METRICS_FILENAME
        )

        self.model_info_filepath = os.path.join(
            training_pipeline_config.reports_dirname, MODEL_INFO_FILENAME
        )
