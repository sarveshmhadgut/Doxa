from dataclasses import dataclass


@dataclass
class DataIngestionArtifacts:
    """
    Artifact object produced by the data ingestion step.

    Attributes:
        raw_train_filepath (str): Local path to the ingested training CSV.
        raw_test_filepath (str): Local path to the ingested testing CSV.
    """

    raw_train_filepath: str
    raw_test_filepath: str


@dataclass
class DataPreprocessingArtifacts:
    """
    Artifact object produced by the data preprocessing step.

    Attributes:
        processed_train_filepath (str): Local path to the preprocessed training CSV.
        processed_test_filepath (str): Local path to the preprocessed testing CSV.
    """

    processed_train_filepath: str
    processed_test_filepath: str
