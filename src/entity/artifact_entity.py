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
        interim_train_filepath (str): Local path to the preprocessed training CSV.
        interim_test_filepath (str): Local path to the preprocessed testing CSV.
    """

    interim_train_filepath: str
    interim_test_filepath: str


@dataclass
class FeatureEngineeringArtifacts:
    """
    Artifact object produced by the feature engineering / vectorization step.

    Attributes:
        processed_train_filepath (str): Local path to the processed training data
            (typically feature matrix + target, or a serialized representation).
        processed_test_filepath (str): Local path to the processed testing data
            (typically feature matrix + target, or a serialized representation).
        vectorizer_filepath (str): Local path to the persisted vectorizer object
            (e.g., CountVectorizer, TfidfVectorizer).
    """

    processed_train_filepath: str
    processed_test_filepath: str
    vectorizer_filepath: str


@dataclass
class ModelTrainingArtifacts:
    """
    Artifact object produced by the model training step.

    Attributes:
        model_filepath (str): Local path to the persisted trained model file
            (e.g., pickled LogisticRegression instance).
    """

    model_filepath: str


@dataclass
class ModelEvaluationArtifacts:
    """
    Artifact object produced by the model evaluation step.

    Attributes:
        metrics_filepath (str): Local path to the saved metrics report (JSON).
        model_info_filepath (str): Local path to the saved model metadata (JSON),
            including MLflow run ID.
    """

    metrics_filepath: str
    model_info_filepath: str
