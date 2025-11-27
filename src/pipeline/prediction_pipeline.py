import os
import io
import sys
import mlflow
import dagshub
import pandas as pd
from numpy import ndarray
from typing import List, Any
from dotenv import load_dotenv
from src.logger import logging
from src.exception import MyException
from contextlib import redirect_stdout
from src.utils.main_utils import load_object
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from src.data.data_preprocessing import DataPreprocessing
from src.entity.config_entity import PredictionPipelineConfig
from src.constants import DAGSHUB_URI, DAGSHUB_REPO, DAGSHUB_USERNAME, DAGSHUB_TOKEN

load_dotenv()
terminal_width: int = os.get_terminal_size().columns if os.isatty(1) else 80


class PredictionPipeline:
    """
    Orchestrates the end-to-end inference process. It handles connecting to MLflow,
    loading the latest registered model and its associated vectorizer, and processing
    raw text input to generate a prediction.

    Responsibilities:
        - Connect to DagsHub/MLflow.
        - Fetch the latest 'Production' or 'Staging' model.
        - Load the associated vectorizer.
        - Preprocess input text and generate prediction.
    """

    def __init__(self, prediction_pipeline_config=None) -> None:
        """
        Initializes the pipeline, configuration, and necessary components.

        Args:
            prediction_pipeline_config (PredictionPipelineConfig, optional): Configuration object.

        Raises:
            MyException: If initialization fails.
        """
        try:
            self._dagshub_connected = False
            self._vectorizer = None
            self._model = None

            if prediction_pipeline_config is None:
                self.prediction_pipeline_config: PredictionPipelineConfig = (
                    PredictionPipelineConfig()
                )
            else:
                self.prediction_pipeline_config: PredictionPipelineConfig = (
                    prediction_pipeline_config
                )

            self.data_preprocessing: DataPreprocessing = DataPreprocessing()

        except Exception as e:
            raise MyException(e, sys) from e

    def _connect_dagshub(
        self,
        dagshub_uri: str,
        dagshub_repo: str,
        dagshub_username: str,
        dagshub_token: str,
    ) -> None:
        """
        Connects to DagsHub/MLflow tracking URI, suppressing verbose output.

        Args:
            dagshub_uri (str): The URI for DagsHub tracking.
            dagshub_repo (str): The name of the DagsHub repository.
            dagshub_username (str): The username of the DagsHub account.
            dagshub_token (str): The DagsHub authentication token.

        Raises:
            MyException: If connection fails.
        """
        try:
            if self._dagshub_connected:
                return

            logging.getLogger().setLevel(logging.WARNING)

            if not dagshub_username:
                raise EnvironmentError("DAGSHUB_USERNAME is missing!")
            if not dagshub_token:
                raise EnvironmentError("DAGSHUB_TOKEN is missing!")
            else:
                dagshub_token = dagshub_token.strip()
                dagshub.auth.add_app_token(dagshub_token)

            mlflow.set_tracking_uri(dagshub_uri)
            os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username
            os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
            os.environ["DAGSHUB_TOKEN"] = dagshub_token

            with redirect_stdout(io.StringIO()):
                dagshub.init(
                    repo_name=dagshub_repo,
                    repo_owner=dagshub_username,
                    mlflow=True,
                )
            self._dagshub_connected = True

        except Exception as e:
            raise MyException(e, sys) from e
        finally:
            logging.getLogger().setLevel(logging.INFO)

    def _get_latest_model_version(self, model_name: str):
        """
        Retrieves the latest registered model version based on 'stage' tag.
        Prioritizes 'Production' first, then 'Staging', then any version.

        Args:
            model_name (str): Name of the registered model.

        Returns:
            ModelVersion: The latest model version object, or None if not found.

        Raises:
            MyException: If retrieval fails.
        """
        try:
            client: MlflowClient = mlflow.MlflowClient()

            stage_env = os.getenv("MLFLOW_STAGE")
            stage_priority: List[str] = (
                [stage_env] if stage_env else ["Production", "Staging"]
            )

            for stage in stage_priority:
                versions: List[ModelVersion] = client.search_model_versions(
                    f"name='{model_name}' and tags.stage='{stage}'"
                )

                if versions:
                    sorted_versions: List[ModelVersion] = sorted(
                        versions, key=lambda v: int(v.version), reverse=True
                    )
                    return sorted_versions[0]

            versions: ModelVersion = client.search_model_versions(
                f"name='{model_name}'"
            )

            if not versions:
                return None

            sorted_versions: List[ModelVersion] = sorted(
                versions, key=lambda v: int(v.version), reverse=True
            )
            return sorted_versions[0]

        except Exception as e:
            raise MyException(e, sys) from e

    def _load_vectorizer(self, vectorizer_filepath: str) -> Any:
        """
        Loads the saved vectorizer object.

        Args:
            vectorizer_filepath (str): Path to the saved vectorizer file.

        Returns:
            Any: The loaded vectorizer object.

        Raises:
            MyException: If vectorizer loading fails.
        """
        try:
            if self._vectorizer is not None:
                return self._vectorizer

            self._vectorizer: Any = load_object(filepath=vectorizer_filepath)
            return self._vectorizer

        except Exception as e:
            raise MyException(e, sys) from e

    def _load_model(self, model_name: str) -> Any:
        """
        Loads the latest registered model from the MLflow Model Registry.

        Args:
            model_name (str): Name of the registered model.

        Returns:
            Any: The loaded MLflow model.

        Raises:
            MyException: If model loading fails.
        """
        try:
            if self._model is not None:
                return self._model

            version_info: ModelVersion = self._get_latest_model_version(
                model_name=model_name
            )

            if version_info is None:
                raise Exception("Model not found")

            model_uri: str = f"runs:/{version_info.run_id}/model"
            self._model: Any = mlflow.pyfunc.load_model(model_uri)
            return self._model

        except Exception as e:
            raise MyException(e, sys) from e

    def preheat(self) -> None:
        """
        Performs a dummy prediction run to load artifacts and cache models.

        Returns:
            None

        Raises:
            MyException: If preheating fails.
        """
        try:
            dagshub_uri: str = os.getenv(DAGSHUB_URI)
            dagshub_repo: str = os.getenv(DAGSHUB_REPO)
            dagshub_username: str = os.getenv(DAGSHUB_USERNAME)
            dagshub_token: str = os.getenv(DAGSHUB_TOKEN)

            self._connect_dagshub(
                dagshub_uri=dagshub_uri,
                dagshub_repo=dagshub_repo,
                dagshub_username=dagshub_username,
                dagshub_token=dagshub_token,
            )

            vectorizer_filepath = self.prediction_pipeline_config.vectorizer_filepath
            vectorizer: Any = self._load_vectorizer(
                vectorizer_filepath=vectorizer_filepath
            )

            model_name = self.prediction_pipeline_config.model_name
            model: Any = self._load_model(model_name=model_name)

            text: str = self.data_preprocessing._preprocess_text("nice")
            features: ndarray = vectorizer.transform([text])

            dummy_df: pd.DataFrame = pd.DataFrame(
                features.toarray(),
                columns=[f"feature_{i}" for i in range(features.shape[1])],
            )

            model.predict(dummy_df)

        except Exception as e:
            raise MyException(e, sys) from e

    def run_prediction_pipeline(self, text: str) -> str:
        """
        Executes the prediction pipeline on the given text input.

        Args:
            text (str): The raw text input to classify.

        Returns:
            str: The predicted class label.

        Raises:
            MyException: If prediction fails.
        """
        try:
            logging.info("Prediction Pipeline")

            logging.info("Connecting dagshub...")
            dagshub_uri: str = os.getenv(DAGSHUB_URI)
            dagshub_repo: str = os.getenv(DAGSHUB_REPO)
            dagshub_username: str = os.getenv(DAGSHUB_USERNAME)
            dagshub_token: str = os.getenv(DAGSHUB_TOKEN)

            self._connect_dagshub(
                dagshub_uri=dagshub_uri,
                dagshub_repo=dagshub_repo,
                dagshub_username=dagshub_username,
                dagshub_token=dagshub_token,
            )

            logging.info("Fetching vectorizer...")
            vectorizer_filepath = self.prediction_pipeline_config.vectorizer_filepath
            vectorizer: Any = self._load_vectorizer(
                vectorizer_filepath=vectorizer_filepath
            )

            logging.info("Fetching model...")
            model_name = self.prediction_pipeline_config.model_name
            model: Any = self._load_model(model_name=model_name)

            logging.info("Preprocessing text...")
            processed_text = self.data_preprocessing._preprocess_text(text)

            logging.info("Vectorizing features...")
            features = vectorizer.transform([processed_text])

            features_df = pd.DataFrame(
                features.toarray(),
                columns=[f"feature_{i}" for i in range(features.shape[1])],
            )

            logging.info("Predicting sentiment...")
            result = model.predict(features_df)

            prediction = result[0]
            logging.info("Prediction Pipeline completed")

            return prediction

        except Exception as e:
            logging.info("Prediction pipeline failed!")

            raise MyException(e, sys) from e
