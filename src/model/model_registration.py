import os
import sys
import json
import mlflow
import dagshub
from typing import Dict
from dotenv import load_dotenv
from src.logger import logging
from src.exception import MyException
from mlflow.tracking import MlflowClient
from src.entity.config_entity import ModelRegistrationConfig
from src.constants import (
    DAGSHUB_URI,
    DAGSHUB_REPO,
    DAGSHUB_USERNAME,
)


load_dotenv()


class ModelRegistration:
    """
    Handles registering the final evaluated model into the MLflow Model Registry,
    retrieving the necessary run ID and artifact path from the model info report.
    The model is tagged with 'stage=Staging' for version control.

    Responsibilities:
        - Connect to DagsHub/MLflow.
        - Read model metadata (run ID, path) from local report.
        - Register the model in MLflow Model Registry.
        - Tag the registered model as 'Staging'.
    """

    def __init__(
        self, model_registration_config: ModelRegistrationConfig = None
    ) -> None:
        """
        Initializes the ModelRegistration class with the configuration.

        Args:
            model_registration_config (ModelRegistrationConfig): Configuration object
                containing model name and report file paths.

        Raises:
            MyException: If initialization fails.
        """
        try:
            if model_registration_config is None:
                self.model_registration_config: ModelRegistrationConfig = (
                    ModelRegistrationConfig()
                )
            else:
                self.model_registration_config: ModelRegistrationConfig = (
                    model_registration_config
                )

        except Exception as e:
            raise MyException(e, sys) from e

    def _connect_dagshub(
        self, dagshub_uri: str, dagshub_repo: str, dagshub_username: str
    ) -> None:
        """
        Sets up MLflow tracking URI and DagsHub context connection.
        Logging level is temporarily set to WARNING to suppress initialization noise.

        Args:
            dagshub_uri (str): URI for DagsHub MLflow tracking.
            dagshub_repo (str): Name of the DagsHub repository.
            dagshub_username (str): Username of the DagsHub account.

        Raises:
            MyException: If connection fails.
        """

        try:
            logging.getLogger().setLevel(logging.WARNING)

            # Set environment variables for MLflow authentication
            dagshub_token = os.getenv("DAGSHUB_TOKEN")
            if dagshub_token:
                dagshub_token = dagshub_token.strip()
            os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username
            os.environ["MLFLOW_TRACKING_PASSWORD"] = (
                dagshub_token if dagshub_token else ""
            )
            mlflow.set_tracking_uri(dagshub_uri)
            if dagshub_token:
                dagshub.auth.add_app_token(dagshub_token)

        except Exception as e:
            raise MyException(e, sys) from e

        finally:
            logging.getLogger().setLevel(logging.INFO)

    def _read_model_info(self, model_info_filepath: str) -> Dict[str, str]:
        """
        Reads the local model info JSON file to extract MLflow run ID and artifact path.

        Args:
            model_info_filepath (str): Local path to the model info JSON report.

        Returns:
            Dict[str, str]: Dictionary containing 'run_id' and 'model_path'.

        Raises:
            MyException: If reading model info fails.
        """
        try:
            with open(model_info_filepath, "r") as file:
                model_info: dict = json.load(file)
            return model_info
        except Exception as e:
            raise MyException(e, sys) from e

    def _register_model(self, model_name: str, model_info: Dict[str, str]) -> None:
        """
        Registers the model in the MLflow Model Registry and sets its stage tag.

        Args:
            model_name (str)           : The name under which the model will be registered.
            model_info (Dict[str, str]): Contains the MLflow 'run_id' and 'model_path'.

        Raises:
            MyException: If registration fails.
        """
        try:
            model_uri: str = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

            registered_model = mlflow.register_model(
                model_uri=model_uri, name=model_name
            )

            client: MlflowClient = mlflow.tracking.MlflowClient()

            client.set_model_version_tag(
                name=model_name,
                version=registered_model.version,
                key="stage",
                value="Staging",
            )

        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_registration(self) -> None:
        """
        Orchestrates the connection, data reading, and model registration process.

        Raises:
            MyException: If model registration initiation fails.
        """
        try:
            logging.info("Starting model registration...")

            dagshub_uri: str = os.getenv(DAGSHUB_URI)
            dagshub_repo: str = os.getenv(DAGSHUB_REPO)
            dagshub_username: str = os.getenv(DAGSHUB_USERNAME)

            logging.info("Connecting to DagsHub...")
            self._connect_dagshub(
                dagshub_uri=dagshub_uri,
                dagshub_repo=dagshub_repo,
                dagshub_username=dagshub_username,
            )

            logging.info("Reading model info...")
            registered_model_name: str = (
                self.model_registration_config.registered_model_name
            )

            model_info_filepath: str = (
                self.model_registration_config.model_info_filepath
            )

            model_info: dict = self._read_model_info(
                model_info_filepath=model_info_filepath
            )

            logging.info("Registering model...")
            self._register_model(
                model_name=registered_model_name, model_info=model_info
            )

            logging.info("Model Registration complete")

        except Exception as e:
            raise MyException(e, sys) from e


def main():
    """
    Main execution function for the model registration component.

    Raises:
        MyException: If model registration fails.
    """
    try:
        model_registrant: ModelRegistration = ModelRegistration(
            model_registration_config=None
        )
        model_registrant.initiate_model_registration()

    except Exception as e:
        raise MyException(e, sys) from e


if __name__ == "__main__":
    main()
