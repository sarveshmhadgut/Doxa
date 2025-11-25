import os
import sys
import mlflow
import dagshub
from dotenv import load_dotenv
from src.logger import logging
from typing import List, Optional
from src.exception import MyException
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from src.entity.config_entity import ModelPromotionConfig
from src.constants import (
    DAGSHUB_URI,
    DAGSHUB_USERNAME,
    DAGSHUB_TOKEN,
)

load_dotenv()


class ModelPromotion:
    """
    Orchestrates the model lifecycle management within the MLflow Model Registry.
    It connects to the remote MLflow tracking server (DagsHub) and manages
    the promotion of the latest validated model to the 'Production' stage,
    while archiving older 'Production' models.

    Responsibilities:
        - Connect to MLflow tracking server.
        - Archive existing production models.
        - Promote the latest staging/registered model to production.
    """

    def __init__(
        self,
        model_promotion_config: ModelPromotionConfig = None,
    ) -> None:
        """
        Initializes the ModelPromotion class with configuration.

        Args:
            model_promotion_config (ModelPromotionConfig): Configuration object
                containing model name and file paths.

        Raises:
            MyException: If initialization fails.
        """
        try:
            if model_promotion_config is None:
                self.model_promotion_config: ModelPromotionConfig = (
                    ModelPromotionConfig()
                )
            else:
                self.model_promotion_config: ModelPromotionConfig = (
                    model_promotion_config
                )

        except Exception as e:
            raise MyException(e, sys) from e

    def _get_mlflow_client(
        self, dagshub_username: str, dagshub_token: str, dagshub_uri: str
    ) -> MlflowClient:
        """
        Sets up the authentication environment and creates an MLflow client
        connected to the DagsHub tracking server.

        Args:
            dagshub_username (str): DagsHub repository owner username.
            dagshub_token (str): DagsHub access token.
            dagshub_uri (str): The MLflow tracking URI.

        Returns:
            MlflowClient: An initialized MLflow client instance.

        Raises:
            MyException: If client creation fails.
        """
        try:
            if not dagshub_token:
                raise EnvironmentError(
                    "DAGSHUB_TOKEN environment variable is not set for MLflow tracking."
                )

            dagshub_token = dagshub_token.strip()

            # Set environment variables for MLflow authentication
            os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username
            os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

            mlflow.set_tracking_uri(dagshub_uri)
            dagshub.auth.add_app_token(dagshub_token)

            client: MlflowClient = mlflow.MlflowClient()
            return client

        except Exception as e:
            raise MyException(e, sys) from e

    def _archive_production_models(self, model_name: str, client: MlflowClient) -> None:
        """
        Searches for all models currently tagged 'Production' and archives them by
        setting their tag to 'Archived'.

        Args:
            model_name (str): The name of the registered model.
            client (MlflowClient): The initialized MLflow client.

        Raises:
            MyException: If archiving fails.
        """
        try:
            production_models: List[ModelVersion] = client.search_model_versions(
                f"name='{model_name}' and tags.stage='Production'"
            )

            if not production_models:
                return

            sorted_versions: List[ModelVersion] = sorted(
                production_models, key=lambda v: int(v.version), reverse=True
            )

            for model in sorted_versions:
                client.set_model_version_tag(
                    name=model_name,
                    version=model.version,
                    key="stage",
                    value="Archived",
                )

        except Exception as e:
            raise MyException(e, sys) from e

    def promote_latest_model(self, model_name: str, client: MlflowClient) -> int:
        """
        Promotes the latest model version tagged 'Staging' to 'Production'.
        If no model is 'Staging', it promotes the absolute latest version.

        Args:
            model_name (str): The name of the registered model.
            client (MlflowClient): The initialized MLflow client.

        Returns:
            int: model version of the promoted model.

        Raises:
            MyException: If promotion fails.
        """
        try:
            latest_model: Optional[ModelVersion] = None

            staged_models: List[ModelVersion] = client.search_model_versions(
                f"name='{model_name}' and tags.stage='Staging'"
            )

            if staged_models:
                sorted_versions: List[ModelVersion] = sorted(
                    staged_models, key=lambda v: int(v.version), reverse=True
                )
                latest_model: ModelVersion = sorted_versions[0]

            if latest_model:
                model_version: str = latest_model.version
                client.set_model_version_tag(
                    name=model_name,
                    version=model_version,
                    key="stage",
                    value="Production",
                )
                return latest_model.version

            models: List[ModelVersion] = client.search_model_versions(
                f"name='{model_name}'"
            )
            if not models:
                return None

            sorted_versions: List[ModelVersion] = sorted(
                models, key=lambda v: int(v.version), reverse=True
            )

            latest_model: ModelVersion = sorted_versions[0]

            if latest_model:
                model_version: str = latest_model.version
                client.set_model_version_tag(
                    name=model_name,
                    version=model_version,
                    key="stage",
                    value="Production",
                )
                return model_version
            return -1

        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_promotion(self) -> None:
        """
        Orchestrates the entire model promotion lifecycle: client connection,
        archiving, and promotion.

        Raises:
            MyException: If model promotion initiation fails.
        """
        try:
            logging.info("Starting Model Promotion...")

            dagshub_uri: str = os.getenv(DAGSHUB_URI)
            dagshub_username: str = os.getenv(DAGSHUB_USERNAME)
            dagshub_token: str = os.getenv(DAGSHUB_TOKEN)
            model_name: str = self.model_promotion_config.registered_model_name

            mlflow_client: MlflowClient = self._get_mlflow_client(
                dagshub_username=dagshub_username,
                dagshub_token=dagshub_token,
                dagshub_uri=dagshub_uri,
            )

            logging.info("Archiving existing production models...")
            self._archive_production_models(model_name=model_name, client=mlflow_client)

            logging.info("Promoting latest model...")
            self.promote_latest_model(model_name=model_name, client=mlflow_client)

            logging.info("Model Promotion complete.")

        except Exception as e:
            raise MyException(e, sys) from e


def main() -> None:
    """
    Main execution function to orchestrate the model promotion lifecycle.

    Raises:
        MyException: If model promotion fails.
    """
    try:
        model_promoter: ModelPromotion = ModelPromotion()
        model_promoter.initiate_model_promotion()

    except Exception as e:
        raise MyException(e, sys) from e


if __name__ == "__main__":
    main()
