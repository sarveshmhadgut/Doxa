import os
import sys
import mlflow
import dagshub
from dotenv import load_dotenv
from datetime import datetime
from src.logger import logging
from typing import Optional
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

            os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username
            os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

            mlflow.set_tracking_uri(dagshub_uri)
            dagshub.auth.add_app_token(dagshub_token)

            client: MlflowClient = mlflow.MlflowClient()
            return client

        except Exception as e:
            raise MyException(e, sys) from e

    def _archive_current_champion(
        self, model_name: str, client: MlflowClient, new_champion_version: str
    ) -> None:
        """
        Archives the current champion model if it exists and is different from the new champion.

        This involves:
        1. Setting the 'stage' tag to 'Archived'.
        2. Removing the 'champion' alias.
        3. Updating the model description with an archival timestamp.

        Args:
            model_name (str): The name of the registered model.
            client (MlflowClient): The initialized MLflow client.
            new_champion_version (str): The version of the new model being promoted.
        """
        try:
            current_champion = client.get_model_version_by_alias(
                name=model_name, alias="champion"
            )
            if current_champion and current_champion.version != new_champion_version:
                client.set_model_version_tag(
                    name=model_name,
                    version=current_champion.version,
                    key="stage",
                    value="Archived",
                )

                client.delete_registered_model_alias(name=model_name, alias="champion")

                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                client.update_model_version(
                    name=model_name,
                    version=current_champion.version,
                    description=f"Archived on {current_time}",
                )

        except Exception:
            logging.warning("Failed to archive current champion (likely none exists):")
            pass

    def promote_latest_model(self, model_name: str, client: MlflowClient) -> bool:
        """
        Promotes the model aliased as 'challenger' to 'champion'.

        The process involves:
        1. Identifying the model version with the 'challenger' alias.
        2. Archiving the current 'champion' (if any) by setting a tag and updating its description.
        3. Assigning the 'champion' alias to the new model.
        4. Updating the new champion's description.
        5. Removing the 'challenger' alias from the promoted model.

        Args:
            model_name (str): The name of the registered model.
            client (MlflowClient): The initialized MLflow client.

        Returns:
            None

        Raises:
            MyException: If promotion fails.
        """
        try:
            target_model: Optional[ModelVersion] = None

            is_challenger = False
            try:
                target_model = client.get_model_version_by_alias(
                    name=model_name, alias="challenger"
                )
                is_challenger = True
            except Exception:
                target_model = None

            if not target_model:
                return False

            logging.info("Archiving current champion...")
            self._archive_current_champion(
                model_name=model_name,
                client=client,
                new_champion_version=target_model.version,
            )

            client.set_registered_model_alias(
                name=model_name,
                version=target_model.version,
                alias="champion",
            )

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            client.update_model_version(
                name=model_name,
                version=target_model.version,
                description=f"Promoted to champion on {current_time}",
            )

            if is_challenger:
                client.delete_registered_model_alias(
                    name=model_name, alias="challenger"
                )

            return True

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

            logging.info("Promoting latest model...")
            status = self.promote_latest_model(
                model_name=model_name, client=mlflow_client
            )

            if status:
                logging.info("Model Promotion complete.")
            else:
                logging.warning("Model Promotion Aborted, no 'challenger' model found")

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
