import io
import os
import sys
import mlflow
import dagshub
import pandas as pd
from halo import Halo
from numpy import ndarray
from dotenv import load_dotenv
from src.logger import logging
from typing import Dict, Tuple
from src.exception import MyException
from contextlib import redirect_stdout
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelEvaluationArtifacts
from src.constants import DAGSHUB_URI, DAGSHUB_REPO, DAGSHUB_USERNAME

from src.utils.main_utils import (
    load_object,
    read_csv_file,
    read_yaml_file,
    save_as_json,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    roc_auc_score,
)

load_dotenv()


class ModelEvaluation:
    """
    Handles the evaluation of a trained model against test data,
    calculates key metrics, and logs the results and the model
    to MLflow/DVC via DagsHub.

    Responsibilities:
        - Setup MLflow experiment and DagsHub connection.
        - Evaluate model performance on test data.
        - Log metrics, parameters, and model to MLflow.
        - Save local reports (metrics, model info).
    """

    def __init__(self, model_evaluation_config: ModelEvaluationConfig = None) -> None:
        """
        Initializes the ModelEvaluation class by setting configuration parameters.
        Reads model training parameters from 'params.yaml' if no configuration is provided.

        Args:
            model_evaluation_config (ModelEvaluationConfig, optional): Configuration object.

        Raises:
            MyException: If initialization fails.
        """
        try:
            if model_evaluation_config is None:
                params: dict = read_yaml_file(filepath="params.yaml")
                model_training_params: dict = params.get("model_training", {})

                self.model_evaluation_config: ModelEvaluationConfig = (
                    ModelEvaluationConfig(
                        target=str(model_training_params["target"]),
                    )
                )
            else:
                self.model_evaluation_config: ModelEvaluationConfig = (
                    model_evaluation_config
                )

        except Exception as e:
            raise MyException(e, sys) from e

    def setup_experiment(
        self, dagshub_uri: str, dagshub_repo: str, dagshub_username: str
    ) -> None:
        """
        Sets up MLflow tracking, DagsHub context, and the MLflow experiment name,
        temporarily suppressing informational logging during setup.

        Args:
            dagshub_uri (str): URI for DagsHub MLflow tracking.
            dagshub_repo (str): Name of the DagsHub repository.
            dagshub_username (str): Username of the DagsHub account.

        Raises:
            MyException: If experiment setup fails.
        """

        try:
            logging.getLogger().setLevel(logging.WARNING)

            with Halo(text="Configuring experiment...", spinner="dots"):
                mlflow.set_tracking_uri(dagshub_uri)

                with redirect_stdout(io.StringIO()):
                    dagshub.init(
                        repo_name=dagshub_repo,
                        repo_owner=dagshub_username,
                        mlflow=True,
                    )

                mlflow.set_experiment("Model Evaluation")

        except Exception as e:
            raise MyException(e, sys) from e

        finally:
            logging.getLogger().setLevel(logging.INFO)

    def _evaluate_model(
        self, model: LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict:
        """
        Calculates performance metrics for the model.

        Args:
            model (LogisticRegression): The trained scikit-learn model.
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): True labels for the test set.

        Returns:
            Dict[str, float]: Dictionary of calculated metrics.

        Raises:
            MyException: If evaluation fails.
        """
        try:
            with Halo(
                text="Evaluating model...",
                spinner="dots",
            ):
                y_hat: ndarray = model.predict(X_test)
                y_hat_proba: ndarray = model.predict_proba(X_test)

            metrics: dict = {
                "accuracy": accuracy_score(y_test, y_hat),
                "precision": precision_score(y_test, y_hat, zero_division=0),
                "recall": recall_score(y_test, y_hat, zero_division=0),
                "f1_score": f1_score(y_test, y_hat, zero_division=0),
                "log_loss": log_loss(y_test, y_hat_proba[:, 1]),
                "roc_auc": roc_auc_score(y_test, y_hat_proba[:, 1]),
            }
            return metrics

        except Exception as e:
            raise MyException(e, sys) from e

    def run_mlflow_experiment(self) -> Tuple[str, str]:
        """
        Runs the core MLflow experiment logic: loads model/data, evaluates,
        logs parameters, metrics, model, and saves local reports.

        Returns:
            Tuple[str, str]: A tuple containing the metrics_filepath and model_info_filepath.

        Raises:
            MyException: If the MLflow experiment run fails.
        """
        try:
            with mlflow.start_run() as run:
                logging.info("Fetching model...")
                model_filepath: str = self.model_evaluation_config.model_filepath
                model: LogisticRegression = load_object(filepath=model_filepath)

                processed_test_filepath: str = (
                    self.model_evaluation_config.processed_test_filepath
                )
                logging.info("Fetching processed test data...")
                processed_test_data: pd.DataFrame = read_csv_file(
                    filepath=processed_test_filepath
                )

                target: str = self.model_evaluation_config.target
                X_test: pd.DataFrame = processed_test_data.drop(columns=[target])
                y_test: pd.Series = processed_test_data[target].values

                metrics: dict = self._evaluate_model(
                    model=model, X_test=X_test, y_test=y_test
                )

                logging.info("Logging metrics...")
                mlflow.log_metrics(metrics=metrics)

                if hasattr(model, "get_params"):
                    params: dict = model.get_params()
                    mlflow.log_params(params=params)

                logging.info("Logging model...")
                input_slice: pd.DataFrame = X_test[:5]
                signature = infer_signature(
                    model_input=input_slice,
                    model_output=model.predict(input_slice),
                )

                with Halo(
                    text="Logging model...",
                    spinner="dots",
                ):
                    mlflow.sklearn.log_model(
                        model,
                        "model",
                        signature=signature,
                        input_example=input_slice,
                    )

                model_info: dict = {
                    "run_id": run.info.run_id,
                    "model_path": model_filepath,
                }

                logging.info("Saving metrics...")
                metrics_filepath: str = self.model_evaluation_config.metrics_filepath
                save_as_json(data=metrics, filepath=metrics_filepath, indent=4)

                logging.info("Saving model info...")
                model_info_filepath: str = (
                    self.model_evaluation_config.model_info_filepath
                )
                save_as_json(data=model_info, filepath=model_info_filepath, indent=4)

                logging.info("Logging artifacts...")
                mlflow.log_artifact(metrics_filepath)

                return metrics_filepath, model_info_filepath

        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:
        """
        Orchestrates the model evaluation process: loads data, computes metrics,
        and logs everything to MLflow.

        Returns:
            ModelEvaluationArtifacts: Artifacts object containing paths to generated reports.

        Raises:
            MyException: If model evaluation initiation fails.
        """
        try:
            logging.info("Starting model evaluation...")
            dagshub_uri: str = os.getenv(DAGSHUB_URI)
            dagshub_repo: str = os.getenv(DAGSHUB_REPO)
            dagshub_username: str = os.getenv(DAGSHUB_USERNAME)

            logging.info("Configuring experiment...")
            self.setup_experiment(
                dagshub_uri=dagshub_uri,
                dagshub_repo=dagshub_repo,
                dagshub_username=dagshub_username,
            )

            logging.info("Running MLFlow run...")
            metrics_filepath, model_info_filepath = self.run_mlflow_experiment()

            model_evaluation_artifacts: ModelEvaluationArtifacts = (
                ModelEvaluationArtifacts(
                    metrics_filepath=metrics_filepath,
                    model_info_filepath=model_info_filepath,
                )
            )

            logging.info(
                f"Metrics filepath: {model_evaluation_artifacts.metrics_filepath}, Model info filepath: {model_evaluation_artifacts.model_info_filepath}"
            )

            logging.info("Model Evaluation complete.")
            return model_evaluation_artifacts

        except Exception as e:
            logging.error(
                f"Error during ModelEvaluation initiation: {e}", exc_info=True
            )
            raise MyException(e, sys) from e


def main():
    """
    Entry point for running data ingestion as a standalone script.

    Returns:
        ModelEvaluationArtifacts: Artifact containing the metrics & model info filepath.

    Raises:
        MyException: If the model evaluation pipeline fails.
    """
    try:
        model_evaluator: ModelEvaluation = ModelEvaluation()
        model_evaluation_artifacts: ModelEvaluationArtifacts = (
            model_evaluator.initiate_model_evaluation()
        )

        return model_evaluation_artifacts

    except Exception as e:
        raise MyException(e, sys) from e


if __name__ == "__main__":
    main()
