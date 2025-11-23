import sys
import pandas as pd
from halo import Halo
from typing import Optional
from src.logger import logging
from src.exception import MyException
from sklearn.linear_model import LogisticRegression
from src.entity.config_entity import ModelTrainingConfig
from src.entity.artifact_entity import ModelTrainingArtifacts
from src.utils.main_utils import save_object, read_csv_file, read_yaml_file


class ModelTraining:
    """
    Orchestrates model training for the sentiment classifier.

    Responsibilities:
        - Load processed training data.
        - Train Logistic Regression using configured hyperparameters.
        - Persist the trained model to disk.
        - Return artifact metadata (model filepath).
    """

    def __init__(
        self,
        model_training_config: Optional[ModelTrainingConfig] = None,
    ) -> None:
        """
        Initialize ModelTraining with an optional configuration object.

        If no configuration is provided, parameters are loaded from params.yaml
        under the 'model_training' section.

        Args:
            model_training_config (Optional[ModelTrainingConfig]):
                Optional pre-built configuration. If None, configuration will be
                constructed from params.yaml.
        """
        try:
            if model_training_config is None:
                params: dict = read_yaml_file(filepath="params.yaml")
                model_training_params: dict = params.get("model_training", {})

                self.model_training_config: ModelTrainingConfig = ModelTrainingConfig(
                    target=str(model_training_params["target"]),
                    c=float(model_training_params["c"]),
                    solver=str(model_training_params["solver"]),
                    penalty=str(model_training_params["penalty"]),
                    max_iter=int(model_training_params["max_iter"]),
                )
            else:
                self.model_training_config: ModelTrainingConfig = model_training_config

        except Exception as e:
            raise MyException(e, sys) from e

    def _train_model(
        self,
        train_df: pd.DataFrame,
        target: str,
        c: float,
        solver: str,
        penalty: str,
        max_iter: int,
    ) -> LogisticRegression:
        """
        Train a Logistic Regression model on the provided training DataFrame.

        Args:
            train_df (pd.DataFrame): Processed training data.
            target (str): Name of the target column.
            c (float): Inverse regularization strength (C) for Logistic Regression.
            solver (str): Optimization solver name.
            penalty (str): Regularization type ('l1', 'l2', etc.).
            max_iter (int): Maximum number of iterations for the solver.

        Returns:
            LogisticRegression: Fitted Logistic Regression model.

        Raises:
            MyException: If model training fails.
        """
        try:
            X_train: pd.DataFrame = train_df.drop(columns=[target])
            y_train: pd.DataFrame = train_df[target]

            model: LogisticRegression = LogisticRegression(
                C=c,
                solver=solver,
                penalty=penalty,
                max_iter=max_iter,
            )

            with Halo(text="Training model...", spinner="dots"):
                model.fit(X=X_train, y=y_train)

            return model

        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_training(self) -> ModelTrainingArtifacts:
        """
        Execute the full model training workflow:

            1. Read processed training data.
            2. Train Logistic Regression using configured hyperparameters.
            3. Save the trained model to disk.
            4. Return ModelTrainingArtifacts with the model filepath.

        Returns:
            ModelTrainingArtifacts: Artifact containing the model filepath.

        Raises:
            MyException: If model training or saving fails.
        """
        try:
            logging.info("Starting model training...")

            processed_train_filepath: str = (
                self.model_training_config.processed_train_filepath
            )
            logging.info("Fetching processed training data...")

            with Halo(text="Fetching processed training data...", spinner="dots"):
                processed_train_df: pd.DataFrame = read_csv_file(
                    processed_train_filepath
                )

            logging.info("Training model...")
            target: str = self.model_training_config.target
            c: float = self.model_training_config.c
            solver: str = self.model_training_config.solver
            penalty: str = self.model_training_config.penalty
            max_iter: int = self.model_training_config.max_iter

            model: LogisticRegression = self._train_model(
                train_df=processed_train_df,
                target=target,
                c=c,
                solver=solver,
                penalty=penalty,
                max_iter=max_iter,
            )

            model_filepath: str = self.model_training_config.model_filepath
            logging.info("Saving trained model...")
            save_object(obj=model, filepath=model_filepath)

            model_training_artifacts: ModelTrainingArtifacts = ModelTrainingArtifacts(
                model_filepath=model_filepath
            )

            logging.info(f"Model filepath: {model_training_artifacts.model_filepath}")
            logging.info("Model training complete.")
            return model_training_artifacts

        except Exception as e:
            raise MyException(e, sys) from e


def main() -> ModelTrainingArtifacts:
    """
    Entry point for running model training as a standalone script.

    Returns:
        ModelTrainingArtifacts: Artifact containing the model filepath.

    Raises:
        MyException: If the model training pipeline fails.
    """
    try:
        model_trainer: ModelTraining = ModelTraining()
        artifacts: ModelTrainingArtifacts = model_trainer.initiate_model_training()
        return artifacts

    except Exception as e:
        raise MyException(e, sys) from e


if __name__ == "__main__":
    main()
