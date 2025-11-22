import os
import sys
from src.logger import logging
from src.exception import MyException
from src.utils.main_utils import read_yaml_file
from src.data.data_ingestion import DataIngestion
from src.data.data_preprocessing import DataPreprocessing
from src.features.feature_engineering import FeatureEngineering
from src.model.model_training import ModelTraining
from src.entity.config_entity import (
    DataIngestionConfig,
    DataPreprocessingConfig,
    FeatureEngineeringConfig,
    ModelTrainingConfig,
)
from src.entity.artifact_entity import (
    DataIngestionArtifacts,
    DataPreprocessingArtifacts,
    FeatureEngineeringArtifacts,
    ModelTrainingArtifacts,
)

terminal_width: int = os.get_terminal_size().columns if os.isatty(1) else 80


class TrainPipeline:
    """
    Orchestrates execution of the end-to-end training pipeline.

    Stages:
        1. Data Ingestion
        2. Data Preprocessing
        3. Feature Engineering
        4. Model Training
    """

    def __init__(self) -> None:
        """
        Initialize the TrainPipeline by constructing stage configurations
        from params.yaml.
        """
        try:
            params = read_yaml_file(filepath="params.yaml")

            self.data_ingestion_config: DataIngestionConfig = DataIngestionConfig(
                test_size=float(params["data_ingestion"]["test_size"]),
                random_state=int(params["data_ingestion"]["random_state"]),
            )

            self.data_preprocessing_config: DataPreprocessingConfig = (
                DataPreprocessingConfig(
                    features=list(params["data_preprocessing"]["features"]),
                    target=str(params["data_preprocessing"]["target"]),
                )
            )

            self.feature_engineering_config: FeatureEngineeringConfig = (
                FeatureEngineeringConfig(
                    max_features=int(params["feature_engineering"]["max_features"]),
                    feature=str(params["feature_engineering"]["feature"]),
                    target=str(params["data_preprocessing"]["target"]),
                )
            )

            self.model_training_config: ModelTrainingConfig = ModelTrainingConfig(
                target=str(params["model_training"]["target"]),
                c=float(params["model_training"]["c"]),
                solver=str(params["model_training"]["solver"]),
                penalty=str(params["model_training"]["penalty"]),
                max_iter=int(params["model_training"]["max_iter"]),
            )

        except Exception as e:
            raise MyException(e, sys) from e

    def start_data_ingestion(self) -> DataIngestionArtifacts:
        """
        Run the data ingestion stage of the pipeline.

        Returns:
            DataIngestionArtifacts: Paths to the ingested training and testing datasets.
        """
        try:
            data_ingestor: DataIngestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )

            data_ingestion_artifacts: DataIngestionArtifacts = (
                data_ingestor.initiate_data_ingestion()
            )
            return data_ingestion_artifacts

        except Exception as e:
            raise MyException(e, sys) from e

    def start_data_preprocessing(self) -> DataPreprocessingArtifacts:
        """
        Run the data preprocessing stage of the pipeline.

        Returns:
            DataPreprocessingArtifacts: Paths to the interim (preprocessed) train/test CSVs.
        """
        try:
            data_preprocessor: DataPreprocessing = DataPreprocessing(
                data_preprocessing_config=self.data_preprocessing_config
            )

            data_preprocessing_artifacts: DataPreprocessingArtifacts = (
                data_preprocessor.initiate_data_preprocessing()
            )

            return data_preprocessing_artifacts

        except Exception as e:
            raise MyException(e, sys) from e

    def start_feature_engineering(self) -> FeatureEngineeringArtifacts:
        """
        Run the feature engineering stage of the pipeline.

        Returns:
            FeatureEngineeringArtifacts: Paths to processed train/test data and vectorizer.
        """
        try:
            feature_engineer: FeatureEngineering = FeatureEngineering(
                feature_engineering_config=self.feature_engineering_config
            )

            feature_engineering_artifacts: FeatureEngineeringArtifacts = (
                feature_engineer.initiate_feature_engineering()
            )

            return feature_engineering_artifacts

        except Exception as e:
            raise MyException(e, sys) from e

    def start_model_training(self) -> ModelTrainingArtifacts:
        """
        Run the model training stage of the pipeline.

        Returns:
            ModelTrainingArtifacts: Path to the persisted trained model.
        """
        try:
            model_trainer: ModelTraining = ModelTraining(
                model_training_config=self.model_training_config
            )

            model_training_artifacts: ModelTrainingArtifacts = (
                model_trainer.initiate_model_training()
            )

            return model_training_artifacts

        except Exception as e:
            raise MyException(e, sys) from e

    def run_pipeline(self) -> None:
        """
        Execute the full training pipeline in sequence.
        """
        try:
            print("=" * terminal_width)
            logging.info("Executing training pipeline...")
            print("Executing training pipeline...")
            print("-" * terminal_width)

            # Stage 1: Data Ingestion
            logging.info("Executing Data Ingestion stage...")
            print("Stage 1: Data Ingestion")
            data_ingestion_artifacts = self.start_data_ingestion()
            print(f"Ingested train: {data_ingestion_artifacts.raw_train_filepath}")
            print(f"Ingested test: {data_ingestion_artifacts.raw_test_filepath}")
            print("-" * terminal_width)

            # Stage 2: Data Preprocessing
            logging.info("Executing Data Preprocessing stage...")
            print("Stage 2: Data Preprocessing")
            data_preprocessing_artifacts = self.start_data_preprocessing()
            print(
                f"Interim train: {data_preprocessing_artifacts.interim_train_filepath}"
            )
            print(f"Interim test: {data_preprocessing_artifacts.interim_test_filepath}")
            print("-" * terminal_width)

            # Stage 3: Feature Engineering
            logging.info("Executing Feature Engineering stage...")
            print("Stage 3: Feature Engineering")
            feature_engineering_artifacts = self.start_feature_engineering()
            print(
                f"Processed train: {feature_engineering_artifacts.processed_train_filepath}"
            )
            print(
                f"Processed test: {feature_engineering_artifacts.processed_test_filepath}"
            )
            print(f"Vectorizer: {feature_engineering_artifacts.vectorizer_filepath}")
            print("-" * terminal_width)

            # Stage 4: Model Training
            logging.info("Executing Model Training stage...")
            print("Stage 4: Model Training")
            model_training_artifacts = self.start_model_training()
            print(f"Model saved at: {model_training_artifacts.model_filepath}")
            print("-" * terminal_width)

            logging.info("Training pipeline executed successfully.")
            print("Training pipeline completed successfully.")
            print("=" * terminal_width)

        except Exception as e:
            logging.exception("Training pipeline execution failed")
            print("=" * terminal_width)
            print("Training pipeline failed!")
            print("=" * terminal_width)
            raise MyException(e, sys) from e
