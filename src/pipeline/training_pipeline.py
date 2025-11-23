import os
import sys
from src.exception import MyException
from src.utils.main_utils import read_yaml_file
from src.data.data_ingestion import DataIngestion
from src.data.data_preprocessing import DataPreprocessing
from src.features.feature_engineering import FeatureEngineering
from src.model.model_training import ModelTraining
from src.model.model_evaluation import ModelEvaluation
from src.entity.config_entity import (
    DataIngestionConfig,
    DataPreprocessingConfig,
    FeatureEngineeringConfig,
    ModelTrainingConfig,
    ModelEvaluationConfig,
)
from src.entity.artifact_entity import (
    DataIngestionArtifacts,
    DataPreprocessingArtifacts,
    FeatureEngineeringArtifacts,
    ModelTrainingArtifacts,
    ModelEvaluationArtifacts,
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

            self.model_evaluation_config: ModelEvaluationConfig = ModelEvaluationConfig(
                target=str(params["model_training"]["target"]),
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

    def start_model_evaluation(self) -> ModelEvaluationArtifacts:
        """
        Run the model evaluation stage of the pipeline.

        Returns:
            ModelEvaluationArtifacts: Paths to the metrics and model info reports.
        """
        try:
            model_evaluator: ModelEvaluation = ModelEvaluation(
                model_evaluation_config=self.model_evaluation_config
            )

            model_evaluation_artifacts: ModelEvaluationArtifacts = (
                model_evaluator.initiate_model_evaluation()
            )
            return model_evaluation_artifacts

        except Exception as e:
            raise MyException(e, sys) from e

    def run_pipeline(self) -> None:
        """
        Execute the full training pipeline in sequence.
        """
        try:
            print("=" * terminal_width)
            print(f"{'Training Pipeline':^{terminal_width}}")
            print("-" * terminal_width)

            # Stage 1: Data Ingestion
            print(f"{'Stage 1: Data Ingestion':^{terminal_width}}")
            data_ingestion_artifacts: DataIngestionArtifacts = (
                self.start_data_ingestion()
            )
            print("-" * terminal_width)

            # Stage 2: Data Preprocessing
            print(f"{'Stage 2: Data Preprocessing':^{terminal_width}}")
            data_preprocessing_artifacts: DataPreprocessingArtifacts = (
                self.start_data_preprocessing()
            )
            print("-" * terminal_width)

            # Stage 3: Feature Engineering
            print(f"{'Stage 3: Feature Engineering':^{terminal_width}}")
            feature_engineering_artifacts: FeatureEngineeringArtifacts = (
                self.start_feature_engineering()
            )
            print("-" * terminal_width)

            # Stage 4: Model Training
            print(f"{'Stage 4: Model Training':^{terminal_width}}")
            model_training_artifacts: ModelTrainingArtifacts = (
                self.start_model_training()
            )
            print("-" * terminal_width)

            # Stage 5: Model Evaluation
            print(f"{'Stage 5: Model Evaluation':^{terminal_width}}")
            model_evaluation_artifacts: ModelEvaluationArtifacts = (
                self.start_model_evaluation()
            )
            print("-" * terminal_width)

            print(f"{'Training pipeline completed.':^{terminal_width}}")
            print("=" * terminal_width)

        except Exception as e:
            print("=" * terminal_width)
            print(f"{'Training pipeline failed!.':^{terminal_width}}")
            print("=" * terminal_width)
            raise MyException(e, sys) from e
