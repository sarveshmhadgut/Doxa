import sys
import pandas as pd
from halo import Halo
from numpy import ndarray
from src.logger import logging
from typing import Optional, Tuple, List
from src.exception import MyException
from sklearn.feature_extraction.text import TfidfVectorizer
from src.entity.config_entity import FeatureEngineeringConfig
from src.entity.artifact_entity import FeatureEngineeringArtifacts
from src.utils.main_utils import (
    save_object,
    save_df_as_csv,
    read_yaml_file,
    read_csv_file,
)


class FeatureEngineering:
    """
    Handles feature engineering / vectorization for the sentiment analysis project.

    Responsibilities:
        - Load feature engineering configuration (feature, target, max_features) from params.yaml if not provided.
        - Apply TF-IDF vectorization to the configured text feature.
        - Persist processed train/test datasets and the fitted vectorizer.
        - Return artifact metadata with filepaths for downstream stages.
    """

    def __init__(
        self,
        feature_engineering_config: Optional[FeatureEngineeringConfig] = None,
    ) -> None:
        """
        Initialize FeatureEngineering with configuration.

        Args:
            feature_engineering_config (Optional[FeatureEngineeringConfig]):
                Optional pre-built configuration. If None, configuration is
                loaded from params.yaml under 'feature_engineering' and
                'data_preprocessing' (for target).
        """
        try:
            if feature_engineering_config is None:
                params: dict = read_yaml_file(filepath="params.yaml")
                data_preprocessing_params: dict = params.get("data_preprocessing", {})
                feature_engineering_params: dict = params.get("feature_engineering", {})

                self.feature_engineering_config: FeatureEngineeringConfig = (
                    FeatureEngineeringConfig(
                        max_features=int(feature_engineering_params["max_features"]),
                        feature=str(feature_engineering_params["feature"]),
                        target=str(data_preprocessing_params["target"]),
                    )
                )
            else:
                self.feature_engineering_config: FeatureEngineeringConfig = (
                    feature_engineering_config
                )

        except Exception as e:
            raise MyException(e, sys) from e

    def _apply_tfidf(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature: str,
        target: str,
        max_features: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, TfidfVectorizer]:
        """
        Apply TF-IDF vectorization to the specified feature column.

        Args:
            train_df (pd.DataFrame): Interim training DataFrame.
            test_df (pd.DataFrame): Interim testing DataFrame.
            feature (str): Name of the text feature column to vectorize.
            target (str): Name of the target column.
            max_features (int): Maximum vocabulary size for the TF-IDF vectorizer.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, TfidfVectorizer]:
                - Processed training DataFrame (features + target)
                - Processed testing DataFrame (features + target)
                - Fitted TfidfVectorizer instance

        Raises:
            MyException: If feature engineering fails.
        """
        try:
            interim_X_train: ndarray = train_df[feature].fillna("").values
            interim_y_train: ndarray = train_df[target].values

            interim_X_test: ndarray = test_df[feature].fillna("").values
            interim_y_test: ndarray = test_df[target].values

            vectorizer: TfidfVectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
            )

            with Halo(
                text="Applying TF-IDF on interim training data...",
                spinner="dots",
            ):
                processed_X_train = vectorizer.fit_transform(interim_X_train)

            with Halo(
                text="Applying TF-IDF on interim testing data...",
                spinner="dots",
            ):
                processed_X_test = vectorizer.transform(interim_X_test)

            feature_names: List[str] = [
                f"feature_{i}" for i in range(processed_X_train.shape[1])
            ]

            processed_train_df: pd.DataFrame = pd.DataFrame(
                processed_X_train.toarray(),
                columns=feature_names,
            )
            processed_train_df["label"] = interim_y_train

            processed_test_df: pd.DataFrame = pd.DataFrame(
                processed_X_test.toarray(),
                columns=feature_names,
            )
            processed_test_df["label"] = interim_y_test

            return processed_train_df, processed_test_df, vectorizer

        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_feature_engineering(self) -> FeatureEngineeringArtifacts:
        """
        Execute the full feature engineering workflow:

            1. Read interim train/test datasets.
            2. Apply TF-IDF vectorization to the configured feature.
            3. Save processed train/test data and vectorizer.
            4. Return FeatureEngineeringArtifacts with relevant file paths.

        Returns:
            FeatureEngineeringArtifacts: Artifact object containing paths to
            processed train/test data and the saved vectorizer.

        Raises:
            MyException: If any step in the feature engineering pipeline fails.
        """
        try:
            logging.info("Starting Feature Engineering...")

            interim_train_filepath: str = (
                self.feature_engineering_config.interim_train_filepath
            )
            interim_test_filepath: str = (
                self.feature_engineering_config.interim_test_filepath
            )

            logging.info("Fetching interim training & testing data...")
            interim_train_df: pd.DataFrame = read_csv_file(
                filepath=interim_train_filepath
            )
            interim_test_df: pd.DataFrame = read_csv_file(
                filepath=interim_test_filepath
            )

            logging.info("Applying TF-IDF transformation...")
            processed_train_df, processed_test_df, vectorizer = self._apply_tfidf(
                train_df=interim_train_df,
                test_df=interim_test_df,
                feature=self.feature_engineering_config.feature,
                target=self.feature_engineering_config.target,
                max_features=self.feature_engineering_config.max_features,
            )

            processed_train_filepath: str = (
                self.feature_engineering_config.processed_train_filepath
            )
            processed_test_filepath: str = (
                self.feature_engineering_config.processed_test_filepath
            )

            logging.info("Saving processed training & testing data...")
            with Halo(
                text="Saving processed training data...",
                spinner="dots",
            ):
                save_df_as_csv(
                    df=processed_train_df,
                    filepath=processed_train_filepath,
                    index=False,
                )

            with Halo(
                text="Saving processed training data...",
                spinner="dots",
            ):
                save_df_as_csv(
                    df=processed_test_df,
                    filepath=processed_test_filepath,
                    index=False,
                )

            logging.info("Dumping TF-IDF vectorizer...")
            vectorizer_filepath: str = (
                self.feature_engineering_config.vectorizer_filepath
            )
            save_object(obj=vectorizer, filepath=vectorizer_filepath)

            feature_engineering_artifacts: FeatureEngineeringArtifacts = (
                FeatureEngineeringArtifacts(
                    processed_train_filepath=processed_train_filepath,
                    processed_test_filepath=processed_test_filepath,
                    vectorizer_filepath=vectorizer_filepath,
                )
            )

            logging.info(
                f"Processed train filepath: {feature_engineering_artifacts.processed_train_filepath}, Processed test filepath: {feature_engineering_artifacts.processed_test_filepath}, Vectorizer filepath: {feature_engineering_artifacts.vectorizer_filepath}"
            )
            logging.info("Feature Engineering complete.")
            return feature_engineering_artifacts

        except Exception as e:
            raise MyException(e, sys) from e


def main() -> FeatureEngineeringArtifacts:
    """
    Entry point for running feature engineering as a standalone script.

    Returns:
        FeatureEngineeringArtifacts: Artifact containing paths to processed train/test
        data and the saved vectorizer.

    Raises:
        MyException: If the feature engineering pipeline fails.
    """
    try:
        feature_engineer: FeatureEngineering = FeatureEngineering()
        feature_engineering_artifacts: FeatureEngineeringArtifacts = (
            feature_engineer.initiate_feature_engineering()
        )
        return feature_engineering_artifacts

    except Exception as e:
        raise MyException(e, sys) from e


if __name__ == "__main__":
    main()
