import re
import sys
import string
import nltk
import pandas as pd
from src.logger import logging
from nltk.corpus import stopwords
from typing import List, Optional
from src.exception import MyException
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from src.entity.config_entity import DataPreprocessingConfig
from src.entity.artifact_entity import DataPreprocessingArtifacts
from src.utils.main_utils import read_csv_file, save_df_as_csv, read_yaml_file

try:
    nltk.download("wordnet", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
except Exception as e:
    raise MyException(e, sys) from e


class DataPreprocessing:
    """
    Handles text preprocessing for the IMDb sentiment analysis project.

    Responsibilities:
        - Load preprocessing configuration (features) from params.yaml if not provided.
        - Apply text cleaning and normalization to configured text columns.
        - Save processed train/test datasets and return artifact metadata.
    """

    def __init__(
        self,
        data_preprocessing_config: Optional[DataPreprocessingConfig] = None,
    ) -> None:
        """
        Initialize DataPreprocessing with configuration and NLTK-based utilities.

        Args:
            data_preprocessing_config (Optional[DataPreprocessingConfig]):
                Optional pre-built configuration. If None, configuration is
                loaded from params.yaml under 'data_preprocessing'.
        """
        try:
            self.lemmatizer: WordNetLemmatizer = WordNetLemmatizer()
            self.stop_words: set = set(stopwords.words("english"))

            if data_preprocessing_config is None:
                params: dict = read_yaml_file(filepath="params.yaml")
                data_preprocessing_params: dict = params.get("data_preprocessing", {})

                self.data_preprocessing_config: DataPreprocessingConfig = (
                    DataPreprocessingConfig(
                        features=list(data_preprocessing_params["features"]),
                        target=str(data_preprocessing_params["target"]),
                    )
                )
            else:
                self.data_preprocessing_config: DataPreprocessingConfig = (
                    data_preprocessing_config
                )

        except Exception as e:
            raise MyException(e, sys) from e

    def _remove_html(self, text: str) -> str:
        """
        Remove HTML tags from text.

        Args:
            text (str): Input text.

        Returns:
            str: Text with HTML tags removed.
        """
        return re.sub(r"<.*?>", " ", text)

    def _remove_urls(self, text: str) -> str:
        """
        Remove URLs from text.

        Args:
            text (str): Input text.

        Returns:
            str: Text with URLs removed.
        """
        return re.sub(r"http\S+|www\S+", " ", text)

    def _remove_punctuations(self, text: str) -> str:
        """
        Remove punctuation characters from text.

        Args:
            text (str): Input text.

        Returns:
            str: Text with punctuation removed.
        """
        return text.translate(str.maketrans("", "", string.punctuation))

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into a list of word tokens.

        Args:
            text (str): Input text.

        Returns:
            List[str]: List of tokens.
        """
        return word_tokenize(text=text)

    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove common English stop words from a list of tokens.

        Args:
            tokens (List[str]): List of input tokens.

        Returns:
            List[str]: List of tokens with stop words removed.
        """
        return [w for w in tokens if w not in self.stop_words]

    def _lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens using WordNet lemmatizer.

        Args:
            tokens (List[str]): List of input tokens.

        Returns:
            List[str]: List of lemmatized tokens.
        """
        return [self.lemmatizer.lemmatize(w) for w in tokens]

    def _preprocess_text(self, text: str) -> str:
        """
        Apply full text preprocessing pipeline to a single text value.

        Steps:
            - Cast to lowercase string.
            - Remove HTML tags, URLs, punctuation.
            - Tokenize, remove stopwords, lemmatize.

        Args:
            text (str): Input text value.

        Returns:
            str: Preprocessed text.
        """
        try:
            raw_text: str = "" if text is None else str(text)
            raw_text: str = raw_text.lower()
            raw_text: str = self._remove_html(text=raw_text)
            raw_text: str = self._remove_urls(text=raw_text)
            raw_text: str = self._remove_punctuations(text=raw_text)

            tokens: List[str] = self._tokenize(text=raw_text)
            tokens: List[str] = self._remove_stopwords(tokens=tokens)
            tokens: List[str] = self._lemmatize_tokens(tokens=tokens)

            return " ".join(tokens)

        except Exception as e:
            logging.error(f"Error preprocessing text: {e}")
            return ""

    def _preprocess_data(
        self, df: pd.DataFrame, features: List[str], target: str
    ) -> pd.DataFrame:
        """
        Apply text preprocessing to the specified feature columns in a DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.
            features (List[str]): List of column names to preprocess.

        Returns:
            pd.DataFrame: DataFrame with preprocessed text columns.

        Raises:
            MyException: If preprocessing fails.
        """
        try:
            df_processed: pd.DataFrame = df.copy()

            for feature in features:
                if feature not in df_processed.columns:
                    logging.info(
                        f"Feature '{feature}' not found in DataFrame columns. Skipping."
                    )
                    continue

                df_processed[feature] = (
                    df_processed[feature].astype(str).apply(self._preprocess_text)
                )

            df_processed[target] = df_processed[target].map(
                {"negative": 0, "positive": 1}
            )
            return df_processed

        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_data_preprocessing(self) -> DataPreprocessingArtifacts:
        """
        Run the full data preprocessing workflow:

            1. Read raw train/test CSV files.
            2. Apply text preprocessing to configured features.
            3. Save processed train/test CSVs.
            4. Build and return DataPreprocessingArtifacts.

        Returns:
            DataPreprocessingArtifacts: Paths to processed train and test CSVs.

        Raises:
            MyException: If any preprocessing step fails.
        """
        try:
            logging.info("Starting Data Preprocessing...")

            raw_train_filepath: str = self.data_preprocessing_config.raw_train_filepath
            raw_test_filepath: str = self.data_preprocessing_config.raw_test_filepath

            logging.info("Fetching raw training & testing data...")

            raw_train_data: pd.DataFrame = read_csv_file(filepath=raw_train_filepath)
            raw_test_data: pd.DataFrame = read_csv_file(filepath=raw_test_filepath)

            logging.info("Preprocessing training and testing data...")
            features: List[str] = self.data_preprocessing_config.features
            target: str = self.data_preprocessing_config.target

            interim_train_data: pd.DataFrame = self._preprocess_data(
                raw_train_data, features, target=target
            )
            interim_test_data: pd.DataFrame = self._preprocess_data(
                raw_test_data, features, target=target
            )

            interim_train_filepath: str = (
                self.data_preprocessing_config.interim_train_filepath
            )
            interim_test_filepath: str = (
                self.data_preprocessing_config.interim_test_filepath
            )

            logging.info("Saving preprocessed data to data/interim directory...")
            save_df_as_csv(
                df=interim_train_data,
                filepath=interim_train_filepath,
                index=False,
            )
            save_df_as_csv(
                df=interim_test_data,
                filepath=interim_test_filepath,
                index=False,
            )

            data_preprocessing_artifacts: DataPreprocessingArtifacts = (
                DataPreprocessingArtifacts(
                    interim_train_filepath=interim_train_filepath,
                    interim_test_filepath=interim_test_filepath,
                )
            )

            logging.info(
                f"Interim train filepath: {data_preprocessing_artifacts.interim_train_filepath}, Interim test filepath: {data_preprocessing_artifacts.interim_test_filepath}"
            )
            logging.info("Data Preprocessing complete.")

            return data_preprocessing_artifacts

        except Exception as e:
            raise MyException(e, sys) from e


def main() -> DataPreprocessingArtifacts:
    """
    Entry point for running data preprocessing as a standalone script.

    Returns:
        DataPreprocessingArtifacts: Artifact containing paths to processed train/test data.

    Raises:
        MyException: If data preprocessing fails.
    """
    try:
        data_preprocessor: DataPreprocessing = DataPreprocessing()
        data_preprocessing_artifacts: DataPreprocessingArtifacts = (
            data_preprocessor.initiate_data_preprocessing()
        )
        return data_preprocessing_artifacts

    except Exception as e:
        raise MyException(e, sys) from e


if __name__ == "__main__":
    main()
