import os
import sys
import unittest
import pandas as pd
from src.exception import MyException
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from src.constants import (
    DAGSHUB_TOKEN,
    DAGSHUB_USERNAME,
    DAGSHUB_URI,
    DAGSHUB_REPO,
    REGISTRATION_MODEL_NAME,
    MODELS_DIRNAME,
    VECTORIZER_FILENAME,
    DATA_DIRNAME,
    PROCESSED_DATA_DIRNAME,
    PROCESSED_TEST_FILENAME,
)
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils.main_utils import read_csv_file


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """
        Sets up the test environment by loading the model, vectorizer, and test data.

        Responsibilities:
            - Load environment variables for DagsHub/MLflow.
            - Connect to MLflow using PredictionPipeline.
            - Load the latest model from the registry using PredictionPipeline.
            - Load the vectorizer and holdout test data using utility functions.

        Returns:
            None

        Raises:
            MyException: If setup fails.
        """
        try:
            dagshub_token = os.getenv(DAGSHUB_TOKEN)
            dagshub_username = os.getenv(DAGSHUB_USERNAME)
            dagshub_uri = os.getenv(DAGSHUB_URI)
            dagshub_repo = os.getenv(DAGSHUB_REPO)

            if not dagshub_token:
                raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

            cls.prediction_pipeline = PredictionPipeline()

            cls.prediction_pipeline._connect_dagshub(
                dagshub_uri=dagshub_uri,
                dagshub_repo=dagshub_repo,
                dagshub_username=dagshub_username,
                dagshub_token=dagshub_token,
            )

            cls.new_model = cls.prediction_pipeline._load_model(REGISTRATION_MODEL_NAME)

            cls.vectorizer = cls.prediction_pipeline._load_vectorizer(
                os.path.join(MODELS_DIRNAME, VECTORIZER_FILENAME)
            )

            cls.holdout_data = read_csv_file(
                os.path.join(
                    DATA_DIRNAME, PROCESSED_DATA_DIRNAME, PROCESSED_TEST_FILENAME
                )
            )

        except Exception as e:
            raise MyException(e, sys) from e

    def test_model_loaded_properly(self) -> None:
        """
        Tests if the model is successfully loaded from the registry.

        Responsibilities:
            - Verify that the loaded model object is not None.

        Returns:
            None
        """
        self.assertIsNotNone(self.new_model)

    def test_model_signature(self) -> None:
        """
        Tests the input and output signature of the loaded model.

        Responsibilities:
            - Create a dummy input matching the vectorizer's feature set.
            - Verify that the model accepts the input.
            - Verify that the prediction output has the expected shape.

        Returns:
            None
        """
        input_text = "what do the numbers mean, mason?"
        input_data = self.vectorizer.transform([input_text])
        input_df = pd.DataFrame(
            input_data.toarray(),
            columns=[f"feature_{i}" for i in range(input_data.shape[1])],
        )

        prediction = self.new_model.predict(input_df)

        self.assertEqual(
            input_df.shape[1], len(self.vectorizer.get_feature_names_out())
        )

        self.assertEqual(len(prediction), input_df.shape[0])
        self.assertEqual(len(prediction.shape), 1)

    def test_model_performance(self) -> None:
        """
        Tests if the model meets the minimum performance thresholds.

        Responsibilities:
            - Evaluate the model on holdout test data.
            - Calculate accuracy, precision, recall, and F1 score.
            - Assert that all metrics meet or exceed defined thresholds.

        Returns:
            None
        """

        X_holdout = self.holdout_data.iloc[:, 0:-1]
        y_holdout = self.holdout_data.iloc[:, -1]

        y_hat_new = self.new_model.predict(X_holdout)

        accuracy_new = accuracy_score(y_holdout, y_hat_new)
        precision_new = precision_score(y_holdout, y_hat_new)
        recall_new = recall_score(y_holdout, y_hat_new)
        f1_new = f1_score(y_holdout, y_hat_new)

        expected_accuracy = 0.60
        expected_precision = 0.60
        expected_recall = 0.60
        expected_f1 = 0.60

        self.assertGreaterEqual(
            accuracy_new,
            expected_accuracy,
            f"Accuracy should be >= {expected_accuracy}",
        )
        self.assertGreaterEqual(
            precision_new,
            expected_precision,
            f"Precision should be >= {expected_precision}",
        )
        self.assertGreaterEqual(
            recall_new, expected_recall, f"Recall should be >= {expected_recall}"
        )
        self.assertGreaterEqual(
            f1_new, expected_f1, f"F1 score should be >= {expected_f1}"
        )


if __name__ == "__main__":
    unittest.main()
