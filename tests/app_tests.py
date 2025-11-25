import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import unittest
from app.app import app


class FlaskAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_home_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

        body = response.data

        # Validate title and main heading
        self.assertIn(b"<title>Doxa</title>", body)
        self.assertIn(b"Sentiment Classifier", body)

    def test_predict_page(self):
        response = self.client.post("/predict", data={"text": "I love this!"})
        self.assertEqual(response.status_code, 200)

        body = response.data

        self.assertTrue(
            b'class="result positive"' in body or b'class="result negative"' in body,
            "Response must contain a result section with positive/negative class.",
        )

        self.assertTrue(
            b"homer_from_the_bush.gif" in body or b"homer_into_bush.gif" in body,
            "Response must contain one of the sentiment GIFs.",
        )

        self.assertIn(b'alt="Sentiment Result GIF"', body)


if __name__ == "__main__":
    unittest.main()
