PIPELINE_NAME = ""
DATA_DIRNAME: str = "data"
MODELS_DIRNAME: str = "models"

# aws setup
DVC_BUCKET_NAME: str = "DVC_BUCKET_NAME"
DATASET_BUCKET_NAME: str = "DATASET_BUCKET_NAME"
AWS_ACCESS_KEY_ID: str = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY: str = "AWS_SECRET_ACCESS_KEY"
AWS_REGION: str = "AWS_DEFAULT_REGION"
AWS_CSV_FILENAME: str = "sample.csv"

# data ingestion
DATA_INGESTION_DIRNAME: str = "raw"
RAW_TRAIN_FILENAME: str = "raw_train.csv"
RAW_TEST_FILENAME: str = "raw_test.csv"

# data preprocessing
INTERIM_DATA_DIRNAME: str = "interim"
INTERIM_TRAIN_FILENAME: str = "interim_train.csv"
INTERIM_TEST_FILENAME: str = "interim_test.csv"

# feature engineering
PROCESSED_DATA_DIRNAME: str = "processed"
PROCESSED_TRAIN_FILENAME: str = "processed_train.csv"
PROCESSED_TEST_FILENAME: str = "processed_test.csv"
VECTORIZER_FILENAME: str = "vectorizer.pkl"

# model training
MODEL_FILENAME: str = "model.pkl"
