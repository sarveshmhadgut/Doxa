import sys
from src.pipeline.training_pipeline import TrainPipeline
from src.exception import MyException


def main():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()

    except Exception as e:
        raise MyException(e, sys) from e


if __name__ == "__main__":
    main()
