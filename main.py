import sys

# from src.pipeline.training_pipeline import TrainPipeline
from src.utils.main_utils import remove_pycache
from src.exception import MyException


def main():
    try:
        remove_pycache(".")

        # train_pipeline = TrainPipeline()
        # train_pipeline.run_pipeline()

    except Exception as e:
        raise MyException(e, sys) from e


if __name__ == "__main__":
    main()
