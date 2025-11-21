import os
import shutil
from src.pipeline.training_pipeline import TrainPipeline


def remove_pycache(root_dir):
    deleted_count = 0

    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        if "__pycache__" in dirnames:
            pycache_path = os.path.join(dirpath, "__pycache__")
            try:
                shutil.rmtree(pycache_path)
                deleted_count += 1
            except OSError as e:
                print(f"Error removing {pycache_path}: {e}")


def main():
    pipeline = TrainPipeline()
    pipeline.run_pipeline()
    remove_pycache(os.getcwd())


if __name__ == "__main__":
    main()
