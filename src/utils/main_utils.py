import os
import sys
import json
import shutil
from yaml import safe_load
from typing import Any, Dict
from datetime import datetime
from src.exception import MyException
from pandas import read_csv, DataFrame
from dill import load as dill_load, dump as dill_dump
from numpy import load as numpy_load, save as numpy_save, ndarray


def get_current_timestamp() -> str:
    """
    Get the current timestamp formatted as 'day-month-year_hour-minute-second'.

    Returns:
        str: Formatted current timestamp.

    Raises:
        Exception: If timestamp generation fails.
    """
    timestamp: str = datetime.now().strftime("%d-%b-%y_%H-%M-%S")
    return timestamp


def read_csv_file(filepath: str, **kwargs: Any) -> DataFrame:
    """
    Read a CSV file into a pandas DataFrame.

    Args:
        filepath (str): Path to the CSV file.
        **kwargs: Additional keyword arguments passed to pandas.read_csv().

    Returns:
        DataFrame: Loaded data as a pandas DataFrame.

    Raises:
        MyException: If reading the CSV file fails.
    """
    try:
        data: DataFrame = read_csv(filepath, **kwargs)
        return data

    except Exception as e:
        raise MyException(e, sys) from e


def save_df_as_csv(df: DataFrame, filepath: str, **kwargs: Any) -> None:
    """
    Save a pandas DataFrame as a CSV file.

    Args:
        df (DataFrame): DataFrame to save.
        filepath (str): Location where the CSV will be saved.
        **kwargs: Additional keyword arguments for pandas.DataFrame.to_csv().

    Raises:
        MyException: If saving the DataFrame fails.
    """
    try:
        directory: str = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        df.to_csv(filepath, **kwargs)

    except Exception as e:
        raise MyException(e, sys) from e


def read_yaml_file(filepath: str = "params.yaml") -> Any:
    """
    Read and parse a YAML file safely.

    Args:
        filepath (str): Full path to the YAML file.

    Returns:
        Any: Parsed data from the YAML file.

    Raises:
        MyException: If reading or parsing fails.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = safe_load(f)
        return data

    except Exception as e:
        raise MyException(e, sys) from e


def load_object(filepath: str, **kwargs: Any) -> Any:
    """
    Load a Python object using dill from a file.

    Args:
        filepath (str): File path to load the object from.
        **kwargs: Additional keyword arguments forwarded to dill.load().

    Returns:
        Any: The loaded Python object.

    Raises:
        MyException: If loading fails.
    """
    try:
        with open(filepath, "rb") as f:
            obj = dill_load(f, **kwargs)

        return obj

    except Exception as e:
        raise MyException(e, sys) from e


def save_object(obj: Any, filepath: str, **kwargs: Any) -> None:
    """
    Save a Python object to file using dill.

    Args:
        obj (Any): Python object to serialize and save.
        filepath (str): File path where to save the object.
        **kwargs: Additional keyword arguments forwarded to dill.dump().

    Raises:
        MyException: If saving fails.
    """
    try:
        directory: str = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(filepath, "wb") as f:
            dill_dump(obj, f, **kwargs)

    except Exception as e:
        raise MyException(e, sys) from e


def load_numpy_array(filepath: str, **kwargs: Any) -> ndarray:
    """
    Load a NumPy array from a binary file.

    Args:
        filepath (str): Path to the .npy binary file.
        **kwargs: Additional keyword arguments forwarded to numpy.load().

    Returns:
        numpy.ndarray: Loaded NumPy array.

    Raises:
        MyException: If loading fails.
    """
    try:
        with open(filepath, "rb") as f:
            arr: ndarray = numpy_load(f, **kwargs)
        return arr

    except Exception as e:
        raise MyException(e, sys) from e


def save_numpy_array(np_array: ndarray, filepath: str, **kwargs: Any) -> None:
    """
    Save a NumPy array to a binary file.

    Args:
        np_array (numpy.ndarray): NumPy array to save.
        filepath (str): Path where the .npy file will be saved.
        **kwargs: Additional keyword arguments forwarded to numpy.save().

    Raises:
        MyException: If saving fails.
    """
    try:
        directory: str = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(filepath, "wb") as f:
            numpy_save(f, np_array, **kwargs)

    except Exception as e:
        raise MyException(e, sys) from e


def save_as_json(data: Dict[str, Any], filepath: str, **kwargs: Any) -> None:
    """
    Save a dictionary as a JSON file.

    Args:
        data (Dict[str, Any]): Dictionary to save.
        filepath (str): Location where the JSON file will be saved.
        **kwargs: Additional keyword arguments for json.dump().

    Raises:
        MyException: If saving fails.
    """
    try:
        directory: str = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, **kwargs)

    except Exception as e:
        raise MyException(e, sys) from e


def remove_pycache(root_dir: str) -> int:
    """
    Recursively remove all `__pycache__` directories under a given root.

    Args:
        root_dir (str): Root directory to search for `__pycache__` folders.

    Returns:
        int: Number of `__pycache__` directories successfully deleted.

    Raises:
        MyException: If a non-recoverable error occurs while walking the directory tree.
    """
    try:
        deleted_count: int = 0

        for dirpath, dirnames, _ in os.walk(root_dir, topdown=False):
            if "__pycache__" in dirnames:
                pycache_path = os.path.join(dirpath, "__pycache__")
                try:
                    shutil.rmtree(pycache_path)
                    deleted_count += 1
                except OSError as e:
                    raise MyException(e, sys) from e

        return deleted_count

    except Exception as e:
        raise MyException(e, sys) from e
