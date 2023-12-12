import pandas as pd


def load_data(file_path):
    """
    Load a dataset from a file.

    Parameters:
        file_path (str): The path to the file containing the dataset.

    Returns:
        pandas.DataFrame: The loaded dataset.
    """
    return pd.read_csv(file_path)


def save_data(data, file_path):
    """
    Save a dataset to a file.

    Parameters:
        data (pandas.DataFrame): The dataset to be saved.
        file_path (str): The path to the file where the dataset will be saved.
    """
    data.to_csv(file_path, index=False)
