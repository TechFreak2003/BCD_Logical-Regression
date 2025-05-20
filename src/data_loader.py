import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def load_raw_data(path: str = "data/breast_cancer_data.csv") -> pd.DataFrame:
    """
    Loads raw breast cancer dataset as a DataFrame.

    Parameters:
        path (str): Path to the dataset CSV file.

    Returns:
        pd.DataFrame: Loaded raw DataFrame.
    """
    if not os.path.exists(path):
        logging.error(f"File not found: {path}")
        raise FileNotFoundError(f"Could not find the file: {path}")
    
    logging.info(f"Loading raw data from {path}")
    df = pd.read_csv(path)
    logging.info(f"Raw data loaded with shape: {df.shape}")
    return df

def get_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits feature matrix and target into train/test sets.

    Parameters:
        X (pd.DataFrame): Features.
        y (pd.Series): Target labels.
        test_size (float): Proportion for test data.
        random_state (int): Seed for reproducibility.

    Returns:
        Tuple: (X_train, X_test, y_train, y_test)
    """
    logging.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logging.info(f"Train/Test split complete: X_train={X_train.shape}, X_test={X_test.shape}")
    return X_train, X_test, y_train, y_test
