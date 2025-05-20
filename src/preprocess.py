import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def clean_and_encode(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Cleans the DataFrame by removing unwanted columns and encoding the target variable.

    Parameters:
        df (pd.DataFrame): Raw DataFrame.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features and encoded target.
    """
    logging.info("Cleaning and encoding data...")

    # Drop unnecessary columns
    columns_to_drop = [col for col in ['id', 'Unnamed: 32'] if col in df.columns]
    df.drop(columns=columns_to_drop, inplace=True)
    for col in columns_to_drop:
        logging.info(f"Dropped column: {col}")

    # Encode diagnosis (M = 1, B = 0)
    if 'diagnosis' not in df.columns:
        raise ValueError("Expected 'diagnosis' column not found.")
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    logging.info("Mapped 'diagnosis' column to binary format.")

    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']

    return X, y

def scale_features(X: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Applies standard scaling to features.

    Parameters:
        X (pd.DataFrame): Input feature matrix.

    Returns:
        Tuple[pd.DataFrame, StandardScaler]: Scaled features and fitted scaler.
    """
    logging.info("Applying standard scaling to features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logging.info("Feature scaling complete.")
    return pd.DataFrame(X_scaled, columns=X.columns), scaler

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, StandardScaler]:
    """
    Cleans, encodes, and scales the raw data.

    Parameters:
        df (pd.DataFrame): Raw dataset.

    Returns:
        Tuple: (scaled features, target vector, fitted scaler)
    """
    X, y = clean_and_encode(df)
    X_scaled, scaler = scale_features(X)
    return X_scaled, y, scaler
