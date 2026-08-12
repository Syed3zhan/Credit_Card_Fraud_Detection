"""
Data Preprocessing & Feature Engineering Module
Handles data loading, duplicate removal, and Robust Scaling.
"""

import pandas as pd
from sklearn.preprocessing import RobustScaler


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """Loads the dataset from CSV and removes duplicate records.
    
    Args:
        file_path (str): Path to creditcard.csv file.
        
    Returns:
        pd.DataFrame: Cleaned dataframe without duplicates.
    """
    print(f"[PREPROCESSING] Loading dataset from '{file_path}'...")
    df = pd.read_csv(file_path)
    
    # Kaggle dataset has duplicate transactions that need to be dropped
    initial_shape = df.shape
    df.drop_duplicates(inplace=True)
    duplicates_removed = initial_shape[0] - df.shape[0]
    print(f"[PREPROCESSING] Dataset loaded. Removed {duplicates_removed} duplicate rows.")
    return df


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scales 'Time' and 'Amount' features using RobustScaler.
    
    Why RobustScaler:
    'Time' and 'Amount' contain extreme outliers. StandardScaler relies on mean and variance,
    which get severely skewed by outliers. RobustScaler uses median and Interquartile Range (IQR),
    making it resilient to financial transaction outliers.
    """
    print("[PREPROCESSING] Applying RobustScaler to 'Time' and 'Amount' features...")
    df_scaled = df.copy()
    scaler = RobustScaler()
    
    # Transform Time and Amount
    df_scaled["scaled_amount"] = scaler.fit_transform(df_scaled["Amount"].values.reshape(-1, 1))
    df_scaled["scaled_time"] = scaler.fit_transform(df_scaled["Time"].values.reshape(-1, 1))
    
    # Drop raw unscaled features
    df_scaled.drop(["Time", "Amount"], axis=1, inplace=True)
    
    # Move scaled columns to the front of the DataFrame
    scaled_amount = df_scaled.pop("scaled_amount")
    scaled_time = df_scaled.pop("scaled_time")
    df_scaled.insert(0, "scaled_amount", scaled_amount)
    df_scaled.insert(1, "scaled_time", scaled_time)
    
    print("[PREPROCESSING] Feature scaling completed successfully.")
    return df_scaled