"""
preprocess.py
Handles feature encoding and train/test splitting.
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def encode_features(df: pd.DataFrame, categorical_cols=None) -> pd.DataFrame:
    """One-hot encode categorical columns."""
    if categorical_cols is None:
        categorical_cols = ['make', 'model']
    df_encoded = pd.get_dummies(df, columns=categorical_cols)
    return df_encoded


def split_data(df_encoded: pd.DataFrame, target_col: str = 'price',
                test_size: float = 0.2, random_state: int = 42):
    """Split into train/test feature and target sets."""
    X = df_encoded.drop(target_col, axis=1)
    y = df_encoded[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
