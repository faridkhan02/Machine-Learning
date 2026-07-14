"""
train_model.py
Trains a RandomForestRegressor on car data and saves the trained model.
"""

import os
import sys
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Ensure this script's own folder (src/) is on the path, no matter
# where this script is run from (project root, src/, VS Code "Run", etc.)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_data
from preprocess import encode_features, split_data

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "car_price_model.pkl")


def train(csv_path: str = None):
    # 1. Load data
    df = load_data(csv_path)
    print("\nDataset:\n", df)

    # 2. Encode categorical features
    df_encoded = encode_features(df)
    print("\nEncoded Dataset:\n", df_encoded)

    # 3. Split into train/test
    X_train, X_test, y_train, y_test = split_data(df_encoded)

    # 4. Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nMean Squared Error: {mse}")
    print(f"R-squared Score: {r2}")

    # 6. Save model + the feature columns (needed later for prediction)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({
        "model": model,
        "feature_columns": X_train.columns.tolist()
    }, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    return model, mse, r2


if __name__ == "__main__":
    train()
