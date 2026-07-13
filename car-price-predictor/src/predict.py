"""
predict.py
Loads the saved model and predicts price for new car data.
"""

import os
import sys
import joblib
import pandas as pd

# Ensure this script's own folder (src/) is on the path, no matter
# where this script is run from (project root, src/, VS Code "Run", etc.)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_model import MODEL_PATH


def predict_price(make: str, model_name: str, year: int, mileage: int) -> float:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. Run train_model.py first."
        )

    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    feature_columns = saved["feature_columns"]

    # Build a single-row dataframe matching training format
    row = pd.DataFrame([{
        "year": year,
        "mileage": mileage,
        f"make_{make}": 1,
        f"model_{model_name}": 1,
    }])

    # Align columns with training features (fill missing with 0)
    row = row.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(row)[0]
    return prediction


if __name__ == "__main__":
    price = predict_price(make="Toyota", model_name="Corolla", year=2019, mileage=25000)
    print(f"Predicted price: ${price:,.2f}")
