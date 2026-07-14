"""
data_loader.py
Loads the car dataset. Currently uses a small sample dataset,
but is structured so you can swap in a real CSV easily.
"""

import os
import pandas as pd

SAMPLE_DATA = {
    'make': ['Toyota', 'Honda', 'Ford', 'BMW', 'Audi', 'Toyota', 'Honda', 'Ford', 'BMW', 'Audi'],
    'model': ['Corolla', 'Civic', 'F-150', '3 Series', 'A4', 'Camry', 'Accord', 'Mustang', '5 Series', 'A6'],
    'year': [2015, 2017, 2018, 2016, 2015, 2018, 2016, 2015, 2017, 2018],
    'mileage': [50000, 30000, 40000, 60000, 45000, 35000, 55000, 65000, 30000, 20000],
    'price': [15000, 17000, 25000, 27000, 30000, 16000, 18000, 26000, 28000, 31000]
}


def load_data(csv_path: str = None) -> pd.DataFrame:
    """
    Load the dataset.

    If csv_path is provided and exists, load from CSV.
    Otherwise, fall back to the built-in sample dataset.
    """
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Loaded dataset from {csv_path} ({len(df)} rows)")
    else:
        df = pd.DataFrame(SAMPLE_DATA)
        print(f"Loaded built-in sample dataset ({len(df)} rows)")
    return df


if __name__ == "__main__":
    df = load_data()
    print(df)
