# Car Price Predictor

A simple machine learning project that predicts used car prices using a
Random Forest Regressor, based on features like make, model, year, and mileage.

## Project Structure

```
car-price-predictor/
├── data/                  # Place real CSV datasets here (optional)
├── models/                # Saved trained model (car_price_model.pkl)
├── outputs/               # Any generated reports/plots
├── src/
│   ├── data_loader.py     # Loads dataset (sample or CSV)
│   ├── preprocess.py      # Encoding + train/test split
│   ├── train_model.py     # Trains and saves the model
│   └── predict.py         # Loads saved model, predicts price for new input
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### 1. Train the model
```bash
cd src
python train_model.py
```
This prints the dataset, encoded dataset, MSE, R² score, and saves the
trained model to `models/car_price_model.pkl`.

### 2. Predict a price
```bash
python predict.py
```
Or import it in your own code:
```python
from predict import predict_price
price = predict_price(make="Honda", model_name="Civic", year=2020, mileage=15000)
print(price)
```

### Using a real dataset
Drop a CSV (with columns: make, model, year, mileage, price) into `data/`,
then run:
```python
from train_model import train
train(csv_path="../data/your_file.csv")
```

## Notes
- The sample dataset built into `data_loader.py` is tiny (10 rows) and meant
  only for demonstration — accuracy will improve significantly with a real,
  larger dataset.
