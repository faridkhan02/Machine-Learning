import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv(r"D:\telegram\archive\spambase.csv")

# Features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

# Train model
model = GaussianNB()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "spam_model.pkl")

print("Model saved successfully!")

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))