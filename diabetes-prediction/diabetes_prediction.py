"""
=====================================================================
 DIABETES PREDICTION PROJECT — Logistic Regression (Pima Indians)
=====================================================================
A complete, professional end-to-end ML pipeline:
  1. Load Data
  2. Exploratory Data Analysis (EDA)
  3. Data Cleaning & Preprocessing (handle hidden missing values)
  4. Feature Scaling
  5. Train/Test Split (stratified)
  6. Model Training (Logistic Regression)
  7. Model Evaluation (accuracy, confusion matrix, report, ROC-AUC)
  8. Feature Importance
  9. Save the trained model + scaler for later use
 10. Reusable prediction function for new patients
=====================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, roc_auc_score, precision_recall_curve
)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# STEP 1: LOAD DATA
# ---------------------------------------------------------------------------
def load_data():
    """Load the Pima Indians Diabetes dataset from the UCI mirror."""
    url = ("https://raw.githubusercontent.com/jbrownlee/Datasets/"
           "master/pima-indians-diabetes.data.csv")
    column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                     'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    df = pd.read_csv(url, names=column_names)
    print(f"[INFO] Dataset loaded successfully. Shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ---------------------------------------------------------------------------
def explore_data(df, save_dir="."):
    print("\n" + "=" * 70)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    print("\nFirst 5 rows:\n", df.head())
    print("\nData types & non-null counts:")
    df.info()
    print("\nStatistical summary:\n", df.describe())

    print("\nClass balance (Outcome):")
    print(df['Outcome'].value_counts(normalize=True).rename("proportion"))

    # These columns cannot legitimately be 0 -> 0 really means "missing"
    zero_invalid_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    print("\nCount of biologically-impossible zero values (hidden missing data):")
    print((df[zero_invalid_cols] == 0).sum())

    # Class distribution plot
    plt.figure()
    sns.countplot(x='Outcome', data=df, palette="Set2")
    plt.title("Class Distribution (0 = No Diabetes, 1 = Diabetes)")
    plt.xlabel("Outcome")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/eda_class_distribution.png", dpi=120)
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(9, 7))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/eda_correlation_heatmap.png", dpi=120)
    plt.close()

    # Feature distributions
    df.drop(columns='Outcome').hist(bins=20, figsize=(12, 8), color="#4C72B0")
    plt.suptitle("Feature Distributions")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/eda_feature_distributions.png", dpi=120)
    plt.close()

    print("\n[INFO] EDA plots saved: eda_class_distribution.png, "
          "eda_correlation_heatmap.png, eda_feature_distributions.png")

    return zero_invalid_cols


# ---------------------------------------------------------------------------
# STEP 3: DATA CLEANING & PREPROCESSING
# ---------------------------------------------------------------------------
def preprocess_data(df, zero_invalid_cols):
    print("\n" + "=" * 70)
    print("STEP 3: DATA CLEANING & PREPROCESSING")
    print("=" * 70)

    df_clean = df.copy()

    # Replace invalid zeros with NaN, then impute using the median
    # (median is robust to outliers, grouped by Outcome for a more
    # realistic imputation than a single global median)
    for col in zero_invalid_cols:
        df_clean[col] = df_clean[col].replace(0, np.nan)

    print("\nMissing values before imputation:\n", df_clean.isnull().sum())

    for col in zero_invalid_cols:
        df_clean[col] = df_clean.groupby('Outcome')[col].transform(
            lambda x: x.fillna(x.median())
        )

    print("\nMissing values after imputation:\n", df_clean.isnull().sum())

    X = df_clean.drop('Outcome', axis=1)
    y = df_clean['Outcome']
    return X, y


# ---------------------------------------------------------------------------
# STEP 4: TRAIN / TEST SPLIT + FEATURE SCALING
# ---------------------------------------------------------------------------
def split_and_scale(X, y):
    print("\n" + "=" * 70)
    print("STEP 4: TRAIN/TEST SPLIT & FEATURE SCALING")
    print("=" * 70)

    # Stratify keeps the same class ratio in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"[INFO] Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns


# ---------------------------------------------------------------------------
# STEP 5: MODEL TRAINING (with hyperparameter tuning)
# ---------------------------------------------------------------------------
def train_model(X_train, y_train):
    print("\n" + "=" * 70)
    print("STEP 5: MODEL TRAINING (Logistic Regression + GridSearchCV)")
    print("=" * 70)

    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['lbfgs']
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        param_grid, cv=cv, scoring='roc_auc', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    print(f"[INFO] Best parameters: {grid_search.best_params_}")
    print(f"[INFO] Best CV ROC-AUC score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


# ---------------------------------------------------------------------------
# STEP 6: MODEL EVALUATION
# ---------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test, save_dir="."):
    print("\n" + "=" * 70)
    print("STEP 6: MODEL EVALUATION")
    print("=" * 70)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print(f"ROC-AUC Score: {auc:.4f}")
    print("\nConfusion Matrix:\n", conf_matrix)
    print("\nClassification Report:\n", class_report)

    # Confusion matrix heatmap
    plt.figure()
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Diabetes", "Diabetes"],
                yticklabels=["No Diabetes", "Diabetes"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/eval_confusion_matrix.png", dpi=120)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc:.3f})", color="#C44E52")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/eval_roc_curve.png", dpi=120)
    plt.close()

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure()
    plt.plot(recall, precision, color="#55A868")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/eval_precision_recall_curve.png", dpi=120)
    plt.close()

    print("\n[INFO] Evaluation plots saved: eval_confusion_matrix.png, "
          "eval_roc_curve.png, eval_precision_recall_curve.png")

    return {"accuracy": accuracy, "auc": auc}


# ---------------------------------------------------------------------------
# STEP 7: FEATURE IMPORTANCE
# ---------------------------------------------------------------------------
def plot_feature_importance(model, feature_names, save_dir="."):
    print("\n" + "=" * 70)
    print("STEP 7: FEATURE IMPORTANCE (Model Coefficients)")
    print("=" * 70)

    coefs = pd.Series(model.coef_[0], index=feature_names).sort_values()
    print("\nFeature coefficients (impact on log-odds of diabetes):\n", coefs)

    plt.figure(figsize=(8, 6))
    colors = ["#C44E52" if c < 0 else "#55A868" for c in coefs.values]
    coefs.plot(kind="barh", color=colors)
    plt.title("Feature Importance (Logistic Regression Coefficients)")
    plt.xlabel("Coefficient value")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/feature_importance.png", dpi=120)
    plt.close()

    print("[INFO] Plot saved: feature_importance.png")


# ---------------------------------------------------------------------------
# STEP 8: SAVE MODEL & SCALER
# ---------------------------------------------------------------------------
def save_artifacts(model, scaler, save_dir="."):
    joblib.dump(model, f"{save_dir}/diabetes_logreg_model.pkl")
    joblib.dump(scaler, f"{save_dir}/diabetes_scaler.pkl")
    print(f"\n[INFO] Model saved to {save_dir}/diabetes_logreg_model.pkl")
    print(f"[INFO] Scaler saved to {save_dir}/diabetes_scaler.pkl")


# ---------------------------------------------------------------------------
# STEP 9: PREDICT ON NEW PATIENT DATA
# ---------------------------------------------------------------------------
def predict_new_patient(model, scaler, patient_dict, feature_names):
    """
    patient_dict example:
    {
        'Pregnancies': 2, 'Glucose': 130, 'BloodPressure': 70,
        'SkinThickness': 25, 'Insulin': 100, 'BMI': 28.5,
        'DiabetesPedigreeFunction': 0.5, 'Age': 35
    }
    """
    patient_df = pd.DataFrame([patient_dict])[feature_names]
    patient_scaled = scaler.transform(patient_df)
    prediction = model.predict(patient_scaled)[0]
    probability = model.predict_proba(patient_scaled)[0, 1]

    result = "Diabetic" if prediction == 1 else "Not Diabetic"
    print(f"\n[PREDICTION] Result: {result} | Probability of diabetes: {probability:.2%}")
    return result, probability


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def main():
    df = load_data()
    zero_invalid_cols = explore_data(df)
    X, y = preprocess_data(df, zero_invalid_cols)
    X_train, X_test, y_train, y_test, scaler, feature_names = split_and_scale(X, y)

    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    plot_feature_importance(model, feature_names)
    save_artifacts(model, scaler)

    # Example usage: predicting a new patient's diabetes risk
    print("\n" + "=" * 70)
    print("STEP 9: SAMPLE PREDICTION ON A NEW PATIENT")
    print("=" * 70)
    sample_patient = {
        'Pregnancies': 2, 'Glucose': 130, 'BloodPressure': 70,
        'SkinThickness': 25, 'Insulin': 100, 'BMI': 28.5,
        'DiabetesPedigreeFunction': 0.5, 'Age': 35
    }
    predict_new_patient(model, scaler, sample_patient, feature_names)

    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETE — Final Test Accuracy: {metrics['accuracy']*100:.2f}%, "
          f"ROC-AUC: {metrics['auc']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
