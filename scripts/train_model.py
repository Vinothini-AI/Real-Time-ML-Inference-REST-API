"""
Train and save the champion churn model.

This project uses a deterministic synthetic customer dataset so the repository
can be run end-to-end without requiring a private dataset.
"""

from pathlib import Path
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "champion_churn_model.joblib"

def make_dataset(n=2500, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 75, n)
    tenure = rng.integers(0, 73, n)
    monthly_charges = np.round(rng.uniform(20, 140, n), 2)
    support_tickets = rng.poisson(2.2, n)
    late_payments = rng.poisson(0.9, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.28, 0.17])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.32, 0.48, 0.20])
    payment = rng.choice(["Electronic check", "Bank transfer", "Credit card"], n, p=[0.42, 0.30, 0.28])
    paperless = rng.choice(["Yes", "No"], n, p=[0.72, 0.28])

    score = (
        -1.0
        + 0.035 * (monthly_charges - 70)
        + 0.45 * (contract == "Month-to-month")
        - 0.55 * (contract == "Two year")
        + 0.25 * (internet == "Fiber optic")
        + 0.28 * (payment == "Electronic check")
        + 0.24 * support_tickets
        + 0.38 * late_payments
        - 0.025 * tenure
        + 0.18 * (paperless == "Yes")
        + 0.20 * (age < 25)
        + rng.normal(0, 0.65, n)
    )
    probability = 1 / (1 + np.exp(-score))
    churn = (rng.random(n) < probability).astype(int)

    import pandas as pd
    return pd.DataFrame({
        "age": age,
        "tenure_months": tenure,
        "monthly_charges": monthly_charges,
        "support_tickets": support_tickets,
        "late_payments": late_payments,
        "contract_type": contract,
        "internet_service": internet,
        "payment_method": payment,
        "paperless_billing": paperless,
    }), churn

def main():
    X, y = make_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    numeric = ["age", "tenure_months", "monthly_charges", "support_tickets", "late_payments"]
    categorical = ["contract_type", "internet_service", "payment_method", "paperless_billing"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=250, max_depth=7, min_samples_leaf=3,
            random_state=RANDOM_STATE, class_weight="balanced"
        ),
    }

    results = {}
    best_name, best_auc, best_pipeline = None, -1, None

    for name, estimator in candidates.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])
        pipeline.fit(X_train, y_train)
        prob = pipeline.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        auc = roc_auc_score(y_test, prob)
        results[name] = {
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "f1": round(f1_score(y_test, pred), 4),
            "roc_auc": round(auc, 4),
        }
        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_pipeline = pipeline

    artifact = {
        "model": best_pipeline,
        "model_name": best_name,
        "feature_names": list(X.columns),
        "threshold": 0.5,
        "metrics": results[best_name],
        "all_model_metrics": results,
        "task": "binary_customer_churn",
        "version": "1.0.0",
    }
    joblib.dump(artifact, MODEL_PATH)

    print("Champion model:", best_name)
    print("Metrics:", results[best_name])
    print("Saved to:", MODEL_PATH)

if __name__ == "__main__":
    main()
