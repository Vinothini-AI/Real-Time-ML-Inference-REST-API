from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from .schemas import CustomerFeatures, PredictionResponse

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "champion_churn_model.joblib"

if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Model artifact not found at {MODEL_PATH}. Run: python scripts/train_model.py"
    )

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
MODEL_NAME = artifact["model_name"]
MODEL_VERSION = artifact["version"]
THRESHOLD = artifact.get("threshold", 0.5)

app = FastAPI(
    title="Real-Time Customer Churn ML API",
    description=(
        "Production-style FastAPI service for real-time customer churn inference. "
        "Accepts JSON customer features and returns a churn prediction and probability."
    ),
    version=MODEL_VERSION,
)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
    }

@app.get("/model-info")
def model_info() -> dict[str, Any]:
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "task": artifact.get("task"),
        "metrics": artifact.get("metrics", {}),
        "threshold": THRESHOLD,
        "features": artifact.get("feature_names", []),
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures) -> PredictionResponse:
    try:
        row = pd.DataFrame([customer.model_dump()])
        probability = float(model.predict_proba(row)[0, 1])
        prediction = "CHURN" if probability >= THRESHOLD else "NO_CHURN"
        confidence = probability if prediction == "CHURN" else 1 - probability

        return PredictionResponse(
            prediction=prediction,
            churn_probability=round(probability, 6),
            confidence=round(confidence, 6),
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
