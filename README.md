# Real-Time ML Inference REST API & Capstone

A production-style **customer churn machine learning API** built with **FastAPI**, packaged with **Docker**, and protected with automated **Pytest** tests.

This project satisfies the capstone requirements:
- FastAPI `/predict` endpoint accepting JSON
- Prediction probabilities and confidence in the response
- Pinned dependencies
- Dockerfile for containerization
- Unit tests for schemas and response codes
- End-to-end ML architecture documentation

## 1. Project Architecture

```text
Client / Postman / Browser
          |
          | JSON POST /predict
          v
     FastAPI Server
          |
          v
   Pydantic Validation
          |
          v
   Champion ML Pipeline
   ├── StandardScaler
   ├── OneHotEncoder
   └── Classifier
          |
          v
  Churn Probability
          |
          v
 JSON Prediction Response
```

## 2. ML Model

The repository contains a trained **champion model** at:

```text
models/champion_churn_model.joblib
```

The training script compares:
- Logistic Regression
- Random Forest

The model with the highest ROC-AUC on the held-out test set is saved as the champion model.

The data is deterministic synthetic customer data, so the project is self-contained and does not require a private/company dataset.

## 3. Input Features

| Feature | Type | Example |
|---|---|---|
| age | integer | 35 |
| tenure_months | integer | 12 |
| monthly_charges | float | 79.90 |
| support_tickets | integer | 3 |
| late_payments | integer | 1 |
| contract_type | string | Month-to-month |
| internet_service | string | Fiber optic |
| payment_method | string | Electronic check |
| paperless_billing | string | Yes |

## 4. API Endpoints

### GET `/health`
Checks whether the API and model are ready.

### GET `/model-info`
Returns model name, version, evaluation metrics and feature information.

### POST `/predict`
Accepts customer information and returns:
- `prediction`
- `churn_probability`
- `confidence`
- `model_name`
- `model_version`

## 5. Example Request

```json
{
  "age": 35,
  "tenure_months": 12,
  "monthly_charges": 79.90,
  "support_tickets": 3,
  "late_payments": 1,
  "contract_type": "Month-to-month",
  "internet_service": "Fiber optic",
  "payment_method": "Electronic check",
  "paperless_billing": "Yes"
}
```

Example response:

```json
{
  "prediction": "CHURN",
  "churn_probability": 0.78,
  "confidence": 0.78,
  "model_name": "random_forest",
  "model_version": "1.0.0"
}
```

The exact probability depends on the trained model.

## 6. Run From Starting to End

### Step 1 — Create/activate a virtual environment

Windows CMD:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### Step 2 — Install dependencies

```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3 — Train/retrain the champion model

```cmd
python scripts\train_model.py
```

This creates:

```text
models\champion_churn_model.joblib
```

### Step 4 — Start the API

```cmd
uvicorn app.main:app --reload
```

You should see the server running at:

```text
http://127.0.0.1:8000
```

### Step 5 — Open Swagger documentation

Open:

```text
http://127.0.0.1:8000/docs
```

Swagger UI allows you to test `/predict` without installing Postman.

### Step 6 — Test health

Open:

```text
http://127.0.0.1:8000/health
```

Expected:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Step 7 — Test `/predict`

In Swagger:
1. Open `POST /predict`
2. Click **Try it out**
3. Paste the example JSON
4. Click **Execute**
5. Check the prediction and probability

### Step 8 — Run automated tests

Open another terminal while the API is not required:

```cmd
pytest -q
```

All tests should pass.

## 7. Docker

Build the image:

```cmd
docker build -t churn-ml-api .
```

Run the container:

```cmd
docker run -p 8000:8000 churn-ml-api
```

Then open:

```text
http://localhost:8000/docs
```

Or use Docker Compose:

```cmd
docker compose up --build
```

## 8. Why This Is Production-Style

- Request validation with Pydantic
- Separate API schemas and application code
- Serialized model artifact
- Health endpoint
- Model information endpoint
- Prediction endpoint
- Pinned package versions
- Docker container
- Automated unit tests
- Clear project structure
- API documentation through FastAPI Swagger
- Model version returned with every prediction

## 9. Project Structure

```text
Real-Time-ML-Inference-REST-API/
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
├── models/
│   └── champion_churn_model.joblib
├── scripts/
│   └── train_model.py
├── tests/
│   ├── __init__.py
│   └── test_api.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## 10. Capstone Deliverables

| Requirement | Implementation |
|---|---|
| FastAPI microservice | `app/main.py` |
| `/predict` endpoint | Implemented |
| JSON request | Pydantic `CustomerFeatures` |
| Prediction probability | `churn_probability` |
| Dockerfile | `Dockerfile` |
| Pinned dependencies | `requirements.txt` |
| Unit tests | `tests/test_api.py` |
| Champion model | `models/champion_churn_model.joblib` |
| Architecture documentation | This README |
| API documentation | FastAPI `/docs` |

## 11. Final Result

The completed system takes customer information as JSON, validates it, sends it through the trained champion ML pipeline, and returns a real-time churn prediction with probability and confidence.

**Technologies:** Python, FastAPI, Pydantic, Scikit-learn, Joblib, Pytest, Docker.
