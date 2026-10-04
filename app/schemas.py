from typing import Literal
from pydantic import BaseModel, Field

class CustomerFeatures(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Customer age in years")
    tenure_months: int = Field(..., ge=0, le=120, description="Customer tenure in months")
    monthly_charges: float = Field(..., ge=0, le=1000, description="Monthly customer charges")
    support_tickets: int = Field(..., ge=0, le=100, description="Number of support tickets")
    late_payments: int = Field(..., ge=0, le=100, description="Number of late payments")
    contract_type: Literal["Month-to-month", "One year", "Two year"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    payment_method: Literal["Electronic check", "Bank transfer", "Credit card"]
    paperless_billing: Literal["Yes", "No"]

class PredictionResponse(BaseModel):
    prediction: Literal["CHURN", "NO_CHURN"]
    churn_probability: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    model_name: str
    model_version: str
