from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
pipe = joblib.load(ROOT / "models" / "churn_pipeline.pkl")
THRESHOLD = joblib.load(ROOT / "models" / "threshold.pkl")

app = FastAPI(title="Churn Prediction API")


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(c: Customer):
    df = pd.DataFrame([c.model_dump()])
    proba = float(pipe.predict_proba(df)[0, 1])
    return {
        "churn_probability": round(proba, 4),
        "will_churn": proba >= THRESHOLD,
        "threshold": THRESHOLD,
    }