from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

HIGH_RISK = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 1, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 70.0, "TotalCharges": 70.0,
}
LOW_RISK = {
    **HIGH_RISK, "tenure": 60, "InternetService": "DSL", "Contract": "Two year",
    "PaymentMethod": "Credit card (automatic)", "TechSupport": "Yes",
    "TotalCharges": 4200.0,
}


def load():
    model_path = ROOT / "models" / "churn_pipeline.pkl"
    if not model_path.exists():
        import subprocess, sys
        subprocess.run([sys.executable, str(ROOT / "src" / "train.py")], check=True)
    return joblib.load(model_path), joblib.load(ROOT / "models" / "threshold.pkl")


def test_probability_in_range():
    pipe, _ = load()
    p = pipe.predict_proba(pd.DataFrame([HIGH_RISK]))[0, 1]
    assert 0.0 <= p <= 1.0


def test_high_risk_above_low_risk():
    pipe, _ = load()
    p_high = pipe.predict_proba(pd.DataFrame([HIGH_RISK]))[0, 1]
    p_low = pipe.predict_proba(pd.DataFrame([LOW_RISK]))[0, 1]
    assert p_high > p_low


def test_high_risk_flagged_low_risk_not():
    pipe, threshold = load()
    assert pipe.predict_proba(pd.DataFrame([HIGH_RISK]))[0, 1] >= threshold
    assert pipe.predict_proba(pd.DataFrame([LOW_RISK]))[0, 1] < threshold