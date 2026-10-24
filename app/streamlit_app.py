import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai

ROOT = Path(__file__).resolve().parents[1]

load_dotenv(ROOT / ".env")

@st.cache_resource
def load():
    model_path = ROOT / "models" / "churn_pipeline.pkl"
    if not model_path.exists():
        import subprocess, sys
        subprocess.run([sys.executable, str(ROOT / "src" / "train.py")], check=True)
    return (
        joblib.load(model_path),
        joblib.load(ROOT / "models" / "threshold.pkl"),
    )


pipe, THRESHOLD = load()
st.title("Telecom Churn Risk Dashboard")

c1, c2 = st.columns(2)
with c1:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    payment = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    tenure = st.slider("Tenure (months)", 0, 72, 1)
with c2:
    monthly = st.number_input("Monthly Charges", 18.0, 120.0, 70.0)
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    techsupport = st.selectbox("Tech Support", ["No", "Yes"])

row = {
    "gender": "Female", "SeniorCitizen": senior, "Partner": "No", "Dependents": "No",
    "tenure": tenure, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": internet, "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": techsupport, "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": contract, "PaperlessBilling": paperless,
    "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": monthly * tenure,
}

def get_reasons(row):
    reasons = []
    if row["Contract"] == "Month-to-month":
        reasons.append("month-to-month contract (no long-term commitment)")
    if row["tenure"] <= 12:
        reasons.append("short tenure (new customer)")
    if row["InternetService"] == "Fiber optic":
        reasons.append("fiber optic plan (higher churn segment)")
    if row["PaymentMethod"] == "Electronic check":
        reasons.append("pays by electronic check (higher churn segment)")
    if row["TechSupport"] == "No":
        reasons.append("no tech support")
    return reasons


def retention_advice(row, proba, reasons):
    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    prompt = f"""You are a telecom retention specialist.
Customer: contract={row['Contract']}, tenure={row['tenure']} months,
internet={row['InternetService']}, monthly charge={row['MonthlyCharges']},
payment={row['PaymentMethod']}. Churn probability: {proba:.0%}.
Risk factors: {', '.join(reasons) or 'none'}.
Suggest 3 short, specific retention actions based ONLY on these facts.
Do not invent discounts or prices; say "offer a discount" generally."""
    resp = client.models.generate_content(model="gemini-flash-latest", contents=prompt)
    return resp.text

if st.button("Predict churn risk"):
    proba = float(pipe.predict_proba(pd.DataFrame([row]))[0, 1])
    st.metric("Churn probability", f"{proba:.1%}")
    reasons = get_reasons(row)
    if proba >= THRESHOLD:
        st.error("High risk: recommend a retention action")
        st.write("**Risk factors:** " + (", ".join(reasons) or "none"))
        with st.spinner("Generating retention advice..."):
            try:
                st.write(retention_advice(row, proba, reasons))
            except Exception as e:
                st.warning(f"AI advice unavailable: {e}")
    else:
        st.success("Low risk")