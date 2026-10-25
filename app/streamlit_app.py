import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import time
from dotenv import load_dotenv
from google import genai

from rag.retriever import Retriever

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def get_api_key():
    return os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")


@st.cache_resource
def load():
    model_path = ROOT / "models" / "churn_pipeline.pkl"
    if not model_path.exists():
        import subprocess

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


@st.cache_resource
def get_explainer():
    prep = pipe.named_steps["prep"]
    clf = pipe.named_steps["model"]
    df = pd.read_csv(ROOT / "data" / "raw" / "telco_churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    X = df.drop(columns=["customerID", "Churn"])
    bg = prep.transform(X.sample(100, random_state=42))
    bg = bg.toarray() if hasattr(bg, "toarray") else bg
    return shap.LinearExplainer(clf, bg), prep


def shap_reasons(row, top_n=4):
    explainer, prep = get_explainer()
    x = prep.transform(pd.DataFrame([row]))
    x = x.toarray() if hasattr(x, "toarray") else x
    vals = explainer.shap_values(x)[0]
    names = prep.get_feature_names_out()
    order = np.argsort(-vals)
    out = []
    for i in order:
        if vals[i] <= 0 or len(out) == top_n:
            continue
        name = names[i]
        if name.startswith("cat__") and x[0][i] == 0:
            continue
        label = name.replace("cat__", "").replace("num__", "")
        if label == "tenure":
            label = f"short tenure ({row['tenure']} month{'s' if row['tenure'] != 1 else ''})"
        out.append(f"{label} (+{vals[i]:.2f})")
    return out


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


@st.cache_resource
def get_retriever():
    return Retriever(get_api_key())


def retention_advice(row, proba, reasons):
    client = genai.Client(api_key=get_api_key())
    query = (
        f"{row['Contract']} {row['InternetService']} {row['PaymentMethod']} "
        f"tenure {row['tenure']} months tech support {row['TechSupport']}. "
        + ", ".join(reasons)
    )
    chunks = get_retriever().search(query, k=3)
    playbook = "\n\n".join(chunks)
    prompt = f"""You are a telecom retention specialist.
Customer: contract={row['Contract']}, tenure={row['tenure']} months,
internet={row['InternetService']}, monthly charge={row['MonthlyCharges']},
payment={row['PaymentMethod']}. Churn probability: {proba:.0%}.
Risk factors: {', '.join(reasons) or 'none'}.

Company retention playbook (use ONLY this for actions):
{playbook}

Suggest 3 short, specific retention actions based ONLY on the customer facts and the playbook.
Do not invent discounts or prices. Mention which playbook section each action comes from."""

    last_err = None
    for model in ["gemini-flash-latest", "gemini-2.5-flash-lite"]:
        for attempt in range(3):
            try:
                resp = client.models.generate_content(model=model, contents=prompt)
                return resp.text, chunks
            except Exception as e:
                last_err = e
                time.sleep(2 * (attempt + 1))
    raise last_err


if st.button("Predict churn risk"):
    proba = float(pipe.predict_proba(pd.DataFrame([row]))[0, 1])
    st.metric("Churn probability", f"{proba:.1%}")
    if proba >= THRESHOLD:
        reasons = shap_reasons(row) or get_reasons(row)
        st.error("High risk: recommend a retention action")
        st.write("**Risk factors:** " + (", ".join(reasons) or "none"))
        with st.spinner("Generating retention advice..."):
            try:
                advice, used = retention_advice(row, proba, reasons)
                st.write(advice)
                with st.expander("Playbook sections used"):
                    for c in used:
                        st.markdown(c)
            except Exception as e:
                st.warning(f"AI advice unavailable: {e}")
    else:
        st.success("Low risk")