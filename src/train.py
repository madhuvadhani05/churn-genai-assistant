import joblib
import mlflow
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 0.35


def load_data():
    df = pd.read_csv(ROOT / "data" / "raw" / "telco_churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df.drop(columns=["customerID"])


def build_pipeline(num_cols, cat_cols):
    prep = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    return Pipeline([
        ("prep", prep),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def main():
    df = load_data()
    X, y = df.drop(columns=["Churn"]), df["Churn"]
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    cat_cols = [c for c in X.columns if c not in num_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    pipe = build_pipeline(num_cols, cat_cols)
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, proba)
    pr = average_precision_score(y_test, proba)
    print("ROC-AUC:", round(roc, 4))
    print("PR-AUC:", round(pr, 4))

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("churn")
    with mlflow.start_run(run_name="logreg_balanced"):
        mlflow.log_params({
            "model": "LogisticRegression",
            "class_weight": "balanced",
            "threshold": THRESHOLD,
        })
        mlflow.log_metrics({"roc_auc": roc, "pr_auc": pr})

    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(pipe, ROOT / "models" / "churn_pipeline.pkl")
    joblib.dump(THRESHOLD, ROOT / "models" / "threshold.pkl")
    print("saved model")

if __name__ == "__main__":
    main()