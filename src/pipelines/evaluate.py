import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "01_raw"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

FEATURES = ["amount", "payments", "A4", "A15", "A16"]
TARGET = "default"


def build_dataset() -> pd.DataFrame:
    loan = pd.read_csv(RAW_DIR / "loan.csv")
    account = pd.read_csv(RAW_DIR / "account.csv")
    disp = pd.read_csv(RAW_DIR / "disp.csv")
    client = pd.read_csv(RAW_DIR / "client.csv")
    district = pd.read_csv(RAW_DIR / "district.csv")

    df = loan.merge(account, on="account_id", how="left")

    disp_owner = disp[disp["type"] == "OWNER"]
    df = df.merge(disp_owner[["account_id", "client_id"]], on="account_id", how="left")

    df = df.merge(
        client[["client_id", "district_id", "birth_date", "gender"]],
        on="client_id",
        how="left",
    )

    df.rename(
        columns={
            "district_id_x": "district_id_client",
            "district_id_y": "district_id_account",
        },
        inplace=True,
    )

    df["district_mismatch"] = (
        df["district_id_client"] != df["district_id_account"]
    ).astype(int)

    df["district_id_client"] = df["district_id_client"].astype(str)
    district["district_id"] = district["district_id"].astype(str)

    df = df.merge(
        district,
        left_on="district_id_client",
        right_on="district_id",
        how="left",
    )

    df[TARGET] = df["status"].map(
        {
            "A": 0,
            "C": 0,
            "B": 1,
            "D": 1,
        }
    )

    df["A15"] = df["A15"].fillna(df["A15"].mean())

    dataset = df[FEATURES + [TARGET]].copy()
    dataset = dataset.dropna()

    return dataset


def evaluate() -> dict:
    df = build_dataset()

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        test_size=0.3,
        random_state=42,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(
        probability=True,
        class_weight="balanced",
        random_state=42,
        C=1,
        gamma="scale",
        kernel="rbf",
    )

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "model": "SVC",
        "features": FEATURES,
        "target": TARGET,
        "split": {
            "test_size": 0.3,
            "random_state": 42,
            "stratify": True,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
        "metrics": {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision_positive_class_default": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall_positive_class_default": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1_positive_class_default": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "macro_avg": {
                "precision": round(float(report["macro avg"]["precision"]), 4),
                "recall": round(float(report["macro avg"]["recall"]), 4),
                "f1_score": round(float(report["macro avg"]["f1-score"]), 4),
            },
            "weighted_avg": {
                "precision": round(float(report["weighted avg"]["precision"]), 4),
                "recall": round(float(report["weighted avg"]["recall"]), 4),
                "f1_score": round(float(report["weighted avg"]["f1-score"]), 4),
            },
        },
        "confusion_matrix": {
            "labels": ["No Default", "Default"],
            "matrix": cm.astype(int).tolist(),
        },
        "classification_report": report,
    }

    output_path = REPORTS_DIR / "model_metrics.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    acc = metrics["metrics"]["accuracy"]
    precision = metrics["metrics"]["precision_positive_class_default"]
    recall = metrics["metrics"]["recall_positive_class_default"]

    print(f"Saved metrics to {output_path}")
    print(f"Accuracy: {acc}")
    print(f"Precision default class: {precision}")
    print(f"Recall default class: {recall}")

    if not (0.70 <= acc <= 0.75):
        print("WARNING: Accuracy is outside the resume claim range of 70-75%.")

    if not (0.68 <= precision <= 0.75):
        print("WARNING: Precision is outside the resume claim range of 68-75%.")

    if not (0.68 <= recall <= 0.75):
        print("WARNING: Recall is outside the resume claim range of 68-75%.")

    return metrics


if __name__ == "__main__":
    evaluate()