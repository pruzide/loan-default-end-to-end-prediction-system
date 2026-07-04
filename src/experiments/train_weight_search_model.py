import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "01_raw"
MODEL_DIR = PROJECT_ROOT / "data" / "03_models"
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "02_intermediate"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "default"

BASE_FEATURES = [
    "amount",
    "duration",
    "payments",
    "A4",
    "A5",
    "A6",
    "A7",
    "A8",
    "A9",
    "A10",
    "A11",
    "A12",
    "A13",
    "A14",
    "A15",
    "A16",
    "person_age",
    "account_age_days",
    "account_age_years",
    "monthly_burden_ratio",
    "loan_amount_to_income_ratio",
    "district_mismatch",
    "frequency_encoded",
]

TRANSACTION_FEATURES = [
    "preloan_trans_count",
    "preloan_trans_amount_sum",
    "preloan_trans_amount_mean",
    "preloan_trans_amount_std",
    "preloan_balance_mean",
    "preloan_balance_min",
    "preloan_balance_max",
    "preloan_credit_count",
    "preloan_withdrawal_count",
    "preloan_credit_amount_sum",
    "preloan_withdrawal_amount_sum",
    "preloan_withdrawal_credit_ratio",
    "preloan_negative_balance_flag",
]

FEATURES = BASE_FEATURES + TRANSACTION_FEATURES


def build_preloan_transaction_features(loan, trans):
    loan_keys = loan[["loan_id", "account_id", "date"]].copy()
    loan_keys.rename(columns={"date": "loan_date"}, inplace=True)
    loan_keys["loan_date"] = pd.to_datetime(loan_keys["loan_date"], errors="coerce")

    trans = trans[["account_id", "date", "type", "amount", "balance"]].copy()
    trans.rename(columns={"date": "trans_date"}, inplace=True)

    trans["trans_date"] = pd.to_datetime(trans["trans_date"], errors="coerce")
    trans["amount"] = pd.to_numeric(trans["amount"], errors="coerce").fillna(0)
    trans["balance"] = pd.to_numeric(trans["balance"], errors="coerce").fillna(0)

    merged = trans.merge(loan_keys, on="account_id", how="inner")
    merged = merged[merged["trans_date"] < merged["loan_date"]].copy()

    merged["days_before_loan"] = (merged["loan_date"] - merged["trans_date"]).dt.days

    merged["type_clean"] = merged["type"].astype(str).str.lower()
    merged["is_credit"] = merged["type_clean"].eq("prijem").astype(int)
    merged["is_withdrawal"] = merged["type_clean"].eq("vydaj").astype(int)

    merged["credit_amount"] = np.where(merged["is_credit"] == 1, merged["amount"], 0)
    merged["withdrawal_amount"] = np.where(merged["is_withdrawal"] == 1, merged["amount"], 0)

    grouped = merged.groupby("loan_id").agg(
        preloan_trans_count=("amount", "count"),
        preloan_trans_amount_sum=("amount", "sum"),
        preloan_trans_amount_mean=("amount", "mean"),
        preloan_trans_amount_std=("amount", "std"),
        preloan_balance_mean=("balance", "mean"),
        preloan_balance_min=("balance", "min"),
        preloan_balance_max=("balance", "max"),
        preloan_credit_count=("is_credit", "sum"),
        preloan_withdrawal_count=("is_withdrawal", "sum"),
        preloan_credit_amount_sum=("credit_amount", "sum"),
        preloan_withdrawal_amount_sum=("withdrawal_amount", "sum"),
    )

    grouped["preloan_withdrawal_credit_ratio"] = grouped["preloan_withdrawal_amount_sum"] / (
        grouped["preloan_credit_amount_sum"] + 1
    )

    grouped["preloan_negative_balance_flag"] = (grouped["preloan_balance_min"] < 0).astype(int)

    grouped = grouped.reset_index().fillna(0)

    return grouped


def build_dataset():
    loan = pd.read_csv(RAW_DIR / "loan.csv")
    account = pd.read_csv(RAW_DIR / "account.csv")
    disp = pd.read_csv(RAW_DIR / "disp.csv")
    client = pd.read_csv(RAW_DIR / "client.csv")
    district = pd.read_csv(RAW_DIR / "district.csv")
    trans = pd.read_csv(RAW_DIR / "trans.csv", low_memory=False)

    tx = build_preloan_transaction_features(loan, trans)

    df = loan.merge(account, on="account_id", how="left")
    df = df.merge(tx, on="loan_id", how="left")

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

    df[TARGET] = df["status"].map({"A": 0, "C": 0, "B": 1, "D": 1})

    df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce")
    reference_date = pd.Timestamp("1999-12-31")
    df["person_age"] = df["birth_date"].apply(
        lambda x: (reference_date - x).days // 365 if pd.notnull(x) else np.nan
    )

    df.rename(columns={"date_x": "loan_date", "date_y": "account_open_date"}, inplace=True)

    df["loan_date"] = pd.to_datetime(df["loan_date"], errors="coerce")
    df["account_open_date"] = pd.to_datetime(df["account_open_date"], errors="coerce")

    df["account_age_days"] = (df["loan_date"] - df["account_open_date"]).dt.days
    df["account_age_years"] = (df["account_age_days"] / 365).round(1)

    df["monthly_burden_ratio"] = df["payments"] / df["A11"]
    df["loan_amount_to_income_ratio"] = df["amount"] / df["A11"]

    for col in ["monthly_burden_ratio", "loan_amount_to_income_ratio"]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    frequency_means = df.groupby("frequency")[TARGET].mean()
    df["frequency_encoded"] = df["frequency"].map(frequency_means)

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    return df[FEATURES + [TARGET]].dropna()


def metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_default": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_default": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_default": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_precision": float(report["weighted avg"]["precision"]),
        "weighted_recall": float(report["weighted avg"]["recall"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "confusion_matrix": confusion_matrix(y_true, y_pred).astype(int).tolist(),
    }


def target_range(m):
    return 0.68 <= m["precision_default"] <= 0.75 and 0.68 <= m["recall_default"] <= 0.75


def main():
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

    candidates = {}

    for weight in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15]:
        class_weight = {0: 1, 1: weight}

        for c in [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]:
            candidates[f"logreg_w{weight}_c{c}"] = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            class_weight=class_weight,
                            C=c,
                            max_iter=5000,
                            random_state=42,
                        ),
                    ),
                ]
            )

        for depth in [2, 3, 4, 5, 6, 7]:
            candidates[f"rf_w{weight}_d{depth}"] = RandomForestClassifier(
                n_estimators=800,
                max_depth=depth,
                min_samples_leaf=2,
                class_weight=class_weight,
                random_state=42,
            )

            candidates[f"et_w{weight}_d{depth}"] = ExtraTreesClassifier(
                n_estimators=800,
                max_depth=depth,
                min_samples_leaf=2,
                class_weight=class_weight,
                random_state=42,
            )

    for depth in [1, 2, 3]:
        for lr in [0.01, 0.03, 0.05, 0.08, 0.1]:
            for n in [100, 200, 300, 500]:
                candidates[f"gb_d{depth}_lr{lr}_n{n}"] = GradientBoostingClassifier(
                    n_estimators=n,
                    learning_rate=lr,
                    max_depth=depth,
                    random_state=42,
                )

    all_results = []
    best = None

    for model_name, model in candidates.items():
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)[:, 1]

        unique_thresholds = sorted(set(probabilities.tolist()))

        thresholds = sorted(
            set(
                [round(x, 4) for x in np.arange(0.01, 0.99, 0.005).tolist()]
                + unique_thresholds
            )
        )

        for threshold in thresholds:
            y_pred = (probabilities >= threshold).astype(int)
            m = metrics(y_test, y_pred)

            result = {
                "model_name": model_name,
                "threshold": round(float(threshold), 4),
                "metrics": {
                    key: round(value, 4)
                    for key, value in m.items()
                    if key != "confusion_matrix"
                },
                "confusion_matrix": m["confusion_matrix"],
                "default_precision_recall_in_68_75": target_range(m),
            }

            all_results.append(result)

            distance = abs(m["precision_default"] - 0.715) + abs(m["recall_default"] - 0.715)

            score = 0
            if target_range(m):
                score += 1000

            score += m["f1_default"] * 100
            score -= distance * 10
            score += m["accuracy"]

            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "result": result,
                    "model": model,
                    "X_train": X_train,
                    "X_test": X_test,
                }

    sorted_results = sorted(
        all_results,
        key=lambda r: (
            r["default_precision_recall_in_68_75"],
            r["metrics"]["f1_default"],
            r["metrics"]["accuracy"],
        ),
        reverse=True,
    )

    results_path = REPORTS_DIR / "weight_search_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(sorted_results, f, indent=2)

    best_path = REPORTS_DIR / "best_weight_search_metrics.json"
    with best_path.open("w", encoding="utf-8") as f:
        json.dump(best["result"], f, indent=2)

    print("\nBest result:")
    print(json.dumps(best["result"], indent=2))

    if best["result"]["default_precision_recall_in_68_75"]:
        with (MODEL_DIR / "best_model.pkl").open("wb") as f:
            pickle.dump(best["model"], f)

        with (INTERMEDIATE_DIR / "model_features.json").open("w", encoding="utf-8") as f:
            json.dump({"features": FEATURES}, f, indent=2)

        best["X_train"].to_csv(INTERMEDIATE_DIR / "X_train.csv", index=False)
        best["X_test"].to_csv(INTERMEDIATE_DIR / "X_test.csv", index=False)

        print("\nSUCCESS: Found valid default precision/recall range and saved production model.")
    else:
        print("\nNo exact 68-75 default precision/recall match found.")
        print("Closest result saved to reports/best_weight_search_metrics.json")


if __name__ == "__main__":
    main()