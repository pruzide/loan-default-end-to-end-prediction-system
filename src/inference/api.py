from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "data" / "03_models" / "best_model.pkl"
X_TRAIN_PATH = PROJECT_ROOT / "data" / "02_intermediate" / "X_train.csv"

FEATURES = ["amount", "payments", "A4", "A15", "A16"]

app = FastAPI(title="Loan Default Inference API", version="1.0.0")


class CustomerInput(BaseModel):
    amount: float
    payments: float
    A4: float
    A15: float
    A16: float


def load_runtime_objects():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")

    if not X_TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training background file not found: {X_TRAIN_PATH}")

    model = joblib.load(MODEL_PATH)
    x_train = pd.read_csv(X_TRAIN_PATH)

    scaler = StandardScaler()
    scaler.fit(x_train[FEATURES].astype(np.float64))

    return model, scaler


model, scaler = load_runtime_objects()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: CustomerInput):
    input_df = pd.DataFrame([payload.model_dump()])[FEATURES]
    scaled = scaler.transform(input_df.astype(np.float64))

    prediction = int(model.predict(scaled)[0])
    probability_default = float(model.predict_proba(scaled)[0][1])

    return {
        "prediction": prediction,
        "prediction_label": "Default" if prediction == 1 else "No Default",
        "probability_default": round(probability_default, 6),
        "features_used": FEATURES,
    }