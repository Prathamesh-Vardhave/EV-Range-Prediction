import os
import pickle

import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import EVFeatures, RangePrediction

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "best_model.pkl")

app = FastAPI(
    title="EVRange Prediction API",
    description="Predicts real-world EV driving range (km) from vehicle and driving-condition features.",
    version="0.1.0",
)

model = None


@app.on_event("startup")
def load_model():
    """Load the trained pipeline (preprocessing + model) if it exists.

    The API can still start without a model present (useful while the model
    is being trained separately in Kaggle) — /predict will just return a
    clear error until best_model.pkl is added.
    """
    global model
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"No model found at {MODEL_PATH} yet — /predict will be unavailable until it is added.")


@app.get("/")
def root():
    return {"message": "EVRange Prediction API is running.", "model_loaded": model is not None}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=RangePrediction)
def predict(features: EVFeatures):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet. Train and export best_model.pkl, then restart the API.",
        )

    input_df = pd.DataFrame([features.dict()])

    try:
        prediction = model.predict(input_df)[0]
    except Exception as exc:  # noqa: BLE001 - surface a clean error to the client
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return RangePrediction(predicted_range_km=round(float(prediction), 2))
