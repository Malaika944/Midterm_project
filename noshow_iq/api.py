import os
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient

from noshow_iq.model import load_model
from noshow_iq.model import predict as model_predict

load_dotenv()

app = FastAPI(
    title="NoShowIQ",
    description="Predicts which patients will miss their appointment",
    version="0.1.0",
)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)
db = client["noshowiq"]
predictions_col = db["predictions"]
training_runs_col = db["training_runs"]

try:
    model = load_model()
    print("Model loaded successfully.")
except FileNotFoundError as e:
    model = None
    print(f"WARNING: {e}")


class AppointmentInput(BaseModel):
    age: int
    scholarship: int
    hipertension: int
    diabetes: int
    alcoholism: int
    handcap: int
    sms_received: int
    days_in_advance: int
    hour_of_booking: int


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(record: AppointmentInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run train.py first.",
        )

    input_dict = record.dict()
    df = pd.DataFrame([input_dict])
    result = model_predict(model, df)

    predictions_col.insert_one({
        "timestamp": datetime.utcnow(),
        "input": input_dict,
        "risk_level": result["risk_level"],
        "probability": result["probability"],
        "recommendation": result["recommendation"],
    })

    return result


@app.get("/history")
def history():
    docs = list(
        predictions_col.find({}, {"_id": 0})
        .sort("timestamp", -1)
        .limit(20)
    )
    return docs


@app.get("/stats")
def stats():
    pipeline = [
        {
            "$group": {
                "_id": None,
                "total_predictions": {"$sum": 1},
                "high_risk_count": {
                    "$sum": {
                        "$cond": [{"$eq": ["$risk_level", "high"]}, 1, 0]
                    }
                },
                "low_risk_count": {
                    "$sum": {
                        "$cond": [{"$eq": ["$risk_level", "low"]}, 1, 0]
                    }
                },
                "average_probability": {"$avg": "$probability"},
            }
        }
    ]

    result = list(predictions_col.aggregate(pipeline))
    last_run = training_runs_col.find_one(
        {}, {"_id": 0, "timestamp": 1},
        sort=[("timestamp", -1)]
    )

    if result:
        data = result[0]
        data.pop("_id", None)
        data["average_probability"] = round(data["average_probability"], 4)
        data["last_trained"] = (
            last_run["timestamp"].isoformat() if last_run else None
        )
        return data

    return {
        "total_predictions": 0,
        "high_risk_count": 0,
        "low_risk_count": 0,
        "average_probability": 0.0,
        "last_trained": None,
    }