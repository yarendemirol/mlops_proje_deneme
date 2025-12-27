import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Ad Click Prediction Service")

# Modelin kaydedildiği yerden yüklenmesi [cite: 43]
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "final_deployment_model.pkl")
model = joblib.load(MODEL_PATH)


class PredictionRequest(BaseModel):
    Daily_Time_Spent_on_Site: float
    Age: float
    Area_Income: float
    Daily_Internet_Usage: float
    Male: int


@app.post("/predict")
def predict(request: PredictionRequest):
    """Döküman III.3: Stateless Serving Pattern [cite: 43]"""
    input_data = pd.DataFrame([request.dict()])

    # Tahmin yap
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0].tolist()

    return {
        "clicked_on_ad": int(prediction),
        "probability": probability,
        "status": "success"
    }