import os
import sys
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Proje ana dizinini yola ekle
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

app = FastAPI(title="Ad Click Prediction Service")

# Model dosyasını yükle
MODEL_PATH = os.path.join(BASE_DIR, "final_deployment_model.pkl")
model = joblib.load(MODEL_PATH)


# İstek (Request) şeması - Ödevdeki yüksek kardinalite ve feature gereksinimlerine göre
class PredictionRequest(BaseModel):
    Daily_Time_Spent_on_Site: float
    Age: int
    Area_Income: float
    Daily_Internet_Usage: float
    Male: int
    # Not: Modelinizde hangi featurelar varsa buraya eklemelisiniz


@app.get("/")
def home():
    return {"message": "Ad Click Prediction API is running!"}


@app.post("/predict")
def predict(request: PredictionRequest):
    # Gelen veriyi DataFrame'e çevir
    input_data = pd.DataFrame([request.dict()])

    # Model eğitimi sırasında yapılan 'Age_Bucket' gibi feature engineering
    # işlemlerini burada da (input üzerinde) yapmanız gerekebilir.
    # Örnek:
    if 'Age' in input_data.columns:
        import numpy as np
        input_data['Age_Bucket'] = pd.cut(input_data['Age'],
                                          bins=[-np.inf, -0.5, 0.5, np.inf],
                                          labels=[0, 1, 2]).astype(int)

    # Tahmin yap
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0].tolist()

    return {
        "clicked_on_ad": int(prediction),
        "probability": probability,
        "status": "success"
    }