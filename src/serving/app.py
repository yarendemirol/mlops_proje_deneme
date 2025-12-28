import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# 1. FastAPI Uygulamasını Başlat
app = FastAPI(title="Ad Click Prediction Service - MLOps Edition")

# 2. Model Yolunu Belirle ve Yükle
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "final_deployment_model.pkl")

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model başarıyla yüklendi: {MODEL_PATH}")
else:
    # Eğer model dosyası yoksa hata fırlat (Resilience için önemli)
    raise FileNotFoundError(f"❌ Model dosyası bulunamadı: {MODEL_PATH}")


# 3. Request Şeması (Döküman III.3 - Stateless Serving)
class PredictionRequest(BaseModel):
    Daily_Time_Spent_on_Site: float
    Age: float
    Area_Income: float
    Daily_Internet_Usage: float
    Male: int


@app.post("/predict")
def predict(request: PredictionRequest):
    """
    Döküman III.3: Stateless Serving Pattern
    Döküman II.1: Feature Hashing Alignment
    """
    # Gelen veriyi sözlükten DataFrame'e çevir
    raw_data = pd.DataFrame([request.model_dump()])

    # --- FEATURE ALIGNMENT (Kritik Bölüm) ---
    # Modelin eğitimde gördüğü tüm kolon listesini (Hash ve Cross dahil) al
    expected_features = list(model.feature_names_in_)

    # Reindex kullanarak:
    # 1. Eksik olan yüzlerce Hash kolonunu tek seferde ekle.
    # 2. Hepsine 0 (neutral) değerini ata.
    # 3. Fragmented DataFrame uyarısını (PerformanceWarning) engelle.
    final_input = raw_data.reindex(columns=expected_features, fill_value=0)

    # Scikit-learn TypeError: ['str', 'str_'] hatasını önlemek için
    # tüm kolon isimlerini string türüne sabitle
    final_input.columns = [str(col) for col in final_input.columns]

    # 4. TAHMİN (Prediction)
    prediction = model.predict(final_input)[0]
    probability = model.predict_proba(final_input)[0].tolist()

    # 5. SONUÇ DÖNDÜR
    return {
        "clicked_on_ad": int(prediction),
        "prediction_label": "Clicked" if prediction == 1 else "Not Clicked",
        "probability_scores": {
            "not_clicked": round(probability[0], 4),
            "clicked": round(probability[1], 4)
        },
        "metadata": {
            "model_type": "Ensemble (XGB+RF)",
            "status": "success",
            "info": "Processed via MLOps Feature Alignment Layer"
        }
    }


# 6. Server Başlatma
if __name__ == "__main__":
    import uvicorn

    print("🚀 Server başlatılıyor...")
    print("👉 Swagger UI: http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)