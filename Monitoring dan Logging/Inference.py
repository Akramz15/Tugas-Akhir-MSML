from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import time
import os
import logging

# Mengimpor modul metrics yang baru saja dibuat
from Monitoring_dan_Logging.prometheus_exporter import REQUEST_COUNT, PREDICTION_COUNT, LATENCY, start_metrics_server

# Setup FastAPI app
app = FastAPI(title="Churn Prediction API", version="1.0")

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Memuat Model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../Membangun_model/model_artefak.pkl")

try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model berhasil dimuat dari %s", MODEL_PATH)
except Exception as e:
    logger.error("Gagal memuat model: %s", e)
    model = None

# Menjalankan server metrics di port 8001 saat aplikasi FastAPI mulai
@app.on_event("startup")
async def startup_event():
    start_metrics_server(port=8001)

class CustomerData(BaseModel):
    CreditScore: int
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
    Geography_Germany: int
    Geography_Spain: int
    Gender_Male: int

from fastapi.responses import RedirectResponse

@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")

@app.post("/predict")
def predict_churn(data: CustomerData):
    # Mencatat jumlah request (Metric Prometheus)
    REQUEST_COUNT.inc()
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model belum dimuat!")
    
    # Menghitung latensi prediksi (Metric Prometheus)
    start_time = time.time()
    
    try:
        # Konversi input menjadi DataFrame
        input_data = pd.DataFrame([data.dict()])
        
        # Lakukan prediksi
        prediction = model.predict(input_data)[0]
        
        # Hitung waktu yang dibutuhkan
        latency = time.time() - start_time
        LATENCY.observe(latency)
        
        # Mencatat hasil prediksi (0 = Retained, 1 = Exited)
        PREDICTION_COUNT.labels(predicted_class=str(prediction)).inc()
        
        return {
            "prediction": int(prediction),
            "status": "Exited" if prediction == 1 else "Retained",
            "latency_seconds": latency
        }
    except Exception as e:
        logger.error("Error saat prediksi: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
