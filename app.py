from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import io
import tensorflow as tf
import os
import requests
import threading

app = FastAPI(title="Fused Parkinson Detection API")

# Lock for thread safety
model_lock = threading.Lock()

# List of features expected by the Gradient Boost model
FEATURE_NAMES = ['Age', 'Gender', 'Ethnicity', 'EducationLevel', 'BMI',
    'Smoking', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality',
    'FamilyHistoryParkinsons', 'TraumaticBrainInjury', 'Hypertension', 'Diabetes',
    'Depression', 'Stroke', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal',
    'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides', 'UPDRS', 'MoCA',
    'FunctionalAssessment', 'Tremor', 'Rigidity', 'Bradykinesia',
    'PosturalInstability', 'SpeechProblems', 'SleepDisorders', 'Constipation']

# Global models
gboost_model = None
cnn_model = None

def download_model_from_azure():
    model_url = "https://tesismodelo.blob.core.windows.net/cnnmodelo/parkinson_spiral_cnn_82_f1.keras"
    model_path = "models/parkinson_spiral_cnn_82_f1.keras"
    os.makedirs("models", exist_ok=True)
    if not os.path.exists(model_path):
        print("⏬ Downloading CNN model from Azure Blob...")
        r = requests.get(model_url, stream=True)
        with open(model_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("✅ Model download complete.")

def load_models():
    global gboost_model, cnn_model
    with model_lock:
        if gboost_model is None:
            gboost_model = joblib.load("models/best_parkinsons_model.joblib")
        if cnn_model is None:
            download_model_from_azure()
            cnn_model = tf.keras.models.load_model("models/parkinson_spiral_cnn_82_f1.keras")

@app.post("/predict")
async def predict(image: UploadFile = File(...), **data: str):
    try:
        load_models()

        # Gradient Boost prediction
        input_data = [float(data[feature]) for feature in FEATURE_NAMES]
        df = pd.DataFrame([input_data], columns=FEATURE_NAMES)
        gboost_pred = gboost_model.predict_proba(df)[0][1]

        # CNN prediction
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        cnn_pred = cnn_model.predict(img_array)[0][0]

        # Fusion
        final_pred = (gboost_pred + cnn_pred) / 2

        return {
            "gboost_probability": round(gboost_pred, 4),
            "cnn_probability": round(cnn_pred, 4),
            "fused_prediction": round(final_pred, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
