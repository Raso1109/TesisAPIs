from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import joblib
import tensorflow as tf
import numpy as np
import os
import requests
from app.utils.preprocessing import preprocess_image, preprocess_features

def _init():
    app = FastAPI(title="Parkinson Detection API (Fused)")

# Cargar modelo tabular
tabular_model = joblib.load("app/models/best_parkinsons_model.joblib")

# Ruta del modelo CNN descargado
CNN_MODEL_PATH = "app/models/cnn_model.keras"

# URL del modelo en Azure Blob Storage (reemplaza con el tuyo)
CNN_MODEL_URL = os.getenv("https://tesismodelo.blob.core.windows.net/cnnmodelo/parkinson_spiral_cnn_82_f1.keras")  # puedes ponerlo en Azure App Settings

def download_model_if_not_exists():
    if not os.path.exists(CNN_MODEL_PATH):
        r = requests.get(CNN_MODEL_URL, stream=True)
        with open(CNN_MODEL_PATH, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

download_model_if_not_exists()
cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)

@app.get("/")
def read_root():
    return {"message": "Hello from Azure"}

@app.post("/predict/")
async def predict(image: UploadFile = File(...), **form_data):
    try:
        # 1. Procesar imagen
        img_bytes = await image.read()
        img_array = preprocess_image(img_bytes)
        
        # 2. Procesar características
        tabular_array = preprocess_features(form_data)
        
        # 3. Predicciones individuales
        pred_tabular = tabular_model.predict_proba([tabular_array])[0][1]
        pred_image = float(cnn_model.predict(np.expand_dims(img_array, axis=0))[0][0])
        
        # 4. Fusión por promedio
        final_pred = (pred_tabular + pred_image) / 2
        
        return {"prediction": final_pred}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
