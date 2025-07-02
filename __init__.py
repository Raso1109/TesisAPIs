from fastapi import FastAPI
from app.utils.preprocessing import preprocess_image
import joblib
import os
import requests
import tensorflow as tf
from io import BytesIO
import base64

app = FastAPI()

# === Modelo joblib ===
MODEL_JOBLIB_PATH = os.path.join("app", "models", "best_parkinsons_model.joblib")
model_joblib = joblib.load(MODEL_JOBLIB_PATH)

# === Modelo keras ===
CNN_MODEL_URL = os.environ.get("CNN_MODEL_URL")  # <-- asegúrate de tener esta variable en Azure
CNN_MODEL_PATH = os.path.join("app", "models", "cnn_model.keras")

if not os.path.exists(CNN_MODEL_PATH):
    print("Descargando modelo CNN desde:", CNN_MODEL_URL)
    response = requests.get(CNN_MODEL_URL)
    if response.status_code == 200:
        with open(CNN_MODEL_PATH, "wb") as f:
            f.write(response.content)
    else:
        raise Exception("Error al descargar el modelo CNN")

model_cnn = tf.keras.models.load_model(CNN_MODEL_PATH)

@app.get("/")
def root():
    return {"message": "API con modelos .joblib y .keras en funcionamiento."}

@app.post("/predict/")
def predict(image_base64: str):
    # Imagen procesada (formato esperado por ambos modelos)
    image = preprocess_image(image_base64)

    # Modelo 1: sklearn
    prediction_joblib = model_joblib.predict([image])[0]

    # Modelo 2: keras
    image_cnn_input = tf.expand_dims(image, axis=0)  # batch dimension
    prediction_cnn = model_cnn.predict(image_cnn_input)[0][0]

    return {
        "prediction_joblib": int(prediction_joblib),
        "prediction_cnn": float(prediction_cnn)
    }
