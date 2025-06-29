from fastapi import FastAPI
import requests
import numpy as np

app = FastAPI()

@app.post("/fusion")
def fusion(data: dict):
    # Llama a la API del modelo GBoost
    gboost_response = requests.post(
        "https://gboost-api.vercel.app/predict", 
        json={"features": data["features"]}
    ).json()

    # Llama a la API del modelo CNN
    cnn_response = requests.post(
        "https://cnn-api.vercel.app/predict", 
        files={"file": data["image"]}
    ).json()

    # Obtén las probabilidades de cada API
    gboost_probabilities = gboost_response["probabilities"]
    cnn_probabilities = cnn_response["probabilities"]

    # Calcula el promedio de probabilidades
    averaged_probabilities = [
        (g + c) / 2 for g, c in zip(gboost_probabilities, cnn_probabilities)
    ]

    # Determina la clase final
    class_idx = int(np.argmax(averaged_probabilities))
    class_names = ["Healthy", "Parkinson"]
    final_class = class_names[class_idx]

    return {
        "gboost_probabilities": gboost_probabilities,
        "cnn_probabilities": cnn_probabilities,
        "averaged_probabilities": averaged_probabilities,
        "final_class": final_class
    }
