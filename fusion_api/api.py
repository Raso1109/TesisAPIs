from fastapi import FastAPI
import requests

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de Fusión. Usa el endpoint /fusion para predecir."}


@app.post("/fusion")
def fusion(data: dict):
    # Llama al modelo GBoost
    gboost_response = requests.post(
        "https://gboost-api.vercel.app/predict", 
        json={"features": data["features"]}
    ).json()
    
    # Llama al modelo CNN
    cnn_response = requests.post(
        "https://cnn-api.vercel.app/predict", 
        json={"image": data["image"]}
    ).json()
    
    # Obtén las predicciones
    gboost_prediction = gboost_response["prediction"]
    cnn_prediction = cnn_response["prediction"]

    # Calcula el promedio (averaging)
    averaged_prediction = [
        (g + c) / 2 for g, c in zip(gboost_prediction, cnn_prediction)
    ]

    return {
        "gboost_prediction": gboost_prediction,
        "cnn_prediction": cnn_prediction,
        "averaged_prediction": averaged_prediction
    }

