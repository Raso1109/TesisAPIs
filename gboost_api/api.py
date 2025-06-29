from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

@app.route("/")
def index():
    return jsonify({"message": "Bienvenido a la API de GBoost. Usa el endpoint /predict para predecir."})


model = joblib.load("best_parkinsons_model.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)  # Ajusta si los datos son diferentes
    prediction = model.predict(features)
    return jsonify({"prediction": prediction.tolist()})

