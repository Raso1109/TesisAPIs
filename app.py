from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import io
import tensorflow as tf  # Assuming your CNN is a Keras model

app = FastAPI(title="Fused Parkinson Detection API")

gboost_model = None
cnn_model = None

def load_models():
    global gboost_model, cnn_model
    if gboost_model is None:
        gboost_model = joblib.load("models/best_parkinsons_model.joblib")
    if cnn_model is None:
        cnn_model = tf.keras.models.load_model("models/parkinson_spiral_cnn_82_f1.keras")

# List of features expected by the Gradient Boost model
FEATURE_NAMES = ['Age', 'Gender', 'Ethnicity', 'EducationLevel', 'BMI',
    'Smoking', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality',
    'FamilyHistoryParkinsons', 'TraumaticBrainInjury', 'Hypertension', 'Diabetes',
    'Depression', 'Stroke', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal',
    'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides', 'UPDRS', 'MoCA',
    'FunctionalAssessment', 'Tremor', 'Rigidity', 'Bradykinesia',
    'PosturalInstability', 'SpeechProblems', 'SleepDisorders', 'Constipation']

@app.post("/predict")
async def predict(
    # Structured inputs from form
    **data: str,
    image: UploadFile = File(...)
):
    try:
        # Parse form data into ordered array for Gradient Boost
        load_models()
        input_data = [float(data[feature]) for feature in FEATURE_NAMES]
        df = pd.DataFrame([input_data], columns=FEATURE_NAMES)
        gboost_pred = gboost_model.predict_proba(df)[0][1]

        # Read and preprocess image for CNN
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((224, 224))  # Adjust to your CNN's expected size
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        cnn_pred = cnn_model.predict(img_array)[0][0]

        # Fuse both predictions (simple average)
        final_pred = (gboost_pred + cnn_pred) / 2

        return {
            "gboost_probability": round(gboost_pred, 4),
            "cnn_probability": round(cnn_pred, 4),
            "fused_prediction": round(final_pred, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
