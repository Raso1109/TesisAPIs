from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import os
import requests

app = FastAPI()

# Google Drive direct download link
model_url = "https://drive.google.com/uc?export=download&id=1iM-BcZoexO-6Xy2HTbJXzh4S64MvGpJb"
model_path = "/tmp/parkinson_model.keras"

# Download model if not already present
if not os.path.exists(model_path):
    print("Downloading model from Google Drive...")
    response = requests.get(model_url)
    with open(model_path, "wb") as f:
        f.write(response.content)
    print("Download complete.")

# Load the model
model = tf.keras.models.load_model(model_path)

# Input schema
class ImageData(BaseModel):
    image: list

@app.post("/api/predict")
async def predict(data: ImageData):
    image_array = np.array(data.image).reshape((1, 224, 224, 3))
    prediction = model.predict(image_array).tolist()
    return {"prediction": prediction}
