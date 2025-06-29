from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np

app = FastAPI()
model = tf.keras.models.load_model("parkinson_spiral_cnn_82_f1.keras")

class ImageData(BaseModel):
    image: list

@app.post("/predict")
def predict(data: ImageData):
    image_array = np.array(data.image).reshape((1, 224, 224, 3))  # Ajusta al tamaño esperado por el modelo
    prediction = model.predict(image_array).tolist()
    return {"prediction": prediction}