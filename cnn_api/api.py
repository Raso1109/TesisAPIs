import os
import gdown

MODEL_PATH = "parkinson_spiral_cnn_82_f1.keras"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/file/d/1oY_A1YfelAj24jqOtsBSPqdWhfWyst_9/view?usp=sharing"
    gdown.download(url, MODEL_PATH, quiet=False)

from fastapi import FastAPI
import tensorflow as tf

app = FastAPI()
model = tf.keras.models.load_model(MODEL_PATH)

def predict_image(image_path, model):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"error": "Could not read image"}

    img = cv2.resize(img, (128, 128))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)

    prediction = model.predict(img)
    probabilities = prediction[0].tolist()  # Devuelve las probabilidades directamente

    return {"probabilities": probabilities}


# Endpoint para subir y predecir una imagen
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Guarda temporalmente la imagen
        temp_image_path = f"temp_{file.filename}"
        with open(temp_image_path, "wb") as temp_file:
            temp_file.write(await file.read())

        # Realiza la predicción
        result = predict_image(temp_image_path, model)

        # Elimina el archivo temporal
        os.remove(temp_image_path)

        return result

    except Exception as e:
        return {"error": str(e)}