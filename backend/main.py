import io
import os
import traceback
from typing import Dict, Any

import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input  # type: ignore


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "MobileNetV3Large_final_best.keras")
IMG_SIZE = (224, 224)
CLASS_NAMES = ["1", "2", "3", "4", "5"]

app = FastAPI(title="Cow Face Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None


def load_model_safely(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    print(f"Loading model from: {path}")
    print(f"Model file size: {os.path.getsize(path) / (1024 * 1024):.2f} MB")

    loaded_model = tf.keras.models.load_model(
        path,
        compile=False,
        safe_mode=False
    )
    print("Model loaded successfully")
    return loaded_model


try:
    model = load_model_safely(MODEL_PATH)
except Exception as e:
    model = None
    print("Load model failed")
    print(f"Exact error: {repr(e)}")
    traceback.print_exc()


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    arr = np.array(image, dtype=np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Cow Face API is running"}


@app.get("/health")
def health() -> Dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {
        "status": "ok",
        "model_loaded": True,
        "model_path": MODEL_PATH
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        x = preprocess_image(contents)
        preds = model.predict(x, verbose=0)[0]

        max_index = int(np.argmax(preds))
        class_name = CLASS_NAMES[max_index]
        confidence = float(preds[max_index] * 100)
        score = int(class_name)

        return {
            "class_name": class_name,
            "score": score,
            "confidence": round(confidence, 2),
            "probabilities": {
                CLASS_NAMES[i]: round(float(preds[i] * 100), 2)
                for i in range(len(CLASS_NAMES))
            }
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")