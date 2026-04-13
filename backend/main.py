import io
import os
import traceback
from typing import Dict

import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "MobileNetV3Large_final_best.keras")
IMG_SIZE = (224, 224)
CLASS_NAMES = ["1", "2", "3", "4", "5"]


def strip_quant_config(obj):
    if isinstance(obj, dict):
        obj.pop("quantization_config", None)
        for k, v in list(obj.items()):
            obj[k] = strip_quant_config(v)
        return obj
    if isinstance(obj, list):
        return [strip_quant_config(x) for x in obj]
    return obj


def load_compatible_model(path: str):
    import zipfile
    import json
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()

    try:
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(temp_dir)

        config_path = os.path.join(temp_dir, "config.json")
        weights_path = os.path.join(temp_dir, "model.weights.h5")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        config = strip_quant_config(config)

        model = tf.keras.models.model_from_json(json.dumps(config))
        model.load_weights(weights_path)
        return model

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


app = FastAPI(title="Cow Face Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model = load_compatible_model(MODEL_PATH)
    print("โหลดโมเดลสำเร็จ")
except Exception:
    model = None
    print("โหลดโมเดลไม่สำเร็จ")
    traceback.print_exc()


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    arr = np.array(image).astype("float32")
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Cow Face API is running"}


@app.get("/health")
def health() -> Dict[str, str]:
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict:
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
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")