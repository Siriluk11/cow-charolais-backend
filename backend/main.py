import io
import os
import traceback
from typing import Dict, Any, Optional

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input  # type: ignore

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")
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

model: Optional[tf.keras.Model] = None
model_error: Optional[str] = None


def log(*args):
    print(*args, flush=True)


def try_load_model() -> Optional[tf.keras.Model]:
    global model, model_error

    if model is not None:
        return model

    log("🚀 TRY LOAD MODEL")
    log(f"📁 BASE_DIR: {BASE_DIR}")
    log(f"📂 MODEL_PATH: {MODEL_PATH}")

    try:
        files_in_base = os.listdir(BASE_DIR)
        log(f"📄 FILES IN BASE_DIR: {files_in_base}")
    except Exception as e:
        log(f"⚠️ LIST DIR ERROR: {repr(e)}")

    if not os.path.exists(MODEL_PATH):
        model_error = f"Model file not found: {MODEL_PATH}"
        log(f"❌ {model_error}")
        return None

    try:
        file_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        log(f"📦 MODEL FILE SIZE: {file_size_mb:.2f} MB")

        loaded_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        model = loaded_model
        model_error = None
        log("✅ Model loaded successfully")
        return model

    except Exception as e:
        model = None
        model_error = repr(e)
        log("❌ Load model failed")
        log(f"❌ Exact error: {model_error}")
        traceback.print_exc()
        return None


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize(IMG_SIZE)
        arr = np.array(image, dtype=np.float32)
        arr = preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)
        return arr
    except UnidentifiedImageError:
        raise ValueError("Uploaded file is not a valid image")
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {str(e)}")


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Cow Face API is running"}


@app.get("/health")
def health() -> Dict[str, Any]:
    loaded = try_load_model()
    if loaded is None:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Model not loaded",
                "model_path": MODEL_PATH,
                "error": model_error,
            },
        )

    return {
        "status": "ok",
        "model_loaded": True,
        "model_path": MODEL_PATH,
        "img_size": IMG_SIZE,
        "class_names": CLASS_NAMES,
    }


@app.get("/debug-model")
def debug_model() -> Dict[str, Any]:
    exists = os.path.exists(MODEL_PATH)
    files = []
    try:
        files = os.listdir(BASE_DIR)
    except Exception as e:
        files = [f"LIST ERROR: {repr(e)}"]

    loaded = try_load_model()

    return {
        "base_dir": BASE_DIR,
        "model_path": MODEL_PATH,
        "model_exists": exists,
        "files_in_base_dir": files,
        "model_loaded": loaded is not None,
        "model_error": model_error,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    loaded = try_load_model()
    if loaded is None:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Model not loaded",
                "model_path": MODEL_PATH,
                "error": model_error,
            },
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        x = preprocess_image(contents)
        preds = loaded.predict(x, verbose=0)[0]

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

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log("❌ Prediction failed")
        log(f"❌ Prediction error: {repr(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")