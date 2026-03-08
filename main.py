from fastapi import FastAPI, UploadFile, File # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
import os
import requests

app = FastAPI()

FRONTEND_URL = os.getenv("FRONTEND_URL", "*")
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_MODEL = os.getenv("ROBOFLOW_MODEL", "cow-face-detection-z49tn/2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL] if FRONTEND_URL != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Cow beauty API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    url = f"https://detect.roboflow.com/{ROBOFLOW_MODEL}?api_key={ROBOFLOW_API_KEY}"
    response = requests.post(
        url,
        files={"file": ("image.jpg", image_bytes, file.content_type)} # type: ignore
    )

    result = response.json()
    predictions = result.get("predictions", [])
    count = len(predictions)

    if count == 0:
        return {
            "detected": False,
            "stars": 0,
            "message": "ไม่พบหน้าวัว",
            "predictions": []
        }

    best_conf = max(p.get("confidence", 0) for p in predictions)

    # เวอร์ชันเดโม 1-5 ดาว
    if best_conf >= 0.90:
        stars = 5
    elif best_conf >= 0.80:
        stars = 4
    elif best_conf >= 0.70:
        stars = 3
    elif best_conf >= 0.60:
        stars = 2
    else:
        stars = 1

    return {
        "detected": True,
        "stars": stars,
        "message": f"พบหน้าวัว {count} ตัว",
        "confidence": best_conf,
        "predictions": predictions
    }