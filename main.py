from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
from PIL import Image, ImageOps
import io

app = FastAPI()

# อ่านค่าจาก Render Environment
FRONTEND_URL = os.getenv("FRONTEND_URL", "*")
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_MODEL = os.getenv("ROBOFLOW_MODEL", "cow-face-detection-z49tn/2")

# เปิด CORS สำหรับ frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL] if FRONTEND_URL != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ตรวจว่า API ทำงาน
@app.get("/")
def root():
    return {
        "message": "Cow Beauty AI API is running",
        "model": ROBOFLOW_MODEL
    }

# วิเคราะห์ภาพวัว
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()

    # เปิดภาพ
    img = Image.open(io.BytesIO(image_bytes))

    # แก้ทิศภาพจากมือถือ
    img = ImageOps.exif_transpose(img)

    # แปลงเป็น RGB
    img = img.convert("RGB")

    # ลดขนาดภาพ (ช่วยให้ AI เร็วขึ้น)
    img.thumbnail((1280, 1280))

    # บันทึกภาพใหม่
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    buffer.seek(0)

    # URL Roboflow
    url = f"https://detect.roboflow.com/{ROBOFLOW_MODEL}"

    params = {
        "api_key": ROBOFLOW_API_KEY,
        "confidence": 10,
        "overlap": 30
    }

    # ส่งภาพไป AI
    response = requests.post(
        url,
        params=params,
        files={"file": ("image.jpg", buffer, "image/jpeg")}
    )

    result = response.json()

    predictions = result.get("predictions", [])
    cow_count = len(predictions)

    # ถ้าไม่พบหน้าวัว
    if cow_count == 0:
        return {
            "detected": False,
            "stars": 0,
            "message": "ไม่พบหน้าวัว",
            "predictions": predictions,
            "debug": result
        }

    # เอาความมั่นใจสูงสุด
    best_conf = max(p.get("confidence", 0) for p in predictions)

    # แปลงเป็นคะแนนดาว
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
        "message": f"พบหน้าวัว {cow_count} ตัว",
        "confidence": round(best_conf, 3),
        "predictions": predictions
    }