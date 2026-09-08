from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np
import io

app = FastAPI(
    title="SmartVision-YOLO-KVG API",
    description="API Pengesanan Objek YOLOv8 untuk 3 Kelas: Book, Bottle, Phone",
    version="1.0.0"
)

# Load model YOLO
model = YOLO("yolov8n.pt")

@app.get("/")
def read_root():
    return {"message": "Selamat Datang ke SmartVision-YOLO-KVG API!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Baca fail imej daripada request
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Jalankan inferens YOLO
    results = model(img)
    
    # Ambil maklumat pengesanan (detections)
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class_id": int(box.cls[0]),
                "class_name": model.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist()
            })

    return {
        "filename": file.filename,
        "total_detections": len(detections),
        "detections": detections
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)