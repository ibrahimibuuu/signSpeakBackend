from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
import uvicorn
import numpy as np
import cv2
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths for models
YOLO_MODEL_PATH = os.path.join(os.getcwd(), "models", "best.pt")  # Corrected path for YOLO model
SIGNBERT_MODEL_NAME = "bert-base-uncased"  # Replace with actual Hugging Face model name or local path

# Load YOLO model
try:
    if not os.path.exists(YOLO_MODEL_PATH):
        raise FileNotFoundError(f"YOLO model not found at {YOLO_MODEL_PATH}")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("✅ YOLO model loaded")
    print("📦 YOLO Classes:", yolo_model.names)
except Exception as e:
    print(f"❌ Failed to load YOLO model: {e}")

# Load SignBERT model
try:
    signbert_tokenizer = AutoTokenizer.from_pretrained(SIGNBERT_MODEL_NAME)  # Replace with actual model name or path
    signbert_model = AutoModelForSequenceClassification.from_pretrained(SIGNBERT_MODEL_NAME)
    print("✅ SignBERT model loaded")
except Exception as e:
    print(f"❌ Failed to load SignBERT model: {e}")

# Response model for detections
class DetectionResponse(BaseModel):
    detections: list
    translation: str

@app.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        image_data = await file.read()
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Perform YOLO inference
        results = yolo_model(img)

        detections = []
        detected_classes = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])  # Class ID
                conf = float(box.conf[0])  # Confidence score
                xyxy = box.xyxy[0].tolist()  # Bounding box coordinates [x1, y1, x2, y2]
                print(f"✅ Detected: Class={cls}, Conf={conf:.2f}, Box={xyxy}")

                detections.append({
                    "class": cls,
                    "confidence": conf,
                    "box": xyxy
                })
                detected_classes.append(cls)

        # Use SignBERT for gesture-to-text translation
        if detected_classes:
            input_text = " ".join([str(cls) for cls in detected_classes])  # Convert class IDs to a string
            inputs = signbert_tokenizer(input_text, return_tensors="pt")
            outputs = signbert_model(**inputs)
            translation = str(outputs.logits.argmax(dim=1).item())  # Get the predicted class
            print(f"🧠 SignBERT Translation: {translation}")
        else:
            translation = "No gesture detected"

        return {"detections": detections, "translation": translation}

    except Exception as e:
        print(f"❌ Detection error: {e}")
        return {"detections": [], "translation": "Error during detection"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)