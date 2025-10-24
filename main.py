from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import numpy as np
import io
from PIL import Image
from pydantic import BaseModel
from typing import List # Make sure List is imported
import datetime

# --- Pydantic Models for Response ---

class Detection(BaseModel):
    """A single detected object."""
    class_name: str
    confidence: float
    bbox: List[int]  # <-- NEW: We will add the bounding box coordinates

class DetectionResponse(BaseModel):
    """The full API response."""
    timestamp: datetime.datetime
    person_count: int
    detected_objects: List[Detection]

# --- API Setup ---
app = FastAPI(
    title="Real-Time Occupancy Counter API",
    description="An API that uses YOLOv8 to detect and count people in an image.",
    version="1.0"
)

# --- Model Loading ---
print("Loading YOLOv8 model...")
model = YOLO('yolov8n.pt') 
print("Model loaded successfully.")

PERSON_CLASS_ID = -1
for class_id, class_name in model.names.items():
    if class_name == 'person':
        PERSON_CLASS_ID = class_id
        print(f"Found 'person' class ID: {PERSON_CLASS_ID}")
        break
if PERSON_CLASS_ID == -1:
    print("WARNING: 'person' class not found in model vocabulary.")

# --- API Endpoint ---
@app.post("/detect/", response_model=DetectionResponse)
async def detect_people(file: UploadFile = File(...)):
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Invalid image file."}

    results = model(frame, verbose=False)
    result = results[0]

    person_count = 0
    detected_objects_list = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        
        if class_id == PERSON_CLASS_ID and confidence > 0.5:
            person_count += 1
            
            # --- NEW: Get box coordinates ---
            # Get the coordinates in [x1, y1, x2, y2] format
            box_coords_tensor = box.xyxy[0]
            
            # Convert tensor to a list of integers
            box_coords = [int(coord) for coord in box_coords_tensor.cpu().numpy()]
            # --- END NEW ---

            detected_objects_list.append(
                Detection(
                    class_name="person", 
                    confidence=confidence,
                    bbox=box_coords  # <-- NEW: Add coordinates to the response
                )
            )

    response_data = DetectionResponse(
        timestamp=datetime.datetime.now(),
        person_count=person_count,
        detected_objects=detected_objects_list
    )
    
    return response_data

# --- Root Endpoint (for testing) ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the YOLOv8 Occupancy Counter API!"}