import cv2
import time
import json
import datetime
import os
import torch
import numpy as np
import pytesseract
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm

# configuartion
MODEL_PATH = "models/best.pt"
CONFIDENCE = 0.5
IOU = 0.45
INPUT_FOLDER = "test_images"
JSON_FILE = "output/captured_plates.json"

# Output Folders
CROPPED_FOLDER = "output/cropped_plates"
ANNOTATED_FOLDER = "output/annotated_outputs"

USE_GPU = True

Path("output").mkdir(exist_ok=True)
Path(CROPPED_FOLDER).mkdir(exist_ok=True)
Path(ANNOTATED_FOLDER).mkdir(exist_ok=True)

# laoding models
print("Loading YOLO model...")
device = 'cuda' if torch.cuda.is_available() and USE_GPU else 'cpu'
model = YOLO(MODEL_PATH)
model.to(device)
print(f"Model loaded on {device}\n")

# preprocessing
def simple_preprocess(crop):
    """Only one type: Grayscale + Resize (no other filters)"""
    if crop is None or crop.size == 0:
        return None
    
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    #generalising the size of the plates
    target_height = 220
    aspect = gray.shape[1] / gray.shape[0]
    gray = cv2.resize(gray, (int(target_height * aspect), target_height), 
                      interpolation=cv2.INTER_CUBIC)
    
    return gray

#Tesseract tunning
def run_tesseract(crop, img_name):
    processed = simple_preprocess(crop)
    if processed is None:
        return "", 0.0
    
    # Save cropped plate for debugging
    crop_path = os.path.join(CROPPED_FOLDER, f"crop_{img_name}")
    cv2.imwrite(crop_path, processed)
    
    # Tuned Tesseract config for Indian plates
    config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    raw_text = pytesseract.image_to_string(processed, config=config).strip()
    
    # NO correction - as you wanted
    clean_text = raw_text.upper().replace(" ", "")
    
    conf = 0.75 if len(clean_text) >= 8 else 0.40
    
    return clean_text, conf

# annotation 
def draw_annotated(frame, detections, img_name):
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"{det['text']} ({det['ocr_conf']:.2f})"
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    annotated_path = os.path.join(ANNOTATED_FOLDER, f"annotated_{img_name}")
    cv2.imwrite(annotated_path, frame)
    return frame

if __name__ == "__main__":
    image_files = [f for f in os.listdir(INPUT_FOLDER) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print("No images found!")
    else:
        print(f"Found {len(image_files)} images. Running Simple Tesseract...\n")
        
        for img_name in tqdm(image_files, desc="Processing"):
            img_path = os.path.join(INPUT_FOLDER, img_name)
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            
            start = time.time()
            results = model(frame, conf=CONFIDENCE, iou=IOU, verbose=False)
            
            detections = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    plate_conf = float(box.conf[0])
                    
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    
                    text, ocr_conf = run_tesseract(crop, img_name)
                    
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'text': text,
                        'ocr_conf': ocr_conf,
                        'plate_conf': plate_conf
                    })
            
            fps = 1.0 / (time.time() - start) if (time.time() - start) > 0 else 0
            print(f"{img_name} | FPS: {fps:.1f}")
            
            # Save to JSON
            for det in detections:
                if det['text'] and len(det['text']) >= 6:
                    entry = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "image_name": img_name,
                        "plate_text": det['text'],
                        "ocr_confidence": float(det['ocr_conf']),
                        "plate_confidence": float(det['plate_conf']),
                        "bbox": det['bbox']
                    }
                    data = []
                    if os.path.exists(JSON_FILE):
                        with open(JSON_FILE, 'r') as f:
                            data = json.load(f)
                    data.append(entry)
                    with open(JSON_FILE, 'w') as f:
                        json.dump(data, f, indent=2)
                    
                    print(f"   → {det['text']}")
            
            # saving 
            if detections:
                annotated_frame = draw_annotated(frame.copy(), detections, img_name)
                cv2.imshow(f"Result - {img_name}", annotated_frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        print("\nDetection completed.")
        print(f"Cropped plates saved in: {CROPPED_FOLDER}")
        print(f"Annotated images saved in: {ANNOTATED_FOLDER}")
        print(f"Logs saved in: {JSON_FILE}")
