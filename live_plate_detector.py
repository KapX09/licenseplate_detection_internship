import cv2
import time
import json
import datetime
import os
import torch
import numpy as np
import easyocr
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm

# ===================== CONFIG =====================
MODEL_PATH = "models/best.pt"
CONFIDENCE = 0.5
IOU = 0.45
INPUT_FOLDER = "test_images"
JSON_FILE = "output/captured_plates.json"
SAVE_ANNOTATED = True
ANNOTATED_FOLDER = "output/annotated"

Path("output").mkdir(exist_ok=True)
Path(ANNOTATED_FOLDER).mkdir(exist_ok=True)

# ===================== LOAD MODELS =====================
print("Loading models...")

device = 'cpu'
model = YOLO(MODEL_PATH)
model.to(device)

print("Loading EasyOCR...")
reader = easyocr.Reader(['en'], gpu=False)   # Set True if you have GPU

print("Models loaded!\n")

# ===================== SAFE PREPROCESSING =====================
def safe_preprocess(crop):
    """Very safe preprocessing - minimal disturbance"""
    if crop is None or crop.size == 0:
        return None
    
    # 1. Grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # 2. Resize to fixed height while keeping aspect ratio (best for OCR)
    target_height = 200          # You can try 180 or 220 later
    aspect = gray.shape[1] / gray.shape[0]
    new_width = int(target_height * aspect)
    
    gray = cv2.resize(gray, (new_width, target_height), interpolation=cv2.INTER_CUBIC)
    
    # 3. Very mild sharpening (helps edges without creating noise)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)
    gray = cv2.filter2D(gray, -1, kernel)
    
    return gray

# ===================== HELPER FUNCTIONS =====================
def clean_plate_text(text):
    """Safe cleaning for Indian plates"""
    if not text:
        return ""
    
    text = text.upper().strip()
    text = text.replace(" ", "").replace("~", "").replace("|", "").replace("-", "") \
               .replace(".", "").replace(":", "").replace(";", "")
    
    # Only safe replacements
    text = text.replace("O", "0").replace("I", "1").replace("Z", "2")
    
    text = "".join(c for c in text if c.isalnum())
    if len(text) > 12:
        text = text[:12]
    
    return text

def process_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not read {image_path}")
        return None, None, 0
    
    start = time.time()
    
    # YOLO Plate Detection
    results = model(frame, conf=CONFIDENCE, iou=IOU, verbose=False)
    
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_conf = float(box.conf[0])
            
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            
            # Safe Preprocessing
            processed = safe_preprocess(crop)
            
            # EasyOCR
            results_ocr = reader.readtext(processed, detail=1, paragraph=False)
            
            text = ""
            ocr_conf = 0.0
            if results_ocr:
                text = results_ocr[0][1]
                ocr_conf = results_ocr[0][2]
            
            clean_text = clean_plate_text(text)
            
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'text': clean_text,
                'ocr_conf': ocr_conf,
                'plate_conf': plate_conf
            })
    
    fps = 1.0 / (time.time() - start) if (time.time() - start) > 0 else 0
    return frame, detections, fps

def draw_results(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"{det['text']} ({det['ocr_conf']:.2f})"
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame

# ===================== MAIN =====================
if __name__ == "__main__":
    image_files = [f for f in os.listdir(INPUT_FOLDER) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print(f"No images found in '{INPUT_FOLDER}' folder!")
    else:
        print(f"Found {len(image_files)} images. Running EasyOCR with safe preprocessing...\n")
        
        for img_name in tqdm(image_files, desc="Processing"):
            img_path = os.path.join(INPUT_FOLDER, img_name)
            frame, detections, fps = process_image(img_path)
            
            if frame is None:
                continue
                
            print(f"{img_name} | FPS: {fps:.1f}")
            
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
                    
                    print(f"   → {det['text']} (OCR Conf: {det['ocr_conf']:.2f})")
            
            annotated = draw_results(frame.copy(), detections)
            if SAVE_ANNOTATED:
                cv2.imwrite(os.path.join(ANNOTATED_FOLDER, f"annotated_{img_name}"), annotated)
            
            cv2.imshow(f"Result - {img_name}", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        print("\nProcessing finished! Check output/captured_plates.json")
