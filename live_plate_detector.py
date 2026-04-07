import cv2
import time
import json
import datetime
import os
import torch
import easyocr
import numpy as np
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
USE_GPU_OCR = True

# Create folders
Path("output").mkdir(exist_ok=True)
Path(ANNOTATED_FOLDER).mkdir(exist_ok=True)

# ===================== LOAD MODELS =====================
print("Loading models...")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device for detection: {device}")

model = YOLO(MODEL_PATH)
model.to(device)

print(f"Loading EasyOCR (GPU={USE_GPU_OCR})...")
reader = easyocr.Reader(['en'], gpu=USE_GPU_OCR)

print("Models loaded!\n")

# ===================== HELPER FUNCTIONS =====================
def clean_plate_text(text):
    """Post-processing for Indian plates"""
    if not text:
        return ""
    text = text.upper().strip()
    text = text.replace(" ", "").replace("~", "").replace("|", "").replace("-", "").replace(".", "")
    # Common fixes for Indian plates
    text = text.replace("O", "0")
    text = text.replace("I", "1")
    text = text.replace("Z", "2") if len(text) > 8 else text  # careful with Z
    # E <-> 8 heuristic (often E is misread as 8 in middle)
    if len(text) > 8 and text.count("E") > 0:
        text = text.replace("E", "8")
    return text

def process_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not read {image_path}")
        return None, None, 0
    
    start = time.time()
    
    # Plate Detection (.pt model)
    results = model(frame, conf=CONFIDENCE, iou=IOU, verbose=False)
    
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_conf = float(box.conf[0])
            
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            
            # === Improved Preprocessing ===
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            
            # Resize to good height for OCR
            target_height = 180
            aspect = gray.shape[1] / gray.shape[0]
            gray = cv2.resize(gray, (int(target_height * aspect), target_height))
            
            # CLAHE - big accuracy booster for plates
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            
            # Light sharpening
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            gray = cv2.filter2D(gray, -1, kernel)
            
            # === Improved EasyOCR ===
            ocr_results = reader.readtext(
                gray,
                detail=1,
                paragraph=False,
                decoder='greedy',
                contrast_ths=0.5,
                adjust_contrast=0.7,
                text_threshold=0.6,
                low_text=0.3,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'  # Only letters & numbers
            )
            
            text = ""
            ocr_conf = 0.0
            if ocr_results:
                raw_text = ocr_results[0][1]
                ocr_conf = ocr_results[0][2]
                text = clean_plate_text(raw_text)
            
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'text': text,
                'ocr_conf': ocr_conf,
                'plate_conf': plate_conf,
                'raw_text': raw_text if 'raw_text' in locals() else ""
            })
    
    fps = 1.0 / (time.time() - start) if (time.time() - start) > 0 else 0
    return frame, detections, fps

def draw_results(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        text = det['text']
        conf = det['ocr_conf']
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"{text} ({conf:.2f})"
        cv2.putText(frame, label, (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame

# ===================== MAIN PIPELINE =====================
if __name__ == "__main__":
    image_files = [f for f in os.listdir(INPUT_FOLDER) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print(f"No images found in '{INPUT_FOLDER}' folder!")
    else:
        print(f"Found {len(image_files)} images. Starting improved OCR...\n")
        
        for img_name in tqdm(image_files, desc="Processing"):
            img_path = os.path.join(INPUT_FOLDER, img_name)
            
            frame, detections, fps = process_image(img_path)
            if frame is None:
                continue
            
            print(f"{img_name} | FPS: {fps:.1f}")
            
            # Save to JSON
            for det in detections:
                if det['text'] and len(det['text']) >= 6:   # Indian plates are usually 8-10 chars
                    entry = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "image_name": img_name,
                        "plate_text": det['text'],
                        "ocr_confidence": float(det['ocr_conf']),
                        "plate_confidence": float(det['plate_conf']),
                        "bbox": det['bbox']
                    }
                    # Append to JSON
                    data = []
                    if os.path.exists(JSON_FILE):
                        with open(JSON_FILE, 'r') as f:
                            data = json.load(f)
                    data.append(entry)
                    with open(JSON_FILE, 'w') as f:
                        json.dump(data, f, indent=2)
                    
                    print(f"   → Saved: {det['text']} (OCR conf: {det['ocr_conf']:.2f})")
            
            # Draw and save annotated
            annotated = draw_results(frame.copy(), detections)
            if SAVE_ANNOTATED:
                cv2.imwrite(os.path.join(ANNOTATED_FOLDER, f"annotated_{img_name}"), annotated)
            
            # Show result
            cv2.imshow(f"Result: {img_name}", annotated)
            cv2.waitKey(0)   # Press any key for next image
            cv2.destroyAllWindows()
        
        print("\nProcessing finished! Check output/captured_plates.json")
