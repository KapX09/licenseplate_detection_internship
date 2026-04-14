# TES + ocr + preprocessing
import cv2
import time
import json
import datetime
import os
import torch
import numpy as np
import easyocr
import pytesseract
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

USE_GPU = True                    # Change to False if GPU causes issues

Path("output").mkdir(exist_ok=True)
Path(ANNOTATED_FOLDER).mkdir(exist_ok=True)

# ===================== LOAD MODELS =====================
print("Loading models...")

device = 'cuda' if torch.cuda.is_available() and USE_GPU else 'cpu'
print(f"Using device for YOLO: {device}")

model = YOLO(MODEL_PATH)
model.to(device)

print("Loading EasyOCR...")
reader = easyocr.Reader(['en'], gpu=USE_GPU)

print("Loading Tesseract...")
# Note: Make sure Tesseract is installed on your system

print(f"Models loaded! (GPU = {USE_GPU})\n")

# ===================== SAFE PREPROCESSING =====================
def safe_preprocess(crop):
    if crop is None or crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    target_height = 200
    aspect = gray.shape[1] / gray.shape[0]
    new_width = int(target_height * aspect)
    gray = cv2.resize(gray, (new_width, target_height), interpolation=cv2.INTER_CUBIC)
    
    # Mild sharpening
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    gray = cv2.filter2D(gray, -1, kernel)
    return gray

# ===================== SMART CLEANING =====================
def smart_clean_plate_text(text):
    if not text:
        return ""
    text = text.upper().strip()
    text = text.replace(" ", "").replace("~", "").replace("|", "").replace("-", "") \
               .replace(".", "").replace(":", "").replace(";", "").replace(",", "")
    
    text = text.replace("O", "0").replace("I", "1").replace("Z", "2")
    text = text.replace("B", "8") if "B" in text else text
    text = text.replace("S", "5") if "S" in text else text
    text = text.replace("4", "1") if "4" in text[2:6] else text
    text = text.replace("2", "7") if "2" in text[3:7] else text
    
    text = "".join(c for c in text if c.isalnum())
    if len(text) > 12:
        text = text[:12]
    return text

# ===================== DUAL OCR FUNCTION =====================
def get_best_ocr(crop):
    processed = safe_preprocess(crop)
    if processed is None:
        return "", 0.0
    
    # EasyOCR
    easy_results = reader.readtext(processed, detail=1, paragraph=False)
    easy_text = easy_results[0][1] if easy_results else ""
    easy_conf = easy_results[0][2] if easy_results else 0.0
    
    # Tesseract
    tess_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    tess_text = pytesseract.image_to_string(processed, config=tess_config).strip()
    tess_conf = 0.7 if len(tess_text) >= 8 else 0.4
    
    # Clean both
    clean_easy = smart_clean_plate_text(easy_text)
    clean_tess = smart_clean_plate_text(tess_text)
    
    # Choose the better one (prefer longer and higher confidence)
    if len(clean_easy) >= len(clean_tess) and easy_conf > 0.3:
        return clean_easy, easy_conf
    else:
        return clean_tess, tess_conf

def process_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        return None, None, 0
    
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
            
            text, ocr_conf = get_best_ocr(crop)
            
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'text': text,
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
        print("No images found in test_images folder!")
    else:
        print(f"Found {len(image_files)} images. Running Dual OCR (EasyOCR + Tesseract) with GPU={USE_GPU}...\n")
        
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
                    
                    print(f"   → {det['text']} (OCR: {det['ocr_conf']:.2f})")
            
            annotated = draw_results(frame.copy(), detections)
            if SAVE_ANNOTATED:
                cv2.imwrite(os.path.join(ANNOTATED_FOLDER, f"annotated_{img_name}"), annotated)
            
            cv2.imshow(f"Result - {img_name}", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        print("\nDual OCR detection completed!")
