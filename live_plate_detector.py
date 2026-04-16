import cv2
import time
import json
import datetime
import os
import torch
import numpy as np
# import easyocr
import pytesseract
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
import hashlib

# ===================== CONFIG =====================
MODEL_PATH = "models/best.pt"
CONFIDENCE = 0.5
IOU = 0.45
INPUT_FOLDER = "test_images"
JSON_FILE = "output/captured_plates.json"
SAVE_ANNOTATED = True
ANNOTATED_FOLDER = "output/annotated"

USE_GPU = True                    # Change to False if GPU causes issues

_ocr_cache = {}

Path("output").mkdir(exist_ok=True)
Path(ANNOTATED_FOLDER).mkdir(exist_ok=True)

# ===================== LOAD MODELS =====================
print("Loading models...")

device = 'cuda' if torch.cuda.is_available() and USE_GPU else 'cpu'
print(f"Using device for YOLO: {device}")

model = YOLO(MODEL_PATH)
model.to(device)

# print("Loading EasyOCR...")
# reader = easyocr.Reader(['en'], gpu=USE_GPU)

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
    gray = cv2.resize(gray, (int(target_height * aspect), target_height),
                      interpolation=cv2.INTER_CUBIC)
    
    std = gray.std()
    if std < 40:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        gray = clahe.apply(gray)
    
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
    return cv2.filter2D(gray, -1, kernel)


# ===================== SMART CLEANING =====================
def smart_clean_plate_text(text):
    if not text:
        return ""
    
    text = text.upper().strip()
    text = "".join(c for c in text if c.isalnum())
    
    if len(text) < 4:
        return ""
    
    # Only correct at boundaries where format is unambiguous:
    # first 2 chars must be letters — fix obvious digit-looks-like-letter
    result = list(text)
    digit_to_letter = {'0':'O', '1':'I', '5':'S', '8':'B'}
    letter_to_digit = {'O':'0', 'I':'1', 'S':'5', 'B':'8', 'Z':'2', 'G':'6'}

    for i in range(min(2, len(result))):
        if result[i].isdigit():
            result[i] = digit_to_letter.get(result[i], result[i])

    # chars 2-4 must be digits — fix obvious letter-looks-like-digit
    for i in range(2, min(4, len(result))):
        if result[i].isalpha():
            result[i] = letter_to_digit.get(result[i], result[i])

    # last 4 must be digits
    for i in range(max(4, len(result)-4), len(result)):
        if result[i].isalpha():
            result[i] = letter_to_digit.get(result[i], result[i])

    return "".join(result)

# ===================== DUAL OCR FUNCTION =====================
def get_best_ocr(crop):
    crop_hash = hashlib.md5(crop.tobytes()).hexdigest()
    if crop_hash in _ocr_cache:
        return _ocr_cache[crop_hash]

    processed = safe_preprocess(crop)
    if processed is None:
        return "", 0.0

    tess_config = r'--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    data = pytesseract.image_to_data(processed, config=tess_config,
                                     output_type=pytesseract.Output.DICT)
    
    words = [w for w, c in zip(data['text'], data['conf']) 
             if w.strip() and int(c) > 0]
    confs = [int(c) for c in data['conf'] if int(c) > 0]

    text = "".join(words)
    conf = sum(confs) / len(confs) / 100 if confs else 0.0

    text = smart_clean_plate_text(text)
    result = (text, conf)
    _ocr_cache[crop_hash] = result
    return result

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
        # Load existing index
        all_entries = []
        processed_images = set()
        if os.path.exists(JSON_FILE):
            with open(JSON_FILE, 'r') as f:
                all_entries = json.load(f)
            processed_images = {e["image_name"] for e in all_entries}

        to_process = [f for f in image_files if f not in processed_images]
        print(f"{len(to_process)} new images to process (skipping {len(processed_images)} already done)\n")

        if to_process:
            all_paths = [os.path.join(INPUT_FOLDER, f) for f in to_process]

            print("Running YOLO batch detection...")
            batch_results = model(all_paths, conf=CONFIDENCE, iou=IOU,
                                  verbose=False, batch=8)

            for img_name, result in tqdm(zip(to_process, batch_results),
                                         desc="Running OCR", total=len(to_process)):
                frame = cv2.imread(os.path.join(INPUT_FOLDER, img_name))
                if frame is None:
                    continue

                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    plate_conf = float(box.conf[0])
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    text, ocr_conf = get_best_ocr(crop)

                    if text and len(text) >= 8:
                        entry = {
                            "timestamp": datetime.datetime.now().isoformat(),
                            "image_name": img_name,
                            "plate_text": text,
                            "ocr_confidence": float(ocr_conf),
                            "plate_confidence": plate_conf,
                            "bbox": (x1, y1, x2, y2)
                        }
                        all_entries.append(entry)
                        print(f"  {img_name} → {text} (OCR: {ocr_conf:.2f})")

                    if SAVE_ANNOTATED:
                        annotated = draw_results(frame.copy(),
                                                 [{"bbox": (x1, y1, x2, y2),
                                                   "text": text,
                                                   "ocr_conf": ocr_conf}])
                        cv2.imwrite(os.path.join(ANNOTATED_FOLDER,
                                                 f"annotated_{img_name}"), annotated)

            # Single write at the end
            with open(JSON_FILE, 'w') as f:
                json.dump(all_entries, f, indent=2)
            print(f"\nDone. {len(all_entries)} total entries saved.")
