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
    
    # Basic cleaning
    text = "".join(c for c in text if c.isalnum())
    
    if len(text) < 6:
        return text
    
    # Convert to list for position-based editing
    chars = list(text)
    
    # Rule 1: State code (first 2 chars) should be letters
    for i in range(min(2, len(chars))):
        if chars[i].isdigit():
            chars[i] = {'0':'O', '1':'I', '8':'B', '5':'S', '2':'Z'}.get(chars[i], chars[i])
    
    # Rule 2: District code (positions 2-4) should be digits
    for i in range(2, min(4, len(chars))):
        if chars[i].isalpha():
            chars[i] = {'O':'0', 'I':'1', 'B':'8', 'S':'5', 'Z':'2'}.get(chars[i], chars[i])
    
    # Rule 3: Last 4 characters should preferably be digits
    for i in range(max(0, len(chars)-4), len(chars)):
        if chars[i].isalpha():
            chars[i] = {'O':'0', 'I':'1', 'B':'8', 'S':'5', 'Z':'2'}.get(chars[i], chars[i])
    
    # Rule 4: Fix common middle confusions (positions 4 to -4)
    for i in range(4, max(4, len(chars)-4)):
        if chars[i] == 'B':
            chars[i] = '8'
        elif chars[i] == 'S':
            chars[i] = '5'
        elif chars[i] == '4' and i < len(chars)-2:   # avoid changing near end
            chars[i] = '1'
    
    # Rule 5: Remove obvious wrong prefix (only if very long)
    cleaned = "".join(chars)
    if len(cleaned) > 11 and cleaned[0] in "E1IA":
        cleaned = cleaned[1:]
    
    # Final safety: limit length (Indian plates are usually 9-10 chars)
    return cleaned[:10]

# ===================== DUAL OCR FUNCTION =====================
def get_best_ocr(crop):
    #  Hash the crop pixels — same plate crop = same hash = skip reprocessing
    crop_hash = hashlib.md5(crop.tobytes()).hexdigest()
    if crop_hash in _ocr_cache:
        return _ocr_cache[crop_hash]
    
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

    _ocr_cache[crop_hash] = result  # store before returning
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
