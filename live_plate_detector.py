import cv2
import time
import json
import datetime
import os
import torch
from ultralytics import YOLO
import easyocr
import pytesseract
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

# Create folders
Path("output").mkdir(exist_ok=True)
Path(ANNOTATED_FOLDER).mkdir(exist_ok=True)

# ===================== LOAD MODELS =====================
print("Loading models...")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(MODEL_PATH)
model.to(device)

# EasyOCR
easy_reader = easyocr.Reader(['en'], gpu=True)

# Tesseract OCR (make sure tesseract is installed on your system)
print("Models loaded!\n")

# ===================== HELPER FUNCTIONS =====================
def clean_plate_text(text):
    """Very light cleaning"""
    if not text:
        return ""
    text = text.upper().strip()
    text = text.replace(" ", "").replace("~", "").replace("|", "").replace("-", "").replace(".", "")
    return text

def get_best_ocr(crop):
    """Run both EasyOCR and Tesseract and choose the better one"""
    if crop is None or crop.size == 0:
        return "", 0.0
    
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # EasyOCR
    easy_results = easy_reader.readtext(gray, detail=1)
    easy_text = easy_results[0][1] if easy_results else ""
    easy_conf = easy_results[0][2] if easy_results else 0.0
    
    # Tesseract OCR
    tess_config = '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    tess_text = pytesseract.image_to_string(gray, config=tess_config).strip()
    tess_conf = 0.6 if len(tess_text) > 6 else 0.3   # rough confidence estimation
    
    # Choose the better result
    if len(clean_plate_text(easy_text)) > len(clean_plate_text(tess_text)):
        return clean_plate_text(easy_text), easy_conf
    else:
        return clean_plate_text(tess_text), tess_conf

# ===================== MAIN PROCESSING =====================
def process_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        return None, None, 0
    
    start = time.time()
    
    # Detect plate using YOLO .pt
    results = model(frame, conf=CONFIDENCE, iou=IOU, verbose=False)
    
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_conf = float(box.conf[0])
            
            crop = frame[y1:y2, x1:x2]
            
            # Get best text using two OCRs
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

# ===================== RUN =====================
if __name__ == "__main__":
    image_files = [f for f in os.listdir(INPUT_FOLDER) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print("No images found in test_images folder!")
    else:
        print(f"Found {len(image_files)} images. Running with EasyOCR + Tesseract...\n")
        
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
                    
                    print(f"   → Saved: {det['text']} (Conf: {det['ocr_conf']:.2f})")
            
            annotated = draw_results(frame.copy(), detections)
            if SAVE_ANNOTATED:
                cv2.imwrite(os.path.join(ANNOTATED_FOLDER, f"annotated_{img_name}"), annotated)
            
            cv2.imshow(f"Result: {img_name}", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        print("\nProcessing completed! Check output/captured_plates.json")
