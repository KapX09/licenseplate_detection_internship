import cv2
import time
import json
import datetime
import os
import torch
from ultralytics import YOLO
from paddleocr import PaddleOCR
from pathlib import Path
from tqdm import tqdm
import logging

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

# Suppress PaddleOCR logs (since show_log is removed)
logging.getLogger("ppocr").setLevel(logging.ERROR)

# ===================== LOAD MODELS =====================
print("Loading models...")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(MODEL_PATH)
model.to(device)

# PaddleOCR - optimized for Indian plates
# print("Loading PaddleOCR...")
# ocr = PaddleOCR(use_textline_orientation=True, lang='en', det=True, rec=True)   # lang='en' works well for Indian plates

# PaddleOCR - Fixed for PaddleOCR 3.x
print("Loading PaddleOCR...")
ocr = PaddleOCR(
    use_textline_orientation=True,   # ← Replaced use_angle_cls
    lang='en',                       # Good for Indian plates
    # show_log=False → REMOVED (causes error in 3.x)
    det=True,                        # Explicitly enable detection + recognition
    rec=True
)

print("Models loaded successfully!\n")

# ===================== HELPER FUNCTIONS =====================
def clean_plate_text(text):
    """Light cleaning for Indian plates"""
    if not text:
        return ""
    text = text.upper().strip()
    text = text.replace(" ", "").replace("~", "").replace("|", "").replace("-", "").replace(".", "")
    return text

def process_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not read {image_path}")
        return None, None, 0
    
    start = time.time()
    
    # 1. Detect plate with YOLO
    results = model(frame, conf=CONFIDENCE, iou=IOU, verbose=False)
    
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_conf = float(box.conf[0])
            
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            
            # 2. PaddleOCR on cropped plate
            ocr_result = ocr.ocr(crop, cls=True)
            
            text = ""
            ocr_conf = 0.0
            if ocr_result and ocr_result[0]:
                for line in ocr_result[0]:
                    text_part = line[1][0]      # recognized text
                    conf_part = line[1][1]      # confidence
                    if conf_part > ocr_conf:
                        text = text_part
                        ocr_conf = conf_part
            
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
        print(f"Found {len(image_files)} images. Testing with PaddleOCR...\n")
        
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
            
            # Show and save annotated image
            annotated = draw_results(frame.copy(), detections)
            if SAVE_ANNOTATED:
                cv2.imwrite(os.path.join(ANNOTATED_FOLDER, f"annotated_{img_name}"), annotated)
            
            cv2.imshow(f"Result - {img_name}", annotated)
            cv2.waitKey(0)          # Press any key to go to next image
            cv2.destroyAllWindows()
        
        print("\nProcessing finished! Check output/captured_plates.json")
