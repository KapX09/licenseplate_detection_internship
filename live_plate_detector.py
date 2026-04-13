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
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ===================== CONFIG =====================
MODEL_PATH = "models/best.pt"
CONFIDENCE = 0.5
IOU = 0.45
INPUT_FOLDER = "test_images"
JSON_FILE = "output/captured_plates.json"
SAVE_ANNOTATED = True
ANNOTATED_FOLDER = "output/annotated"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# Create folders
Path("output").mkdir(exist_ok=True)
Path(ANNOTATED_FOLDER).mkdir(exist_ok=True)

# Suppress PaddleOCR logs
logging.getLogger("ppocr").setLevel(logging.ERROR)

# ===================== LOAD MODELS =====================
print("Loading models...")

device = 'cpu'                     # Force CPU
model = YOLO(MODEL_PATH)
model.to(device)

# Suppress logs
import logging
logging.getLogger("ppocr").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)

print("Loading PaddleOCR...")
ocr = PaddleOCR(
    lang='en',
    enable_mkldnn=False        # Fixes the oneDNN/PIR crash on CPU
)

print("Models loaded successfully!\n")

# ===================== HELPER FUNCTIONS =====================
def clean_plate_text(text):
    """Improved cleaning for Indian license plates"""
    if not text:
        return ""
    
    text = text.upper().strip()
    
    # Remove unwanted characters
    text = text.replace(" ", "").replace("~", "").replace("|", "").replace("-", "") \
               .replace(".", "").replace(":", "").replace(";", "").replace(",", "")
    
    # Common OCR mistakes on Indian plates
    text = text.replace("O", "0").replace("I", "1").replace("Z", "2") \
               .replace("S", "5").replace("B", "8").replace("G", "6")
    
    # Keep only alphanumeric (Indian plates are like MH20EE1234, DL7CA2345, etc.)
    text = "".join(c for c in text if c.isalnum())
    
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
            
            # 2. PaddleOCR on cropped plate (safe for 3.4.0)
            ocr_result = ocr.ocr(crop)
            
            text = ""
            ocr_conf = 0.0
            
            if ocr_result and ocr_result[0]:          # ocr_result[0] contains the list of lines
                for line in ocr_result[0]:
                    if not line or len(line) < 2:
                        continue
                    
                    item = line[1]                    # This can be str or [text, conf]
                    
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        text_part = item[0]
                        conf_part = float(item[1]) if isinstance(item[1], (int, float)) else 0.0
                    elif isinstance(item, str):
                        text_part = item
                        conf_part = 0.8                   # default confidence when not returned
                    else:
                        continue
                    
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
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        print("\nProcessing finished! Check output/captured_plates.json")
