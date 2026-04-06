import cv2
import time
import json
import datetime
import os
import torch
import easyocr
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm  # progress bar

# ===================== CONFIG =====================
MODEL_PATH = "models/best.pt"
CONFIDENCE = 0.5
IOU = 0.45
INPUT_FOLDER = "test_images"
JSON_FILE = "output/captured_plates.json"
SAVE_ANNOTATED = True                  # Set True to save images with boxes
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
def save_to_json(plate_text, ocr_conf, plate_conf, image_name, bbox):
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "image_name": image_name,
        "plate_text": plate_text,
        "ocr_confidence": float(ocr_conf),
        "plate_confidence": float(plate_conf),
        "bbox": bbox
    }
    
    data = []
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'r') as f:
            data = json.load(f)
    
    data.append(entry)
    
    with open(JSON_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"SAVED → {image_name} | {plate_text}  (OCR: {ocr_conf:.2f})")

def process_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not read {image_path}")
        return None, None
    
    start = time.time()
    
    # Plate Detection
    results = model(frame, conf=CONFIDENCE, iou=IOU, verbose=False)
    
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_conf = float(box.conf[0])
            
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            
            # OCR on cropped plate
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            ocr_results = reader.readtext(gray, detail=1)
            
            text = ""
            ocr_conf = 0.0
            if ocr_results:
                text = ocr_results[0][1].replace(" ", "").upper()
                ocr_conf = ocr_results[0][2]
            
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
        print("Please put your test images there and run again.")
    else:
        print(f"Found {len(image_files)} images. Starting processing...\n")
        
        for img_name in tqdm(image_files, desc="Processing images"):
            img_path = os.path.join(INPUT_FOLDER, img_name)
            
            frame, detections, fps = process_image(img_path)
            if frame is None:
                continue
            
            print(f"{img_name} → FPS: {fps:.1f}")
            
            # Save detections to JSON
            for det in detections:
                if det['text'] and len(det['text']) >= 4:
                    save_to_json(det['text'], det['ocr_conf'], 
                                det['plate_conf'], img_name, det['bbox'])
            
            # Draw and save/show annotated image
            annotated = draw_results(frame.copy(), detections)
            
            if SAVE_ANNOTATED:
                save_path = os.path.join(ANNOTATED_FOLDER, f"annotated_{img_name}")
                cv2.imwrite(save_path, annotated)
            
            # Show the result (press any key to go to next image)
            cv2.imshow(f"Result: {img_name}", annotated)
            cv2.waitKey(0)   # Change to 1 for auto (very fast)
            cv2.destroyAllWindows()
        
        print("\nAll images processed!")
        print(f"Results saved in: {JSON_FILE}")
        if SAVE_ANNOTATED:
            print(f"Annotated images saved in: {ANNOTATED_FOLDER}")