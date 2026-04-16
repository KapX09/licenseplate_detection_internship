# License Plate Detection
Modeling license plate detector for live detection in camera for vehicles using Yolo, CNN and fetching License Number Characters using OCR in tensor flow.
It works as Automatic License Plate Recognition (ALPR) pipeline with detection, OCR text recognition, and fast GPU-based correction/prediction module.

***Structure used in LPD***
```
├── models/
├── output/
├── test_images/
├── live_plate_detector.py
└── requirements.txt
```

***REQUIREMENTS***
Python 3.8+ | OpenCV | TensorFlow | NumPy | OCR engine (Tesseract, easyocr paddleocr/paddlepaddle)
torch | torchvision | torchaudio | tqdm | opencv-python | easyocr | PyYAML | numpy | pytesseract

***Install:***
```
pip install -r requirements.txt
```

***Run:***
```
python live_plate_detector.py
```

***Pipeline:*** 
Frame Capture → Plate Detection → ROI Extraction → OCR → Output Visualization
Place trained model weights inside `models/`.

### Prototype1: Testing models with the images of Licensed plates of vehicles with Trained model and OCR ( ***Optical Character Recognition***) engine.
**_Overview_**
Pipeline for Indian license plate recognition using a custom YOLOv8 detector (`best.pt`) followed by OCR-based character extraction. Input: static images. Output: recognized plate string.

---
### Pipeline
1. Plate Detection → YOLOv8 (`best.pt`)
2. Region Cropping → bounding box extraction
3. OCR → EasyOCR (primary)
4. Post-processing → rule-based correction
---

#### Development & Experiments
_Phase 1: Baseline_
- YOLOv8 detection + EasyOCR (no preprocessing) | Accuracy: ~15–20% | Issues: low confidence, incorrect characters

_Phase 2: Preprocessing_
- Techniques: CLAHE, sharpening, bilateral filtering, deskew
- Observation: Minor gains on some samples | Degradation on others due to artifacts
- Decision: removed heavy preprocessing (kept minimal ops only)

_Phase 3: OCR Evaluation_
- EasyOCR (grayscale + resize + mild sharpening) → **most stable**
- Tesseract: Frequent prefix noise | Lower consistency | inconsistent parsing
- PaddleOCR:| Config issues (`det=False`, shape tuning) | Inconsistent outputs

_Ensemble + Optimization_ : Dual OCR (EasyOCR + Tesseract) with voting | Added lightweight GPU optimizations | Result: improved usable accuracy

_Phase 4: Post-processing_
- Rule-based corrections: Character mapping: O→0, I→1, Z→2, B→8 | Position-aware fixes | Length validation + noise filtering
- Result: significant improvement in usable outputs

_(Current)_ : Tesseract only + minimal preprocessing | Focus: cleaner pipeline with comparable performance  

---
_**Performance Summary**_

| Trial | Approach                          | Exact Match | Usable Accuracy | Major Observation                  |
|------|----------------------------------|------------|-----------------|----------------------------------|
| 1    | Basic EasyOCR                    | ~15%       | ~25%            | Very low confidence              |
| 2    | EasyOCR + Heavy Preprocessing    | ~20%       | ~40%            | Artifacts introduced             |
| 3    | EasyOCR + Tesseract (Dual)       | 22%        | 55%             | Better coverage                  |
| 4    | PaddleOCR                        | 25%        | 45%             | Inconsistent                     |
| 5    | EasyOCR + Smart Rules            | 33%        | 67%             | Good but slightly overfitted     |
| 6    | Dual OCR + GPU Optimization      | 33%        | 75%             | Best usable accuracy             |
| 7    | Tesseract + Minimal Prep         | 33%        | 66.7%           | Simpler and stable pipeline      |
---

_**Current Limitations**_
- Performance drops on: angled plates | low contrast  | large bounding regions | No temporal modeling (single-frame only)
- Common misclassifications:
  - 1 / 4 / 7
  - B / 8
  - 2 / 7
  - Extra prefix noise (E, 1)
---
