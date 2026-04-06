# License Plate Detection
Modeling license plate detector for live detection in camera for vehicles using Yolo, CNN and fetching License Number Characters using OCR in tensor flow.
It works as Automatic License Plate Recognition (ALPR) pipeline with detection, OCR text recognition, and fast GPU-based correction/prediction module.

### Prototype1: Testing models with the images of Licensed plates of vehicles with Trained model and OCR ( _Optical Character Recognition_) engine.

_**Structure used in LPD**_
.
├── models/
├── output/
├── test_images/
├── live_plate_detector.py
└── requirements.txt

_**REQUIREMENTS**_
Python 3.8+ | OpenCV | TensorFlow | NumPy | OCR engine (Tesseract)

_**Install:**_
pip install -r requirements.txt

_**Run:**_
python live_plate_detector.py

_**Pipeline: **_
Frame Capture → Plate Detection → ROI Extraction → OCR → Output Visualization
Place trained model weights inside models/.
