# License Plate Detection
Modeling license plate detector for live detection in camera for vehicles using Yolo, CNN and fetching License Number Characters using OCR in tensor flow.
It works as Automatic License Plate Recognition (ALPR) pipeline with detection, OCR text recognition, and fast GPU-based correction/prediction module.

### Prototype1: Testing models with the images of Licensed plates of vehicles with Trained model and OCR ( ***Optical Character Recognition***) engine.

***Structure used in LPD***
```
├── models/
├── output/
├── test_images/
├── live_plate_detector.py
└── requirements.txt
```

***REQUIREMENTS***
Python 3.8+ | OpenCV | TensorFlow | NumPy | OCR engine (Tesseract)

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
