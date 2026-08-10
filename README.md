# TrafficVision-KG

Experimental computer-vision pipeline for traffic video analytics.

The project was built step by step as a practical preparation project for a Computer Vision Engineer role: from basic vehicle detection to multi-object tracking, directional counting, ANPR experiments, best-shot selection and experimental road geometry.

## Pipeline

```text
Input video
    ↓
YOLO11 vehicle detection
    ↓
ByteTrack multi-object tracking
    ↓
Track ID + trajectory
    ↓
Directional line counting
    ↓
Vehicle crop
    ↓
License plate YOLO detector
    ↓
Best-shot selection per track
    ↓
EasyOCR / PaddleOCR experiments
    ↓
Experimental homography
    ↓
Rough speed estimation
```

## Project evolution

### Day 1 — Vehicle Detection

Implemented baseline vehicle detection with YOLO11.

Detected COCO vehicle classes:

* car;
* motorcycle;
* bus;
* truck.

Experiments were performed with different confidence thresholds.

CPU baseline performance:

```text
Resolution: 1280x720
Source FPS: 30
Pipeline FPS: ~21–22
```

### Day 2 — Detection Analysis

Extended the baseline detector with:

* confidence threshold experiments;
* NMS IoU experiments;
* class statistics;
* terminal metrics;
* analysis of precision, recall, AP and mAP.

A separate script was preserved to keep the project evolution visible.

### Day 3 — Multi-Object Tracking

Added ByteTrack using Ultralytics YOLO tracking.

Features:

* persistent `track_id`;
* vehicle trajectories;
* active-track statistics;
* unique track-ID diagnostics.

Important limitation:

`Unique vehicle IDs` is not equal to the real number of vehicles because track fragmentation and ID switches can occur.

### Day 4 — Directional Vehicle Counting

Added line-crossing analytics.

Each tracked vehicle uses its bottom-center point.

The pipeline counts:

* UP;
* DOWN;
* TOTAL.

Each `track_id` is counted only once per direction.

### Day 5 — ANPR / OCR

Added a second YOLO model for license-plate detection.

Pipeline:

```text
Tracked vehicle
    ↓
Vehicle crop
    ↓
Plate detector
    ↓
Plate crop
    ↓
OCR
```

EasyOCR was integrated into the main environment.

PaddleOCR was tested separately in a Python 3.12 environment because the main project environment uses a newer Python version.

OCR experiments showed that recognition quality is strongly limited by the source CCTV footage:

* small distant plates;
* motion blur;
* compression;
* occlusion;
* camera motion;
* false plate detections on bus route displays.

OCR confidence is therefore not treated as recognition accuracy.

### Day 6 — Best-Shot and Geometry

Added per-track best-shot selection before OCR.

Candidate quality is estimated using:

* plate crop size;
* Variance of Laplacian sharpness;
* license-plate detector confidence.

Instead of sending every plate observation to OCR, the pipeline keeps only TOP-K candidates per track.

Day 6 results:

```text
Plate observations:       848
Candidates entering TOP-K: 106
Final best-shot files:      76
```

Compared with the earlier OCR experiment, the amount of OCR input was reduced substantially.

Also implemented:

* interactive four-point homography calibration;
* pixel-to-world transformation;
* rough speed estimation;
* experimental red-light logic;
* experimental wrong-way logic.

Speed estimation is currently experimental because the calibration dimensions are demonstrational and the source camera is not perfectly static.

## Day 7 — Integrated Demo

Day 7 combines the working stages into one demonstration pipeline:

```text
src/demo_pipeline.py
```

Current demo stages:

* YOLO11 vehicle detection;
* ByteTrack tracking;
* persistent track IDs;
* vehicle trajectories;
* directional line crossing;
* vehicle counting;
* license-plate detection;
* per-track best-shot selection;
* EasyOCR experiments;
* temporal OCR aggregation;
* experimental homography;
* rough speed estimation;
* annotated output video.

### Current Day 7 result

Input:

```text
Resolution: 1280x720
Source FPS: 30.00
Frames: 939
```

Traffic analytics:

```text
Unique track IDs: 115

UP:    20
DOWN:   6
TOTAL: 26
```

ANPR pipeline:

```text
Plate candidates seen:       848
Candidates entering TOP-K:   106
Final best shots:             76
EasyOCR calls:               106
Validated aggregated plates:   0
```

Performance on CPU:

```text
Processing time: 313.24 sec
Pipeline FPS: 3.00
```

The full pipeline is significantly slower than the Day 1 detector because it additionally performs tracking, plate detection, best-shot analysis, OCR, geometry and visualization.

This CPU result is used as the baseline for further inference optimization.

## Run

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Run the integrated demo:

```powershell
python src\demo_pipeline.py `
    --source data\input\intersection.mp4 `
    --output outputs\day7_demo.mp4
```

Default best-shot output:

```text
outputs/day7_bestshots/
```

## Models

Vehicle detector:

```text
yolo11n.pt
```

License-plate detector:

```text
models/license_plate_detector.pt
```

Model weights are not stored in Git.

Place the license-plate detector manually at:

```text
models/license_plate_detector.pt
```

## Main technologies

* Python
* OpenCV
* Ultralytics YOLO11
* ByteTrack
* PyTorch
* NumPy
* EasyOCR
* PaddleOCR experiments
* Homography / perspective transformation

## Current limitations

The current system is an engineering prototype, not a production traffic-enforcement system.

Known limitations:

* CPU-only inference is slow;
* full pipeline runs at approximately 3 FPS;
* distant vehicles may lose tracking IDs;
* occlusion can cause track fragmentation;
* unique track IDs are not a true vehicle count;
* distant license plates contain too few useful pixels;
* motion blur reduces OCR quality;
* EasyOCR currently produces no reliably validated plates on this video;
* bus LED route displays can be detected as license plates;
* OCR confidence is not equivalent to accuracy;
* no complete manually labelled ANPR ground-truth dataset exists yet;
* camera motion violates the ideal static-homography assumption;
* current metric road dimensions are demonstrational;
* speed values must therefore be treated as rough estimates;
* red-light and wrong-way modules are experimental prototypes.

## Next steps

Planned engineering improvements:

1. TensorRT FP16 optimization.
2. TensorRT INT8 calibration.
3. ANPR-specific ROI.
4. Class-specific plate filtering.
5. Manual plate ground-truth dataset.
6. Character-level and plate-level OCR metrics.
7. Camera stabilization.
8. Real metric road calibration.
9. Validated speed measurement.
10. DeepStream pipeline.
11. Multi-stream processing.
12. Kafka/backend integration.

## Repository structure

```text
trafficvision-kg/
│
├── configs/
│   └── homography.json
│
├── data/
│   └── input/
│       └── intersection.mp4
│
├── models/
│   └── license_plate_detector.pt
│
├── src/
│   ├── detect_vehicles.py
│   ├── detect_vehicles2.py
│   ├── track_vehicles.py
│   ├── count_vehicles.py
│   ├── anpr_vehicles.py
│   ├── anpr_bestshot.py
│   ├── demo_pipeline.py
│   ├── test_paddleocr.py
│   └── test_paddle_recognition.py
│
└── README.md
```

Generated videos, model weights, Python environments and runtime outputs are excluded from Git.
