# RapidClaim

Machine Learning System for Pedestrian and Traffic Light Detection with Driver Fault Classification

## Overview

RapidClaim is an end-to-end machine learning system that:
1. Detects pedestrians, vehicles, and traffic lights in video footage
2. Tracks objects across frames
3. Classifies traffic light states (red, yellow, green)
4. Extracts features related to traffic violations
5. Classifies drivers as "at fault" or "not at fault" based on detected violations

## Features

- **Object Detection**: YOLOv8-based detection for pedestrians, vehicles, and traffic lights
- **Traffic Light Classification**: Color-based and neural network-based state classification with temporal smoothing
- **Multi-Object Tracking**: Kalman filter-based tracking with Hungarian algorithm for association
- **Feature Engineering**: Automatic extraction of violation features (red light violations, pedestrian crossing violations, collision proximity, etc.)
- **Fault Classification**: XGBoost/Random Forest classifier for determining driver fault
- **Video Processing**: End-to-end pipeline for processing video files
- **Interactive Annotation**: Tool for annotating videos with fault labels

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for faster inference)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd rapidclaim
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download YOLOv8 weights (automatically downloaded on first use):
```bash
# Weights are downloaded automatically when using YOLOv8
```

## Project Structure

```
rapidclaim/
├── src/
│   ├── detection/          # Object detection modules
│   │   ├── object_detector.py
│   │   └── traffic_light_classifier.py
│   ├── tracking/           # Object tracking modules
│   │   └── tracker.py
│   ├── features/           # Feature extraction
│   │   └── feature_extractor.py
│   ├── classification/     # Fault classification
│   │   └── fault_classifier.py
│   ├── pipeline/           # Main pipeline
│   │   └── video_processor.py
│   └── utils/              # Utility functions
│       └── config_loader.py
├── scripts/                # Utility scripts
│   ├── process_video.py
│   ├── train_fault_classifier.py
│   ├── prepare_training_data.py
│   └── annotate_video.py
├── config.yaml             # Configuration file
├── requirements.txt        # Python dependencies
└── README.md
```

## Configuration

Edit `config.yaml` to customize the system:

- **Detection**: Model selection, confidence thresholds, class filters
- **Traffic Light**: Classification parameters, temporal smoothing
- **Tracking**: Maximum age, minimum hits, IoU threshold
- **Features**: Proximity thresholds, violation detection parameters
- **Fault Classification**: Model type, feature selection
- **Video Processing**: FPS, frame skip, resolution

## Usage

### 1. Process a Video

Process a video and detect faults:

```bash
python scripts/process_video.py --video data/videos/input.mp4 --output outputs/output.mp4
```

Options:
- `--video`: Path to input video
- `--output`: Path to save output video (optional)
- `--config`: Path to config file (default: config.yaml)
- `--no-visualize`: Disable visualization
- `--save-results`: Path to save results JSON

### 2. Annotate Videos for Training

Interactively annotate videos with fault labels:

```bash
python scripts/annotate_video.py --video data/videos/input.mp4 --output data/annotations/annotations.json
```

Controls:
- `SPACE`: Play/Pause
- `a`: Mark current frame as AT FAULT
- `n`: Mark current frame as NOT AT FAULT
- `LEFT/RIGHT`: Navigate frames
- `s`: Save annotations
- `q`: Quit

### 3. Prepare Training Data

Extract features from annotated videos:

```bash
python scripts/prepare_training_data.py \
    --video-dir data/videos \
    --annotations data/annotations/annotations.json \
    --output data/training_data.json
```

### 4. Train Fault Classifier

Train the fault classification model:

```bash
python scripts/train_fault_classifier.py \
    --data data/training_data.json \
    --output models/fault_classifier.pkl
```

Options:
- `--data`: Path to training data (JSON or CSV)
- `--output`: Path to save trained model
- `--config`: Path to config file
- `--test-size`: Test set size (default: 0.2)

## Workflow

### Phase 1: Data Preparation

1. Collect video datasets with traffic scenarios
2. Annotate videos using `annotate_video.py`
3. Prepare training data using `prepare_training_data.py`

### Phase 2: Model Training

1. Train fault classifier using `train_fault_classifier.py`
2. Evaluate model performance
3. Fine-tune hyperparameters in `config.yaml`

### Phase 3: Inference

1. Process videos using `process_video.py`
2. Review results and predictions
3. Export results for further analysis

## Features Extracted

The system extracts the following features for fault classification:

1. **Red Light Violation**: Whether vehicle crossed intersection on red light
2. **Pedestrian Crossing Violation**: Whether vehicle failed to yield to pedestrian
3. **Collision Proximity**: Minimum distance to pedestrians
4. **Vehicle Speed**: Estimated speed in pixels per frame
5. **Time to Collision**: Estimated time until collision with pedestrian
6. **Intersection Zone**: Whether vehicle is in intersection zone
7. **Traffic Light State Duration**: Duration of current traffic light state

## Model Architecture

### Object Detection
- **Model**: YOLOv8 (nano, small, medium, large, or xlarge)
- **Classes**: Pedestrians, Vehicles (car, motorcycle, bus, truck), Traffic Lights
- **Output**: Bounding boxes, class IDs, confidence scores

### Traffic Light Classification
- **Method 1**: Color-based detection (HSV color space)
- **Method 2**: Neural network classifier (CNN)
- **Temporal Smoothing**: Window-based state smoothing to reduce flicker

### Object Tracking
- **Algorithm**: Kalman Filter + Hungarian Algorithm
- **Features**: Multi-object tracking with velocity estimation
- **Tracking**: Maintains track IDs across frames

### Fault Classification
- **Models**: XGBoost or Random Forest
- **Input**: Extracted features from tracking and detection
- **Output**: Binary classification (at fault / not at fault) with probability

## Performance Considerations

- **GPU Acceleration**: YOLOv8 uses GPU automatically if available
- **Frame Skipping**: Process every Nth frame to speed up processing
- **Batch Processing**: Process multiple videos in batch
- **Model Size**: Use smaller YOLOv8 models (nano, small) for faster inference

## Limitations

- Traffic light detection depends on camera angle and lighting conditions
- Intersection zone detection requires manual configuration or additional detection
- Speed estimation is in pixels per frame (not real-world units)
- Fault classification requires training data with expert annotations

## Future Improvements

- [ ] Real-world speed estimation using camera calibration
- [ ] Automatic intersection zone detection
- [ ] Support for multiple camera angles
- [ ] Real-time processing capabilities
- [ ] Integration with insurance claim systems
- [ ] Web interface for annotation and analysis
- [ ] Support for additional violation types
- [ ] Improved traffic light state classification
- [ ] Pedestrian behavior prediction

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details

## Acknowledgments

- YOLOv8 by Ultralytics
- OpenCV for computer vision utilities
- XGBoost for gradient boosting
- Scikit-learn for machine learning utilities

## Contact

For questions or issues, please open an issue on GitHub.
