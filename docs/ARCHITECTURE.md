# System Architecture

This document describes the architecture of the RapidClaim system.

## Overview

RapidClaim is a modular system consisting of several interconnected components:

1. **Object Detection**: Detects pedestrians, vehicles, and traffic lights
2. **Traffic Light Classification**: Classifies traffic light states
3. **Object Tracking**: Tracks objects across frames
4. **Feature Extraction**: Extracts features for fault classification
5. **Fault Classification**: Classifies drivers as at fault or not at fault

## Component Architecture

### 1. Object Detection Module

**Location**: `src/detection/object_detector.py`

**Purpose**: Detect objects in video frames using YOLOv8

**Components**:
- `ObjectDetector`: Main detection class
  - Uses YOLOv8 for object detection
  - Filters detections by class and confidence
  - Returns bounding boxes and class information

**Input**: Video frame (numpy array)
**Output**: List of detections with bounding boxes, class IDs, and confidence scores

### 2. Traffic Light Classification Module

**Location**: `src/detection/traffic_light_classifier.py`

**Purpose**: Classify traffic light states (red, yellow, green)

**Components**:
- `TrafficLightStateClassifier`: CNN model for state classification
- `TrafficLightClassifier`: Main classifier with temporal smoothing
  - Color-based detection (HSV color space)
  - Neural network-based classification
  - Temporal smoothing to reduce flicker

**Input**: Video frame and traffic light detection
**Output**: Traffic light state and confidence

### 3. Object Tracking Module

**Location**: `src/tracking/tracker.py`

**Purpose**: Track objects across frames

**Components**:
- `KalmanTracker`: Kalman filter for individual object tracking
- `MultiObjectTracker`: Multi-object tracker using Hungarian algorithm
  - Associates detections to tracks
  - Maintains track history
  - Estimates velocity

**Input**: List of detections from current frame
**Output**: List of tracked objects with track IDs

### 4. Feature Extraction Module

**Location**: `src/features/feature_extractor.py`

**Purpose**: Extract features for fault classification

**Components**:
- `FeatureExtractor`: Main feature extraction class
  - Red light violation detection
  - Pedestrian crossing violation detection
  - Collision proximity calculation
  - Speed estimation
  - Time to collision estimation

**Features Extracted**:
1. Red light violation (binary)
2. Pedestrian crossing violation (binary)
3. Collision proximity (float)
4. Vehicle speed (float)
5. Time to collision (float)
6. Intersection zone (binary)
7. Traffic light state duration (int)

**Input**: Tracked objects, frame information
**Output**: Feature dictionary for each vehicle

### 5. Fault Classification Module

**Location**: `src/classification/fault_classifier.py`

**Purpose**: Classify drivers as at fault or not at fault

**Components**:
- `FaultClassifier`: Main classification class
  - XGBoost or Random Forest classifier
  - Feature importance analysis
  - Model training and inference

**Input**: Feature dictionary
**Output**: Fault prediction (binary) and probability

### 6. Video Processing Pipeline

**Location**: `src/pipeline/video_processor.py`

**Purpose**: End-to-end video processing

**Components**:
- `VideoProcessor`: Main pipeline class
  - Coordinates all components
  - Processes video frames
  - Visualizes results
  - Generates summaries

**Workflow**:
1. Load video
2. For each frame:
   - Detect objects
   - Classify traffic lights
   - Track objects
   - Extract features
   - Classify fault
   - Visualize results
3. Generate summary

## Data Flow

```
Video Frame
    ↓
Object Detection (YOLOv8)
    ↓
Traffic Light Classification
    ↓
Object Tracking (Kalman Filter + Hungarian)
    ↓
Feature Extraction
    ↓
Fault Classification (XGBoost/Random Forest)
    ↓
Results (Visualization + Summary)
```

## Configuration

**Location**: `config.yaml`

The system is configured through a YAML file that allows customization of:
- Detection parameters (model, confidence thresholds)
- Tracking parameters (max age, min hits)
- Feature extraction parameters (thresholds, windows)
- Classification parameters (model type, features)
- Video processing parameters (FPS, frame skip)

## Extension Points

The system is designed to be extensible:

1. **Custom Detectors**: Replace YOLOv8 with other detection models
2. **Custom Trackers**: Replace Kalman filter with other tracking algorithms
3. **Custom Features**: Add new features to the feature extractor
4. **Custom Classifiers**: Replace XGBoost with other classification models
5. **Custom Visualizations**: Extend visualization in video processor

## Performance Considerations

### Optimization Strategies

1. **GPU Acceleration**: YOLOv8 uses GPU automatically
2. **Frame Skipping**: Process every Nth frame
3. **Model Size**: Use smaller models for faster inference
4. **Batch Processing**: Process multiple videos in parallel

### Scalability

- **Single Video**: Processes one video at a time
- **Batch Processing**: Can process multiple videos sequentially
- **Real-time**: Not currently optimized for real-time processing

## Dependencies

### Core Dependencies

- **PyTorch**: Deep learning framework
- **Ultralytics YOLOv8**: Object detection
- **OpenCV**: Computer vision utilities
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning utilities
- **XGBoost**: Gradient boosting

### Optional Dependencies

- **CUDA**: GPU acceleration
- **FFmpeg**: Video processing

## Testing

### Unit Tests

Each module should have unit tests:
- Object detection tests
- Tracking tests
- Feature extraction tests
- Classification tests

### Integration Tests

End-to-end tests for the pipeline:
- Video processing tests
- Annotation tests
- Training tests

## Deployment

### Requirements

- Python 3.8+
- CUDA (optional, for GPU acceleration)
- Sufficient memory for video processing
- Storage for models and data

### Deployment Options

1. **Local**: Run on local machine
2. **Cloud**: Deploy on cloud platforms (AWS, GCP, Azure)
3. **Edge**: Deploy on edge devices (with model optimization)

## Future Improvements

1. **Real-time Processing**: Optimize for real-time inference
2. **Multi-camera Support**: Process multiple camera feeds
3. **Web Interface**: Add web-based UI for annotation and analysis
4. **API**: Provide REST API for integration
5. **Model Serving**: Deploy models as microservices


