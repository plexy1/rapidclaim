# Data Preparation Guide

This guide explains how to prepare data for training the RapidClaim fault classification system.

## Data Requirements

### Video Data

You need video files containing traffic scenarios with:
- Pedestrians
- Vehicles
- Traffic lights
- Various traffic situations (intersections, crosswalks, etc.)

**Supported formats**: MP4, AVI, MOV, MKV, FLV

### Annotation Data

You need to annotate videos with fault labels indicating whether the driver is "at fault" or "not at fault" in each frame or scenario.

## Annotation Format

Annotations should be in JSON format:

```json
{
  "video1": {
    "100": true,
    "101": true,
    "102": false,
    "vehicle_faults": {
      "1": true,
      "2": false
    }
  },
  "video2": {
    "50": false,
    "51": false
  }
}
```

Where:
- Keys are video names (without extension)
- Frame numbers (as strings) indicate fault status for that frame
- `vehicle_faults` maps vehicle track IDs to fault status

## Annotation Tools

### Option 1: Interactive Annotation Tool

Use the provided annotation script:

```bash
python scripts/annotate_video.py --video data/videos/video1.mp4 --output data/annotations.json
```

**Controls**:
- `SPACE`: Play/Pause
- `a`: Mark current frame as AT FAULT
- `n`: Mark current frame as NOT AT FAULT
- `LEFT/RIGHT`: Navigate frames
- `s`: Save annotations
- `q`: Quit

### Option 2: Manual Annotation

Create a JSON file manually with the format shown above.

### Option 3: External Annotation Tools

You can use external tools like:
- **LabelImg**: For bounding box annotations
- **CVAT**: For video annotation
- **VGG Image Annotator (VIA)**: For image/video annotation

Then convert the annotations to the required JSON format.

## Data Collection Guidelines

### 1. Video Quality

- **Resolution**: Minimum 720p, preferably 1080p or higher
- **Frame Rate**: 30 FPS or higher
- **Lighting**: Good lighting conditions for better detection
- **Camera Angle**: Front-facing or dashcam view

### 2. Scenario Diversity

Include diverse scenarios:
- Different weather conditions (sunny, rainy, cloudy)
- Different times of day (day, night, dawn, dusk)
- Different locations (urban, suburban, highway)
- Different violation types:
  - Red light violations
  - Pedestrian crossing violations
  - Right-of-way violations
  - Speeding violations

### 3. Balanced Dataset

Ensure balanced representation:
- Equal number of "at fault" and "not at fault" cases
- Various types of violations
- Different vehicle types
- Different pedestrian scenarios

## Data Preparation Workflow

### Step 1: Collect Videos

1. Gather video footage from various sources
2. Organize videos in `data/videos/` directory
3. Ensure videos are in supported formats

### Step 2: Annotate Videos

1. Use the annotation tool to label videos
2. Mark frames where driver is at fault
3. Save annotations to `data/annotations.json`

### Step 3: Prepare Training Data

1. Extract features from annotated videos:
```bash
python scripts/prepare_training_data.py \
    --video-dir data/videos \
    --annotations data/annotations.json \
    --output data/training_data.json
```

2. Verify the training data:
```python
import json
with open('data/training_data.json', 'r') as f:
    data = json.load(f)
    print(f"Total samples: {len(data['features'])}")
    print(f"At fault: {sum(data['labels'])}")
    print(f"Not at fault: {len(data['labels']) - sum(data['labels'])}")
```

### Step 4: Train Model

1. Train the fault classifier:
```bash
python scripts/train_fault_classifier.py \
    --data data/training_data.json \
    --output models/fault_classifier.pkl
```

2. Evaluate model performance
3. Fine-tune if necessary

## Dataset Recommendations

### Minimum Dataset Size

- **Training**: At least 1000 labeled samples
- **Validation**: At least 200 labeled samples
- **Testing**: At least 200 labeled samples

### Recommended Dataset Size

- **Training**: 5000+ labeled samples
- **Validation**: 1000+ labeled samples
- **Testing**: 1000+ labeled samples

## Data Sources

### Public Datasets

1. **CADP (Car Accident Detection and Prediction)**: Traffic accident datasets
2. **CCD (Car Crash Dataset)**: Car crash detection datasets
3. **BDD100K**: Berkeley DeepDrive dataset with traffic scenarios
4. **Cityscapes**: Urban scene understanding dataset

### Custom Data

- Dashcam footage
- Traffic camera footage
- Insurance claim videos
- Police report videos

## Quality Assurance

### Annotation Quality

1. **Consistency**: Ensure consistent labeling across annotators
2. **Accuracy**: Verify labels are correct
3. **Completeness**: Ensure all relevant frames are labeled

### Data Quality

1. **Video Quality**: Check for blur, low resolution, or corruption
2. **Detection Quality**: Verify objects are detected correctly
3. **Feature Quality**: Check extracted features are meaningful

## Troubleshooting

### Issue: Low annotation accuracy

**Solution**:
- Use multiple annotators and average their labels
- Provide clear annotation guidelines
- Review and correct annotations regularly

### Issue: Unbalanced dataset

**Solution**:
- Collect more samples from underrepresented classes
- Use data augmentation techniques
- Use class weights in training

### Issue: Poor feature extraction

**Solution**:
- Improve object detection accuracy
- Adjust feature extraction parameters
- Add more relevant features

## Next Steps

After preparing your data:

1. **Train the model** using the prepared data
2. **Evaluate performance** on test set
3. **Fine-tune** hyperparameters if needed
4. **Deploy** the trained model for inference


