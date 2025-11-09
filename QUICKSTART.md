# Quick Start Guide

This guide will help you get started with RapidClaim in minutes.

## Installation

1. **Install Python 3.8+** (if not already installed)

2. **Clone and install dependencies**:
```bash
cd rapidclaim
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python -c "import torch; import ultralytics; print('Installation successful!')"
```

## Quick Start: Process a Video

1. **Place your video in the data directory**:
```bash
mkdir -p data/videos
# Copy your video to data/videos/input.mp4
```

2. **Process the video**:
```bash
python scripts/process_video.py --video data/videos/input.mp4 --output outputs/output.mp4
```

3. **View results**: The output video will be saved with bounding boxes and fault labels.

## Quick Start: Train a Model

1. **Annotate videos** (create fault labels):
```bash
python scripts/annotate_video.py --video data/videos/input.mp4 --output data/annotations.json
```

2. **Prepare training data**:
```bash
python scripts/prepare_training_data.py \
    --video-dir data/videos \
    --annotations data/annotations.json \
    --output data/training_data.json
```

3. **Train the classifier**:
```bash
python scripts/train_fault_classifier.py \
    --data data/training_data.json \
    --output models/fault_classifier.pkl
```

4. **Use the trained model**: Update `config.yaml` to point to your trained model:
```yaml
fault_classification:
  model_path: "models/fault_classifier.pkl"
```

## Configuration

Edit `config.yaml` to customize:

- **Detection model**: Change `detection.model_name` to use different YOLOv8 models
  - `yolov8n.pt` - Nano (fastest, least accurate)
  - `yolov8s.pt` - Small
  - `yolov8m.pt` - Medium
  - `yolov8l.pt` - Large
  - `yolov8x.pt` - XLarge (slowest, most accurate)

- **Confidence threshold**: Adjust `detection.confidence_threshold` (0.0-1.0)

- **Frame skip**: Process every Nth frame by setting `video.frame_skip`

## Example Workflow

### 1. Single Video Processing
```bash
# Process a video and visualize results
python scripts/process_video.py \
    --video data/videos/accident.mp4 \
    --output outputs/accident_analyzed.mp4 \
    --save-results outputs/accident_results.json
```

### 2. Batch Processing
```bash
# Process multiple videos
for video in data/videos/*.mp4; do
    python scripts/process_video.py \
        --video "$video" \
        --output "outputs/$(basename $video)" \
        --save-results "outputs/$(basename $video .mp4).json"
done
```

### 3. Training Pipeline
```bash
# Step 1: Annotate videos
python scripts/annotate_video.py --video data/videos/video1.mp4 --output data/annotations.json

# Step 2: Prepare training data
python scripts/prepare_training_data.py \
    --video-dir data/videos \
    --annotations data/annotations.json \
    --output data/training_data.json

# Step 3: Train model
python scripts/train_fault_classifier.py \
    --data data/training_data.json \
    --output models/fault_classifier.pkl
```

## Troubleshooting

### Issue: CUDA out of memory
**Solution**: Use a smaller YOLOv8 model (yolov8n.pt) or reduce batch size

### Issue: Slow processing
**Solution**: 
- Increase `video.frame_skip` to process fewer frames
- Use GPU acceleration (install CUDA)
- Use smaller YOLOv8 model

### Issue: Poor detection accuracy
**Solution**:
- Use larger YOLOv8 model (yolov8l.pt or yolov8x.pt)
- Adjust confidence threshold
- Ensure good video quality and lighting

### Issue: Traffic light not detected correctly
**Solution**:
- Check camera angle and lighting
- Adjust traffic light classification thresholds in config.yaml
- Consider training a custom traffic light classifier

## Next Steps

1. **Read the full README.md** for detailed documentation
2. **Check examples/example_usage.py** for code examples
3. **Customize config.yaml** for your specific use case
4. **Train on your own data** for better accuracy

## Support

For issues or questions:
- Check the README.md for detailed documentation
- Open an issue on GitHub
- Review the examples in the examples/ directory


