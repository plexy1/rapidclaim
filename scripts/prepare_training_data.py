"""Script to prepare training data from annotated videos"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path
from src.pipeline import VideoProcessor
from src.utils.config_loader import load_config, ensure_directories


def load_annotations(annotations_path: str):
    """Load fault annotations from JSON file"""
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    return annotations


def extract_features_from_videos(video_dir: str, annotations: dict,
                                processor: VideoProcessor):
    """Extract features from videos with annotations"""
    features_list = []
    labels = []
    
    video_dir = Path(video_dir)
    video_files = list(video_dir.glob('*.mp4')) + list(video_dir.glob('*.avi'))
    
    for video_file in video_files:
        video_name = video_file.stem
        
        if video_name not in annotations:
            print(f"Warning: No annotations for {video_name}, skipping...")
            continue
        
        print(f"Processing {video_name}...")
        
        # Process video
        results = processor.process_video(
            str(video_file),
            output_path=None,
            visualize=False
        )
        
        # Extract features and labels
        video_annotations = annotations[video_name]
        
        for result in results:
            frame_number = result['frame_number']
            
            # Get fault label for this frame (if available)
            frame_label = video_annotations.get(str(frame_number))
            if frame_label is None:
                # Try to get label for vehicle track IDs
                frame_label = video_annotations.get('vehicle_faults', {})
            
            # Add features
            for feature_dict in result['features']:
                features_list.append(feature_dict)
                
                # Determine label
                vehicle_id = feature_dict['vehicle_track_id']
                if isinstance(frame_label, dict):
                    label = 1 if frame_label.get(str(vehicle_id), False) else 0
                else:
                    label = 1 if frame_label else 0
                
                labels.append(label)
    
    return features_list, labels


def main():
    parser = argparse.ArgumentParser(description='Prepare training data from videos')
    parser.add_argument('--video-dir', type=str, required=True,
                       help='Directory containing videos')
    parser.add_argument('--annotations', type=str, required=True,
                       help='Path to annotations JSON file')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save training data (JSON)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    ensure_directories(config)
    
    # Load annotations
    print(f"Loading annotations from {args.annotations}...")
    annotations = load_annotations(args.annotations)
    
    # Initialize processor
    processor = VideoProcessor(config_path=args.config)
    
    # Extract features
    print("Extracting features from videos...")
    features_list, labels = extract_features_from_videos(
        args.video_dir,
        annotations,
        processor
    )
    
    # Save training data
    print(f"Saving training data to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump({
            'features': features_list,
            'labels': labels
        }, f, indent=2)
    
    print(f"\nTraining data prepared:")
    print(f"  Total samples: {len(features_list)}")
    print(f"  At fault: {sum(labels)}")
    print(f"  Not at fault: {len(labels) - sum(labels)}")


if __name__ == '__main__':
    main()


