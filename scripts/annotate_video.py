"""Interactive script for annotating videos with fault labels"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import cv2
import json
from pathlib import Path
from src.pipeline import VideoProcessor
from src.utils.config_loader import load_config


class VideoAnnotator:
    """Interactive video annotator for fault labeling"""
    
    def __init__(self, processor: VideoProcessor):
        self.processor = processor
        self.annotations = {}
        self.current_video = None
        self.current_frame = 0
        self.playing = True
    
    def annotate_video(self, video_path: str, output_path: str):
        """Annotate video interactively"""
        self.current_video = Path(video_path).stem
        self.annotations[self.current_video] = {}
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print("\nControls:")
        print("  SPACE: Play/Pause")
        print("  'a': Mark current frame as AT FAULT")
        print("  'n': Mark current frame as NOT AT FAULT")
        print("  LEFT/RIGHT: Navigate frames")
        print("  's': Save annotations")
        print("  'q': Quit")
        print()
        
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            result = self.processor.process_frame(frame, self.current_frame)
            
            # Visualize
            vis_frame = self.processor.visualize_frame(frame, result)
            
            # Add annotation status
            frame_label = self.annotations[self.current_video].get(str(self.current_frame))
            if frame_label:
                label_text = "AT FAULT" if frame_label else "NOT AT FAULT"
                cv2.putText(vis_frame, f"Label: {label_text}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Add frame info
            cv2.putText(vis_frame, f"Frame: {self.current_frame}/{total_frames}",
                       (10, vis_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Video Annotator', vis_frame)
            
            key = cv2.waitKey(30 if self.playing else 0) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.playing = not self.playing
            elif key == ord('a'):
                self.annotations[self.current_video][str(self.current_frame)] = True
                print(f"Frame {self.current_frame}: Marked as AT FAULT")
            elif key == ord('n'):
                self.annotations[self.current_video][str(self.current_frame)] = False
                print(f"Frame {self.current_frame}: Marked as NOT AT FAULT")
            elif key == ord('s'):
                self.save_annotations(output_path)
                print(f"Annotations saved to {output_path}")
            elif key == 81 or key == 2:  # Left arrow
                self.current_frame = max(0, self.current_frame - 1)
                self.playing = False
            elif key == 83 or key == 3:  # Right arrow
                self.current_frame = min(total_frames - 1, self.current_frame + 1)
                self.playing = False
            
            if self.playing:
                self.current_frame += 1
                if self.current_frame >= total_frames:
                    self.current_frame = 0
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Save annotations
        self.save_annotations(output_path)
    
    def save_annotations(self, output_path: str):
        """Save annotations to JSON file"""
        # Load existing annotations if file exists
        if Path(output_path).exists():
            with open(output_path, 'r') as f:
                existing = json.load(f)
                existing.update(self.annotations)
                self.annotations = existing
        
        with open(output_path, 'w') as f:
            json.dump(self.annotations, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Annotate videos with fault labels')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to video file')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save annotations (JSON)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = VideoProcessor(config_path=args.config)
    
    # Create annotator
    annotator = VideoAnnotator(processor)
    
    # Annotate video
    annotator.annotate_video(args.video, args.output)
    
    print(f"\nAnnotations saved to {args.output}")


if __name__ == '__main__':
    main()


