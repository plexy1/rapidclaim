"""Quick test script to process a video and see results"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
from src.pipeline import VideoProcessor


def quick_test(video_path: str, output_dir: str = "outputs", max_frames: int = 100):
    """
    Quick test on a video (process first N frames)
    
    Args:
        video_path: Path to video file
        output_dir: Output directory
        max_frames: Maximum number of frames to process (for quick testing)
    """
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Error: Video file '{video_path}' not found")
        return
    
    print(f"Quick Test: Processing video")
    print(f"  Video: {video_path}")
    print(f"  Max frames: {max_frames}")
    print("-" * 50)
    
    # Initialize processor
    try:
        processor = VideoProcessor(config_path="config.yaml")
    except Exception as e:
        print(f"Error initializing processor: {e}")
        print("Make sure config.yaml exists and dependencies are installed")
        return
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set output path
    output_video = output_dir / f"{video_path.stem}_test_output.mp4"
    output_results = output_dir / f"{video_path.stem}_test_results.json"
    
    # Process video (limited frames for testing)
    print("\nProcessing video (this may take a few minutes)...")
    print("Note: For quick testing, processing first {} frames".format(max_frames))
    
    try:
        import cv2
        import json
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video '{video_path}'")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")
        print(f"  Processing: {min(max_frames, total_frames)} frames")
        
        # Setup output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
        
        results = []
        frame_number = 0
        processed_frames = 0
        
        while processed_frames < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            try:
                result = processor.process_frame(frame, frame_number)
                results.append(result)
                
                # Visualize
                vis_frame = processor.visualize_frame(frame, result)
                out.write(vis_frame)
                
                processed_frames += 1
                
                # Print progress every 10 frames
                if processed_frames % 10 == 0:
                    print(f"  Processed {processed_frames}/{min(max_frames, total_frames)} frames...")
                
            except Exception as e:
                print(f"  Warning: Error processing frame {frame_number}: {e}")
            
            frame_number += 1
        
        cap.release()
        out.release()
        
        # Get summary
        summary = processor.get_summary(results)
        
        # Save results
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (int, float, str, bool)) or obj is None:
                return obj
            elif hasattr(obj, 'item'):
                return obj.item()
            else:
                return str(obj)
        
        serializable_results = convert_to_serializable(results)
        with open(output_results, 'w') as f:
            json.dump({
                'video': str(video_path),
                'summary': summary,
                'results': serializable_results
            }, f, indent=2)
        
        print("\n" + "=" * 50)
        print("Test Complete!")
        print("=" * 50)
        print(f"\nOutput video: {output_video.absolute()}")
        print(f"Results JSON: {output_results.absolute()}")
        print(f"\nSummary:")
        print(f"  Frames processed: {summary['total_frames']}")
        print(f"  Frames with fault: {summary['frames_with_fault']}")
        print(f"  Vehicle fault counts: {summary['vehicle_fault_counts']}")
        print(f"\n✅ Check the output video to see detections and predictions!")
        
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Quick test on a video')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to video file')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory (default: outputs)')
    parser.add_argument('--max-frames', type=int, default=100,
                       help='Maximum frames to process for quick test (default: 100)')
    
    args = parser.parse_args()
    
    quick_test(args.video, args.output_dir, args.max_frames)


if __name__ == '__main__':
    main()


