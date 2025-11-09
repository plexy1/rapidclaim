"""Script to process video and detect faults"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from src.pipeline import VideoProcessor
from src.utils.config_loader import load_config


def main():
    parser = argparse.ArgumentParser(description='Process video and detect faults')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to input video')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save output video')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Disable visualization')
    parser.add_argument('--save-results', type=str, default=None,
                       help='Path to save results JSON')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = VideoProcessor(config_path=args.config)
    
    # Set output path
    if args.output is None:
        output_dir = processor.config['data']['output_dir']
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        args.output = os.path.join(output_dir, f'{video_name}_output.mp4')
    
    # Process video
    print(f"Processing video: {args.video}")
    results = processor.process_video(
        args.video,
        output_path=args.output if not args.no_visualize else None,
        visualize=not args.no_visualize
    )
    
    # Get summary
    summary = processor.get_summary(results)
    print("\nSummary:")
    print(f"  Total frames: {summary['total_frames']}")
    print(f"  Frames with fault: {summary['frames_with_fault']}")
    print(f"  Vehicle fault counts: {summary['vehicle_fault_counts']}")
    
    # Save results if requested
    if args.save_results:
        import json
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (int, float, str, bool)) or obj is None:
                return obj
            elif hasattr(obj, 'item'):  # numpy types
                return obj.item()
            else:
                return str(obj)
        
        serializable_results = convert_to_serializable(results)
        with open(args.save_results, 'w') as f:
            json.dump({
                'summary': summary,
                'results': serializable_results
            }, f, indent=2)
        print(f"\nResults saved to {args.save_results}")
    
    print(f"\nOutput video saved to {args.output}")


if __name__ == '__main__':
    main()


