"""Batch process multiple videos"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
from src.pipeline import VideoProcessor
from src.utils.config_loader import load_config, ensure_directories
import json


def batch_process_videos(video_dir: str, output_dir: str, config_path: str = "config.yaml"):
    """
    Process all videos in a directory
    
    Args:
        video_dir: Directory containing video files
        output_dir: Directory to save output videos
        config_path: Path to config file
    """
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    processor = VideoProcessor(config_path=config_path)
    
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(video_dir.glob(f'*{ext}')))
        video_files.extend(list(video_dir.glob(f'*{ext.upper()}')))
    
    if not video_files:
        print(f"No video files found in {video_dir}")
        return
    
    print(f"Found {len(video_files)} video files")
    
    # Process each video
    all_results = {}
    for video_file in video_files:
        print(f"\nProcessing: {video_file.name}")
        
        try:
            # Set output path
            output_video = output_dir / f"{video_file.stem}_output{video_file.suffix}"
            output_results = output_dir / f"{video_file.stem}_results.json"
            
            # Process video
            results = processor.process_video(
                video_path=str(video_file),
                output_path=str(output_video),
                visualize=True
            )
            
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
                    'video': str(video_file),
                    'summary': summary,
                    'results': serializable_results
                }, f, indent=2)
            
            all_results[video_file.name] = summary
            
            print(f"  Output: {output_video}")
            print(f"  Results: {output_results}")
            print(f"  Frames with fault: {summary['frames_with_fault']}")
            
        except Exception as e:
            print(f"  Error processing {video_file.name}: {e}")
            all_results[video_file.name] = {'error': str(e)}
    
    # Save summary
    summary_file = output_dir / "batch_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nBatch processing complete!")
    print(f"Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='Batch process videos')
    parser.add_argument('--video-dir', type=str, required=True,
                       help='Directory containing video files')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save output videos')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    batch_process_videos(args.video_dir, args.output_dir, args.config)


if __name__ == '__main__':
    main()


