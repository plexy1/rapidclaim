"""Script to import videos from a source directory to the project structure"""

import sys
import os
import shutil
from pathlib import Path
import argparse


def import_videos(source_dir: str, target_dir: str = "data/videos", copy: bool = True):
    """
    Import videos from source directory to target directory
    
    Args:
        source_dir: Source directory containing videos
        target_dir: Target directory (default: data/videos)
        copy: If True, copy files. If False, move files.
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    # Create target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)
    
    if not source_dir.exists():
        print(f"Error: Source directory '{source_dir}' does not exist")
        return
    
    # Supported video formats
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.MP4', '.AVI', '.MOV', '.MKV', '.FLV']
    
    # Find all video files
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(source_dir.glob(f'*{ext}')))
        video_files.extend(list(source_dir.rglob(f'*{ext}')))  # Recursive search
    
    if not video_files:
        print(f"No video files found in '{source_dir}'")
        print(f"Looking for files with extensions: {', '.join(video_extensions)}")
        return
    
    print(f"Found {len(video_files)} video file(s)")
    print(f"\n{'Copying' if copy else 'Moving'} videos to '{target_dir}'...")
    print("-" * 50)
    
    imported = 0
    skipped = 0
    
    for video_file in video_files:
        # Clean filename (remove spaces, special characters)
        clean_name = video_file.stem.replace(' ', '_').replace('(', '').replace(')', '')
        target_file = target_dir / f"{clean_name}{video_file.suffix}"
        
        # Skip if file already exists
        if target_file.exists():
            print(f"[SKIP] Skipped (already exists): {video_file.name}")
            skipped += 1
            continue
        
        try:
            if copy:
                shutil.copy2(video_file, target_file)
                action = "Copied"
            else:
                shutil.move(str(video_file), str(target_file))
                action = "Moved"
            
            print(f"[OK] {action}: {video_file.name} -> {target_file.name}")
            imported += 1
        except Exception as e:
            print(f"[ERROR] Error processing {video_file.name}: {e}")
    
    print("-" * 50)
    print(f"\nSummary:")
    print(f"  Imported: {imported}")
    print(f"  Skipped: {skipped}")
    print(f"  Total: {len(video_files)}")
    print(f"\nVideos are now in: {target_dir.absolute()}")


def main():
    parser = argparse.ArgumentParser(description='Import videos to project structure')
    parser.add_argument('--source', type=str, required=True,
                       help='Source directory containing videos')
    parser.add_argument('--target', type=str, default='data/videos',
                       help='Target directory (default: data/videos)')
    parser.add_argument('--move', action='store_true',
                       help='Move files instead of copying (default: copy)')
    
    args = parser.parse_args()
    
    import_videos(args.source, args.target, copy=not args.move)


if __name__ == '__main__':
    # If run without arguments, try to find common video directories
    if len(sys.argv) == 1:
        print("Video Import Script")
        print("=" * 50)
        print("\nUsage:")
        print("  python scripts/import_videos.py --source <source_directory>")
        print("\nExample:")
        print("  python scripts/import_videos.py --source 'Videos(Raw)'")
        print("  python scripts/import_videos.py --source 'Videos(Raw)' --move  # Move instead of copy")
        print("\nLooking for common video directories...")
        
        # Check for common directory names
        current_dir = Path('.')
        common_names = ['Videos(Raw)', 'Videos', 'videos', 'Video', 'Raw Videos']
        
        found_dirs = []
        for name in common_names:
            if (current_dir / name).exists():
                found_dirs.append(current_dir / name)
        
        if found_dirs:
            print(f"\nFound potential video directories:")
            for i, dir_path in enumerate(found_dirs, 1):
                video_count = len(list(dir_path.rglob('*.mp4'))) + len(list(dir_path.rglob('*.MP4')))
                print(f"  {i}. {dir_path} ({video_count} MP4 files found)")
            
            if len(found_dirs) == 1:
                print(f"\nAuto-importing from: {found_dirs[0]}")
                import_videos(str(found_dirs[0]))
            else:
                print("\nPlease specify which directory to import from using --source")
        else:
            print("\nNo common video directories found.")
            print("Please specify the source directory using --source")
    else:
        main()

