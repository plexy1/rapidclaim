"""Setup script to create necessary directories"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_directories():
    """Create all necessary directories for the project"""
    # Try to load config, but don't fail if it's not available
    try:
        from src.utils.config_loader import load_config, ensure_directories
        config = load_config("config.yaml")
        ensure_directories(config)
        print("Loaded configuration from config.yaml")
    except Exception as e:
        print(f"Note: Could not load config.yaml ({e}), creating directories manually")
    
    # Create additional directories
    directories = [
        "data/videos",
        "data/annotations",
        "data/images",
        "data/training",
        "models",
        "outputs",
        "outputs/videos",
        "outputs/results",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Create placeholder files
    placeholder_files = {
        "data/videos/.gitkeep": "# Place your video files here (.mp4, .avi, etc.)",
        "data/annotations/.gitkeep": "# Annotation files will be saved here",
        "data/images/.gitkeep": "# Place test images here",
        "outputs/.gitkeep": "# Output files will be saved here"
    }
    
    for file_path, content in placeholder_files.items():
        file = Path(file_path)
        if not file.exists():
            file.write_text(content)
            print(f"Created placeholder: {file_path}")
    
    print("\n[SUCCESS] Directory structure created successfully!")
    print("\nNext steps:")
    print("1. Import your videos: python scripts/import_videos.py --source 'Videos(Raw)'")
    print("2. Quick test: python scripts/quick_test.py --video data/videos/download.MP4")
    print("3. Process full video: python scripts/process_video.py --video data/videos/download.MP4 --output outputs/output.mp4")


if __name__ == '__main__':
    setup_directories()
