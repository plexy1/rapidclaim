"""Configuration loader utility"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str = "config.yaml"):
    """Save configuration to YAML file"""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def ensure_directories(config: Dict[str, Any]):
    """Create necessary directories if they don't exist"""
    paths = [
        config['data']['video_dir'],
        config['data']['annotations_dir'],
        config['data']['models_dir'],
        config['data']['output_dir']
    ]
    
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


