"""Training script for fault classifier"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import pandas as pd
from pathlib import Path
from src.classification import FaultClassifier
from src.utils.config_loader import load_config, ensure_directories


def load_training_data(data_path: str):
    """Load training data from JSON or CSV file"""
    data_path = Path(data_path)
    
    if data_path.suffix == '.json':
        with open(data_path, 'r') as f:
            data = json.load(f)
        features_list = data.get('features', [])
        labels = data.get('labels', [])
    elif data_path.suffix == '.csv':
        df = pd.read_csv(data_path)
        # Assume last column is label
        features_list = df.iloc[:, :-1].to_dict('records')
        labels = df.iloc[:, -1].tolist()
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")
    
    return features_list, labels


def main():
    parser = argparse.ArgumentParser(description='Train fault classifier')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to training data (JSON or CSV)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save trained model')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    ensure_directories(config)
    
    # Set output path
    if args.output is None:
        model_dir = config['data']['models_dir']
        model_type = config['fault_classification']['model_type']
        args.output = os.path.join(model_dir, f'fault_classifier_{model_type}.pkl')
    
    # Load training data
    print(f"Loading training data from {args.data}...")
    features_list, labels = load_training_data(args.data)
    print(f"Loaded {len(features_list)} samples")
    
    # Initialize classifier
    classifier = FaultClassifier(
        model_type=config['fault_classification']['model_type']
    )
    
    # Train
    print("Training classifier...")
    metrics = classifier.train(features_list, labels, test_size=args.test_size)
    
    # Save model
    print(f"Saving model to {args.output}...")
    classifier.save_model(args.output)
    
    # Print feature importance
    importance = classifier.get_feature_importance()
    if importance:
        print("\nFeature Importance:")
        for feature, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {feature}: {imp:.4f}")
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main()


