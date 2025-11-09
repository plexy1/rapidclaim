"""Fault classification model"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from xgboost import XGBClassifier


class FaultClassifier:
    """Classifier for determining driver fault"""
    
    def __init__(self, model_type: str = "xgboost", model_path: Optional[str] = None):
        """
        Initialize fault classifier
        
        Args:
            model_type: Type of model ('xgboost', 'random_forest', 'neural_network')
            model_path: Path to saved model (if available)
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = [
            'red_light_violation',
            'pedestrian_crossing_violation',
            'collision_proximity',
            'vehicle_speed',
            'time_to_collision',
            'intersection_zone',
            'traffic_light_state_duration'
        ]
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._initialize_model()
            # Don't show warnings - this is expected behavior when model hasn't been trained yet
            # The system works fine without a trained fault classifier
            # (object detection and tracking still work)
    
    def _initialize_model(self):
        """Initialize model based on type"""
        if self.model_type == "xgboost":
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def prepare_features(self, features_list: List[Dict]) -> np.ndarray:
        """Prepare features for model input"""
        if not features_list:
            return np.array([]).reshape(0, len(self.feature_names))
        
        feature_matrix = []
        for features in features_list:
            feature_vector = [features.get(name, 0) for name in self.feature_names]
            feature_matrix.append(feature_vector)
        
        return np.array(feature_matrix)
    
    def train(self, features_list: List[Dict], labels: List[int],
              test_size: float = 0.2, random_state: int = 42):
        """
        Train the fault classifier
        
        Args:
            features_list: List of feature dictionaries
            labels: List of labels (1 = at fault, 0 = not at fault)
            test_size: Proportion of data for testing
            random_state: Random state for reproducibility
        """
        # Prepare features
        X = self.prepare_features(features_list)
        y = np.array(labels)
        
        if len(X) == 0:
            raise ValueError("No features provided for training")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        
        print(f"Training Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def is_trained(self) -> bool:
        """Check if model has been trained"""
        if self.model is None:
            return False
        
        # Check if model has been fitted
        # For XGBoost - check if booster exists and has trees
        if hasattr(self.model, 'get_booster'):
            try:
                booster = self.model.get_booster()
                # Check if booster has any trees (trained model)
                num_trees = booster.num_boosted_rounds()
                return num_trees > 0
            except (AttributeError, ValueError, RuntimeError):
                # Model hasn't been trained yet
                return False
        
        # For scikit-learn models (RandomForest, etc.)
        if hasattr(self.model, 'n_features_in_'):
            return self.model.n_features_in_ is not None
        
        # Fallback: try to check if model has been fitted
        # This is a safe check - if predict would fail, model isn't trained
        try:
            # Try a dummy prediction to see if model is trained
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dummy_X = np.zeros((1, len(self.feature_names)))
                self.model.predict(dummy_X)
            return True
        except:
            return False
    
    def predict(self, features_list: List[Dict]) -> List[Dict]:
        """
        Predict fault for given features
        
        Args:
            features_list: List of feature dictionaries
            
        Returns:
            List of predictions with format:
            {
                'vehicle_track_id': int,
                'fault_probability': float,
                'is_at_fault': bool
            }
        """
        if not features_list:
            return []
        
        # Check if model is trained
        if not self.is_trained():
            # Model not trained - return empty predictions
            # This allows the system to work without a trained fault classifier
            return []
        
        X = self.prepare_features(features_list)
        
        if self.model is None:
            return []
        
        try:
            # Predict probabilities
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X)[:, 1]  # Probability of fault
            else:
                predictions = self.model.predict(X)
                probabilities = predictions.astype(float)
            
            # Make predictions
            predictions = self.model.predict(X)
            
            results = []
            for i, features in enumerate(features_list):
                result = {
                    'vehicle_track_id': features.get('vehicle_track_id'),
                    'fault_probability': float(probabilities[i]),
                    'is_at_fault': bool(predictions[i] == 1),
                    'frame': features.get('frame')
                }
                results.append(result)
            
            return results
        except Exception as e:
            # If prediction fails, return empty list
            print(f"Warning: Could not predict faults: {e}")
            return []
    
    def save_model(self, model_path: str):
        """Save model to file"""
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'model_type': self.model_type,
                'feature_names': self.feature_names
            }, f)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load model from file"""
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.model_type = data.get('model_type', self.model_type)
            self.feature_names = data.get('feature_names', self.feature_names)
        print(f"Model loaded from {model_path}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from model"""
        if self.model is None:
            return {}
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            return dict(zip(self.feature_names, importances))
        else:
            return {}


