"""Traffic light state classification with temporal smoothing"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F


class TrafficLightStateClassifier(nn.Module):
    """Simple CNN for traffic light state classification"""
    
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TrafficLightClassifier:
    """Traffic light state classifier with temporal smoothing"""
    
    STATES = ['red', 'yellow', 'green', 'unknown']
    STATE_TO_ID = {state: idx for idx, state in enumerate(STATES)}
    ID_TO_STATE = {idx: state for state, idx in STATE_TO_ID.items()}
    
    def __init__(self, model_path: Optional[str] = None,
                 temporal_window: int = 5,
                 confidence_threshold: float = 0.7,
                 use_color_based: bool = True):
        """
        Initialize traffic light classifier
        
        Args:
            model_path: Path to trained model (if available)
            temporal_window: Number of frames for temporal smoothing
            confidence_threshold: Minimum confidence for classification
            use_color_based: Use color-based detection as fallback
        """
        self.temporal_window = temporal_window
        self.confidence_threshold = confidence_threshold
        self.use_color_based = use_color_based
        self.state_history = {}  # Track state history per traffic light
        
        # Initialize model if path provided
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if model_path:
            try:
                self.model = TrafficLightStateClassifier(num_classes=3)
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                self.use_color_based = False
            except FileNotFoundError:
                # Model file doesn't exist - this is expected if not trained yet
                # Silently use color-based classification (default behavior)
                self.use_color_based = True
            except Exception as e:
                # Other errors - log but continue with color-based
                print(f"Note: Could not load traffic light model from {model_path}: {e}")
                print("  Using color-based classification (this is the default method)")
                self.use_color_based = True
    
    def extract_traffic_light_roi(self, frame: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """Extract region of interest for traffic light"""
        # Validate bbox
        if not bbox or len(bbox) < 4:
            return None
        
        try:
            x1, y1, x2, y2 = map(int, bbox[:4])
            
            # Validate frame dimensions
            if len(frame.shape) < 2:
                return None
            
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            # Ensure valid bbox
            if x2 <= x1 or y2 <= y1:
                return None
            
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return None
            return roi
        except (IndexError, ValueError, TypeError):
            return None
    
    def classify_by_color(self, roi: np.ndarray) -> Tuple[str, float]:
        """
        Classify traffic light state based on color
        
        Args:
            roi: Region of interest containing traffic light
            
        Returns:
            Tuple of (state, confidence)
        """
        if roi is None or roi.size == 0:
            return 'unknown', 0.0
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Define color ranges (HSV)
        red_lower1 = np.array([0, 50, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 50, 50])
        red_upper2 = np.array([180, 255, 255])
        
        yellow_lower = np.array([20, 50, 50])
        yellow_upper = np.array([30, 255, 255])
        
        green_lower = np.array([40, 50, 50])
        green_upper = np.array([80, 255, 255])
        
        # Create masks
        red_mask = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # Calculate color intensities
        try:
            if len(roi.shape) < 2 or roi.shape[0] == 0 or roi.shape[1] == 0:
                return 'unknown', 0.0
            
            total_pixels = roi.shape[0] * roi.shape[1]
            if total_pixels == 0:
                return 'unknown', 0.0
            
            red_intensity = np.sum(red_mask) / (total_pixels * 255.0)
            yellow_intensity = np.sum(yellow_mask) / (total_pixels * 255.0)
            green_intensity = np.sum(green_mask) / (total_pixels * 255.0)
        except (IndexError, ValueError, ZeroDivisionError):
            return 'unknown', 0.0
        
        # Determine dominant color
        intensities = {
            'red': red_intensity,
            'yellow': yellow_intensity,
            'green': green_intensity
        }
        
        max_state = max(intensities, key=intensities.get)
        max_intensity = intensities[max_state]
        
        # Threshold for confidence - lowered for easier detection
        if max_intensity < 0.05:  # Lowered from 0.1 to 0.05 for better detection
            return 'unknown', max_intensity
        
        return max_state, min(max_intensity * 2, 1.0)  # Scale confidence
    
    def classify_by_model(self, roi: np.ndarray) -> Tuple[str, float]:
        """Classify using neural network model"""
        if self.model is None:
            return self.classify_by_color(roi)
        
        # Preprocess ROI
        roi_resized = cv2.resize(roi, (32, 32))
        roi_normalized = roi_resized.astype(np.float32) / 255.0
        roi_tensor = torch.from_numpy(roi_normalized).permute(2, 0, 1).unsqueeze(0)
        roi_tensor = roi_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(roi_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, pred_id = torch.max(probs, 1)
            
            state = self.ID_TO_STATE[pred_id.item()]
            conf = confidence.item()
            
            if conf < self.confidence_threshold:
                return 'unknown', conf
            
            return state, conf
    
    def classify(self, frame: np.ndarray, detection: Dict, 
                 track_id: Optional[int] = None) -> Tuple[str, float]:
        """
        Classify traffic light state with temporal smoothing
        
        Args:
            frame: Input frame
            detection: Traffic light detection dictionary
            track_id: Tracking ID for temporal smoothing
            
        Returns:
            Tuple of (state, confidence)
        """
        # Validate detection has bbox
        bbox = detection.get('bbox', [])
        if not bbox or len(bbox) < 4:
            return 'unknown', 0.0
        
        roi = self.extract_traffic_light_roi(frame, bbox)
        
        # Classify current frame
        if self.model and not self.use_color_based:
            state, confidence = self.classify_by_model(roi)
        else:
            state, confidence = self.classify_by_color(roi)
        
        # Apply temporal smoothing if track_id provided
        if track_id is not None:
            if track_id not in self.state_history:
                self.state_history[track_id] = deque(maxlen=self.temporal_window)
            
            self.state_history[track_id].append((state, confidence))
            
            # Get most common state in window
            if len(self.state_history[track_id]) >= 3:
                states = [s for s, _ in self.state_history[track_id]]
                state_counts = {}
                for s, c in self.state_history[track_id]:
                    state_counts[s] = state_counts.get(s, 0) + c
                
                # Weighted voting
                state = max(state_counts, key=state_counts.get)
                confidence = state_counts[state] / len(self.state_history[track_id])
        
        return state, confidence
    
    def update_detection(self, detection: Dict, state: str, confidence: float):
        """Update detection dictionary with state information"""
        detection['traffic_light_state'] = state
        detection['traffic_light_confidence'] = confidence
        return detection

