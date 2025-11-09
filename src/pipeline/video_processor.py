"""End-to-end video processing pipeline"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import yaml
from tqdm import tqdm

from src.detection.object_detector import ObjectDetector
from src.detection.traffic_light_classifier import TrafficLightClassifier
from src.tracking import MultiObjectTracker
from src.features import FeatureExtractor
from src.classification import FaultClassifier
from src.utils.config_loader import load_config, ensure_directories


class VideoProcessor:
    """Main pipeline for processing videos and determining fault"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize video processor with configuration"""
        self.config = load_config(config_path)
        ensure_directories(self.config)
        
        # Initialize components
        self.detector = ObjectDetector(
            model_name=self.config['detection']['model_name'],
            confidence_threshold=self.config['detection']['confidence_threshold'],
            iou_threshold=self.config['detection']['iou_threshold']
        )
        
        self.traffic_light_classifier = TrafficLightClassifier(
            model_path=self.config['traffic_light'].get('state_model_path'),
            temporal_window=self.config['traffic_light']['temporal_smoothing_window'],
            confidence_threshold=self.config['traffic_light']['state_confidence_threshold']
        )
        
        self.tracker = MultiObjectTracker(
            max_age=self.config['tracking']['max_age'],
            min_hits=self.config['tracking']['min_hits'],
            iou_threshold=self.config['tracking']['iou_threshold']
        )
        
        self.feature_extractor = FeatureExtractor(
            proximity_threshold=self.config['features']['proximity_threshold'],
            intersection_zone_threshold=self.config['features']['intersection_zone_threshold'],
            red_light_violation_threshold=self.config['features']['red_light_violation_threshold'],
            speed_calculation_window=self.config['features']['speed_calculation_window']
        )
        
        self.fault_classifier = FaultClassifier(
            model_type=self.config['fault_classification']['model_type'],
            model_path=self.config['fault_classification'].get('model_path')
        )
        
        self.frame_skip = self.config['video'].get('frame_skip', 1)
        self.fps = self.config['video'].get('fps', 30)
    
    def process_frame(self, frame: np.ndarray, frame_number: int) -> Dict:
        """
        Process a single frame
        
        Args:
            frame: Input frame
            frame_number: Frame number
            
        Returns:
            Dictionary with detections, tracks, features, and predictions
        """
        # Detect objects
        detections = self.detector.detect(frame)
        
        # Separate detections by type
        pedestrians = self.detector.get_pedestrians(detections)
        vehicles = self.detector.get_vehicles(detections)
        traffic_lights = self.detector.get_traffic_lights(detections)
        
        # Classify traffic light states
        for tl in traffic_lights:
            try:
                # Validate traffic light detection has required fields
                if 'bbox' not in tl or not tl.get('bbox'):
                    continue
                
                state, confidence = self.traffic_light_classifier.classify(
                    frame, tl, tl.get('track_id')
                )
                self.traffic_light_classifier.update_detection(tl, state, confidence)
            except (KeyError, IndexError, ValueError, TypeError) as e:
                # Skip invalid traffic light detections
                continue
        
        # Track objects
        all_tracks = pedestrians + vehicles + traffic_lights
        tracked_objects = self.tracker.update(all_tracks)
        
        # Separate tracked objects
        tracked_pedestrians = [obj for obj in tracked_objects if obj['class_id'] == 0]
        tracked_vehicles = [obj for obj in tracked_objects if obj['class_id'] in [2, 3, 5, 7]]
        tracked_traffic_lights = [obj for obj in tracked_objects if obj['class_id'] == 9]
        
        # Extract features
        features = self.feature_extractor.extract_features(
            frame=frame,
            vehicle_tracks=tracked_vehicles,
            pedestrian_tracks=tracked_pedestrians,
            traffic_light_tracks=tracked_traffic_lights,
            tracker=self.tracker,
            current_frame=frame_number,
            fps=self.fps
        )
        
        # Classify fault
        fault_predictions = []
        if features:
            fault_predictions = self.fault_classifier.predict(features)
        
        return {
            'frame_number': frame_number,
            'detections': detections,
            'tracked_pedestrians': tracked_pedestrians,
            'tracked_vehicles': tracked_vehicles,
            'tracked_traffic_lights': tracked_traffic_lights,
            'features': features,
            'fault_predictions': fault_predictions
        }
    
    def process_video(self, video_path: str, output_path: Optional[str] = None,
                     visualize: bool = True) -> List[Dict]:
        """
        Process entire video
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (if None, only return results)
            visualize: Whether to visualize results
            
        Returns:
            List of frame results
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.fps = fps if fps > 0 else self.fps
        
        # Setup output video writer
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        results = []
        frame_number = 0
        
        # Process frames
        with tqdm(total=total_frames, desc="Processing video") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if needed
                if frame_number % self.frame_skip != 0:
                    frame_number += 1
                    pbar.update(1)
                    continue
                
                # Process frame
                result = self.process_frame(frame, frame_number)
                results.append(result)
                
                # Visualize if requested
                if visualize or output_path:
                    vis_frame = self.visualize_frame(frame, result)
                    if output_path and out:
                        out.write(vis_frame)
                
                frame_number += 1
                pbar.update(1)
        
        cap.release()
        if out:
            out.release()
        
        return results
    
    def visualize_frame(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """Visualize detections and predictions on frame"""
        vis_frame = frame.copy()
        
        # Draw pedestrians
        for ped in result.get('tracked_pedestrians', []):
            bbox = ped.get('bbox', [])
            if len(bbox) < 4:
                continue
            
            try:
                track_id = ped.get('track_id', -1)
                cv2.rectangle(vis_frame, 
                             (int(bbox[0]), int(bbox[1])),
                             (int(bbox[2]), int(bbox[3])),
                             (0, 255, 0), 2)
                cv2.putText(vis_frame, f"Ped {track_id}",
                           (int(bbox[0]), int(bbox[1]) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            except (IndexError, ValueError, TypeError):
                continue
        
        # Draw vehicles
        for vehicle in result.get('tracked_vehicles', []):
            bbox = vehicle.get('bbox', [])
            if len(bbox) < 4:
                continue
            
            try:
                track_id = vehicle.get('track_id', -1)
                
                # Check if vehicle is at fault
                fault_pred = next((p for p in result.get('fault_predictions', []) 
                                 if p.get('vehicle_track_id') == track_id), None)
                
                if fault_pred and fault_pred.get('is_at_fault'):
                    color = (0, 0, 255)  # Red for at fault
                    label = f"Vehicle {track_id} - AT FAULT"
                else:
                    color = (255, 0, 0)  # Blue for not at fault
                    label = f"Vehicle {track_id}"
                
                cv2.rectangle(vis_frame,
                             (int(bbox[0]), int(bbox[1])),
                             (int(bbox[2]), int(bbox[3])),
                             color, 2)
                cv2.putText(vis_frame, label,
                           (int(bbox[0]), int(bbox[1]) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            except (IndexError, ValueError, TypeError):
                continue
        
        # Draw traffic lights
        for tl in result.get('tracked_traffic_lights', []):
            bbox = tl.get('bbox', [])
            if len(bbox) < 4:
                continue
            
            try:
                state = tl.get('traffic_light_state', 'unknown')
                track_id = tl.get('track_id', -1)
                
                # Color based on state
                color_map = {
                    'red': (0, 0, 255),
                    'yellow': (0, 255, 255),
                    'green': (0, 255, 0),
                    'unknown': (128, 128, 128)
                }
                color = color_map.get(state, (128, 128, 128))
                
                cv2.rectangle(vis_frame,
                             (int(bbox[0]), int(bbox[1])),
                             (int(bbox[2]), int(bbox[3])),
                             color, 2)
                cv2.putText(vis_frame, f"TL {track_id} {state.upper()}",
                           (int(bbox[0]), int(bbox[1]) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            except (IndexError, ValueError, TypeError):
                continue
        
        return vis_frame
    
    def get_summary(self, results: List[Dict]) -> Dict:
        """Get summary statistics from results"""
        total_frames = len(results)
        total_at_fault = sum(1 for r in results 
                           if any(p['is_at_fault'] for p in r['fault_predictions']))
        
        vehicle_faults = {}
        for result in results:
            for pred in result['fault_predictions']:
                vehicle_id = pred['vehicle_track_id']
                if pred['is_at_fault']:
                    vehicle_faults[vehicle_id] = vehicle_faults.get(vehicle_id, 0) + 1
        
        return {
            'total_frames': total_frames,
            'frames_with_fault': total_at_fault,
            'vehicle_fault_counts': vehicle_faults
        }

