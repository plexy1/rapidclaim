"""Object detection using YOLOv8"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
import torch


class ObjectDetector:
    """YOLOv8-based object detector for pedestrians, vehicles, and traffic lights"""
    
    def __init__(self, model_name: str = "yolov8n.pt", 
                 confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.45,
                 device: Optional[str] = None):
        """
        Initialize object detector
        
        Args:
            model_name: YOLOv8 model name or path
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on (cuda/cpu)
        """
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # COCO class IDs
        self.class_names = self.model.names
        self.pedestrian_id = 0
        self.vehicle_ids = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.traffic_light_id = 9
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in a frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of detections with format:
            {
                'class_id': int,
                'class_name': str,
                'confidence': float,
                'bbox': [x1, y1, x2, y2],
                'center': (x, y)
            }
        """
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            
            for box in boxes:
                try:
                    # Validate box data exists
                    if box.cls is None or len(box.cls) == 0:
                        continue
                    if box.conf is None or len(box.conf) == 0:
                        continue
                    if box.xyxy is None or len(box.xyxy) == 0:
                        continue
                    
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Get bounding box coordinates
                    bbox_array = box.xyxy[0].cpu().numpy()
                    if len(bbox_array) < 4:
                        continue
                    
                    x1, y1, x2, y2 = bbox_array[:4]
                    
                    # Validate bbox values
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    detection = {
                        'class_id': class_id,
                        'class_name': self.class_names.get(class_id, f'class_{class_id}'),
                        'confidence': confidence,
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'center': ((x1 + x2) / 2, (y1 + y2) / 2)
                    }
                    detections.append(detection)
                except (IndexError, ValueError, AttributeError) as e:
                    # Skip invalid detections
                    continue
        
        return detections
    
    def filter_detections(self, detections: List[Dict], 
                         class_ids: Optional[List[int]] = None,
                         min_confidence: Optional[float] = None) -> List[Dict]:
        """
        Filter detections by class and confidence
        
        Args:
            detections: List of detections
            class_ids: List of class IDs to keep (None = all)
            min_confidence: Minimum confidence threshold
            
        Returns:
            Filtered list of detections
        """
        filtered = detections
        
        if class_ids is not None:
            filtered = [d for d in filtered if d['class_id'] in class_ids]
        
        if min_confidence is not None:
            filtered = [d for d in filtered if d['confidence'] >= min_confidence]
        
        return filtered
    
    def get_pedestrians(self, detections: List[Dict]) -> List[Dict]:
        """Get pedestrian detections"""
        return self.filter_detections(detections, class_ids=[self.pedestrian_id])
    
    def get_vehicles(self, detections: List[Dict]) -> List[Dict]:
        """Get vehicle detections"""
        return self.filter_detections(detections, class_ids=self.vehicle_ids)
    
    def get_traffic_lights(self, detections: List[Dict]) -> List[Dict]:
        """Get traffic light detections"""
        return self.filter_detections(detections, class_ids=[self.traffic_light_id])


