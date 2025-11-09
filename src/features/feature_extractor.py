"""Feature extraction for fault classification"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import math


class FeatureExtractor:
    """Extract features for fault classification"""
    
    def __init__(self, 
                 proximity_threshold: float = 50,
                 intersection_zone_threshold: float = 100,
                 red_light_violation_threshold: int = 5,
                 speed_calculation_window: int = 10):
        """
        Initialize feature extractor
        
        Args:
            proximity_threshold: Proximity threshold in pixels
            intersection_zone_threshold: Intersection zone threshold in pixels
            red_light_violation_threshold: Frames after red light to consider violation
            speed_calculation_window: Window for speed calculation
        """
        self.proximity_threshold = proximity_threshold
        self.intersection_zone_threshold = intersection_zone_threshold
        self.red_light_violation_threshold = red_light_violation_threshold
        self.speed_calculation_window = speed_calculation_window
        
        # Track state over time
        self.vehicle_states: Dict[int, List[Dict]] = defaultdict(list)
        self.traffic_light_states: Dict[int, List[Dict]] = defaultdict(list)
        self.red_light_frames: Dict[int, int] = {}  # Track when red light started
    
    def calculate_distance(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate distance between two bounding boxes (center to center)"""
        # Validate bbox length
        if len(bbox1) < 4 or len(bbox2) < 4:
            return float('inf')
        
        try:
            cx1 = (bbox1[0] + bbox1[2]) / 2
            cy1 = (bbox1[1] + bbox1[3]) / 2
            cx2 = (bbox2[0] + bbox2[2]) / 2
            cy2 = (bbox2[1] + bbox2[3]) / 2
            
            return math.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)
        except (IndexError, TypeError, ValueError):
            return float('inf')
    
    def calculate_speed(self, track_history: List[Dict], fps: float = 30.0) -> float:
        """Calculate speed in pixels per frame"""
        if len(track_history) < 2:
            return 0.0
        
        recent = track_history[-self.speed_calculation_window:]
        if len(recent) < 2:
            return 0.0
        
        total_distance = 0.0
        valid_pairs = 0
        for i in range(1, len(recent)):
            try:
                prev_entry = recent[i-1]
                curr_entry = recent[i]
                
                # Validate entries have bbox
                if 'bbox' not in prev_entry or 'bbox' not in curr_entry:
                    continue
                
                prev_bbox = prev_entry.get('bbox', [])
                curr_bbox = curr_entry.get('bbox', [])
                
                # Validate bbox length
                if len(prev_bbox) < 4 or len(curr_bbox) < 4:
                    continue
                
                dist = self.calculate_distance(prev_bbox, curr_bbox)
                if not np.isinf(dist) and not np.isnan(dist):
                    total_distance += dist
                    valid_pairs += 1
            except (IndexError, KeyError, TypeError, ValueError):
                continue
        
        if valid_pairs == 0:
            return 0.0
        
        avg_speed = total_distance / valid_pairs
        return avg_speed if not np.isnan(avg_speed) and not np.isinf(avg_speed) else 0.0
    
    def is_in_intersection_zone(self, bbox: List[float], 
                                intersection_bbox: Optional[List[float]] = None) -> bool:
        """Check if object is in intersection zone"""
        # Validate bbox length
        if len(bbox) < 4:
            return False
        
        # If intersection bbox provided, check overlap
        if intersection_bbox and len(intersection_bbox) >= 4:
            try:
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                
                ix1, iy1, ix2, iy2 = intersection_bbox[:4]
                return ix1 <= cx <= ix2 and iy1 <= cy <= iy2
            except (IndexError, TypeError, ValueError):
                return False
        
        # Simple heuristic: assume intersection is in center-bottom of frame
        # This is a placeholder - should be replaced with actual intersection detection
        return False
    
    def extract_red_light_violation(self, 
                                    vehicle_track_id: int,
                                    traffic_light_tracks: List[Dict],
                                    current_frame: int) -> bool:
        """Check if vehicle violated red light"""
        # Find traffic light in intersection zone
        red_light_detected = False
        red_light_frame = None
        
        for tl in traffic_light_tracks:
            if tl.get('traffic_light_state') == 'red':
                red_light_detected = True
                tl_id = tl.get('track_id')
                if tl_id in self.red_light_frames:
                    red_light_frame = self.red_light_frames[tl_id]
                else:
                    self.red_light_frames[tl_id] = current_frame
                    red_light_frame = current_frame
                break
        
        if not red_light_detected:
            return False
        
        # Check if vehicle crossed intersection after red light
        if red_light_frame is None:
            return False
        
        frames_since_red = current_frame - red_light_frame
        if frames_since_red < self.red_light_violation_threshold:
            return False
        
        # Check if vehicle is in intersection zone
        # This is simplified - should check actual vehicle position
        return True
    
    def extract_pedestrian_crossing_violation(self,
                                             vehicle_bbox: List[float],
                                             pedestrian_tracks: List[Dict],
                                             proximity_threshold: Optional[float] = None) -> bool:
        """Check if vehicle failed to yield to pedestrian"""
        if proximity_threshold is None:
            proximity_threshold = self.proximity_threshold
        
        for ped in pedestrian_tracks:
            ped_bbox = ped.get('bbox', [])
            if not ped_bbox:
                continue
            
            distance = self.calculate_distance(vehicle_bbox, ped_bbox)
            
            # Check if pedestrian is in crosswalk and vehicle is too close
            if distance < proximity_threshold:
                # Additional check: pedestrian should be in front of vehicle
                try:
                    if len(vehicle_bbox) >= 4 and len(ped_bbox) >= 4:
                        vehicle_cy = (vehicle_bbox[1] + vehicle_bbox[3]) / 2
                        ped_cy = (ped_bbox[1] + ped_bbox[3]) / 2
                        
                        # Pedestrian should be below vehicle center (in front)
                        if ped_cy > vehicle_cy:
                            return True
                except (IndexError, TypeError, ValueError):
                    pass
        
        return False
    
    def extract_collision_proximity(self,
                                   vehicle_bbox: List[float],
                                   pedestrian_tracks: List[Dict]) -> float:
        """Calculate minimum distance to pedestrian (collision proximity)"""
        min_distance = float('inf')
        
        for ped in pedestrian_tracks:
            ped_bbox = ped.get('bbox', [])
            if not ped_bbox:
                continue
            
            distance = self.calculate_distance(vehicle_bbox, ped_bbox)
            min_distance = min(min_distance, distance)
        
        return min_distance if min_distance != float('inf') else 1000.0
    
    def extract_time_to_collision(self,
                                  vehicle_bbox: List[float],
                                  vehicle_speed: float,
                                  pedestrian_bbox: List[float],
                                  pedestrian_speed: float = 0.0) -> float:
        """Estimate time to collision"""
        distance = self.calculate_distance(vehicle_bbox, pedestrian_bbox)
        
        relative_speed = vehicle_speed - pedestrian_speed
        if relative_speed <= 0:
            return float('inf')
        
        ttc = distance / (relative_speed + 1e-6)
        return ttc
    
    def extract_features(self,
                        frame: np.ndarray,
                        vehicle_tracks: List[Dict],
                        pedestrian_tracks: List[Dict],
                        traffic_light_tracks: List[Dict],
                        tracker,
                        current_frame: int,
                        fps: float = 30.0) -> List[Dict]:
        """
        Extract features for all vehicles
        
        Args:
            frame: Current frame
            vehicle_tracks: List of tracked vehicles
            pedestrian_tracks: List of tracked pedestrians
            traffic_light_tracks: List of tracked traffic lights
            tracker: MultiObjectTracker instance
            current_frame: Current frame number
            fps: Frames per second
            
        Returns:
            List of feature dictionaries, one per vehicle
        """
        features_list = []
        
        for vehicle in vehicle_tracks:
            vehicle_track_id = vehicle.get('track_id')
            vehicle_bbox = vehicle.get('bbox', [])
            
            # Validate vehicle data
            if not vehicle_track_id:
                continue
            if not vehicle_bbox or len(vehicle_bbox) < 4:
                continue
            
            # Get track history
            track_history = tracker.get_track_history(vehicle_track_id)
            
            # Calculate speed
            vehicle_speed = self.calculate_speed(track_history, fps)
            
            # Extract features
            red_light_violation = self.extract_red_light_violation(
                vehicle_track_id, traffic_light_tracks, current_frame
            )
            
            pedestrian_crossing_violation = self.extract_pedestrian_crossing_violation(
                vehicle_bbox, pedestrian_tracks
            )
            
            collision_proximity = self.extract_collision_proximity(
                vehicle_bbox, pedestrian_tracks
            )
            
            # Calculate time to collision with nearest pedestrian
            min_ttc = float('inf')
            for ped in pedestrian_tracks:
                ped_bbox = ped.get('bbox', [])
                if not ped_bbox:
                    continue
                ttc = self.extract_time_to_collision(
                    vehicle_bbox, vehicle_speed, ped_bbox, 0.0
                )
                min_ttc = min(min_ttc, ttc)
            
            # Traffic light state duration
            traffic_light_state_duration = 0
            for tl in traffic_light_tracks:
                if tl.get('traffic_light_state') == 'red':
                    tl_id = tl.get('track_id')
                    if tl_id in self.red_light_frames:
                        traffic_light_state_duration = current_frame - self.red_light_frames[tl_id]
                    break
            
            # Check if in intersection zone
            in_intersection_zone = self.is_in_intersection_zone(vehicle_bbox)
            
            # Compile features
            features = {
                'vehicle_track_id': vehicle_track_id,
                'red_light_violation': int(red_light_violation),
                'pedestrian_crossing_violation': int(pedestrian_crossing_violation),
                'collision_proximity': collision_proximity,
                'vehicle_speed': vehicle_speed,
                'time_to_collision': min_ttc if min_ttc != float('inf') else 1000.0,
                'intersection_zone': int(in_intersection_zone),
                'traffic_light_state_duration': traffic_light_state_duration,
                'frame': current_frame
            }
            
            features_list.append(features)
        
        return features_list


