"""Multi-object tracking using Kalman filter and Hungarian algorithm"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import cv2
from scipy.optimize import linear_sum_assignment


class KalmanTracker:
    """Simple Kalman filter tracker for object tracking"""
    
    def __init__(self, bbox: List[float], track_id: int):
        """
        Initialize tracker with bounding box
        
        Args:
            bbox: [x1, y1, x2, y2]
            track_id: Unique track ID
        """
        # Validate bbox
        if not bbox or len(bbox) < 4:
            raise ValueError(f"Invalid bbox: must have 4 elements, got {len(bbox) if bbox else 0}")
        
        # Ensure bbox is valid
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ValueError(f"Invalid bbox: x2 must be > x1 and y2 must be > y1")
        
        self.track_id = track_id
        self.hits = 1
        self.time_since_update = 0
        self.age = 0
        
        # Initialize state: [cx, cy, s, r, vx, vy, vs, vr]
        # cx, cy: center, s: scale (area), r: aspect ratio
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        s = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        r = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1] + 1e-6)
        
        self.state = np.array([cx, cy, s, r, 0, 0, 0, 0], dtype=np.float32)
        
        # Simple constant velocity model
        self.P = np.eye(8) * 1000  # Covariance matrix
        self.Q = np.eye(8) * 0.03  # Process noise
        self.R = np.eye(4) * 1.0   # Measurement noise
    
    def predict(self):
        """Predict next state"""
        # Check for invalid state before prediction
        if np.any(np.isnan(self.state)) or np.any(np.isinf(self.state)):
            # Reset state if invalid
            self.state = np.array([0, 0, 100, 1, 0, 0, 0, 0], dtype=np.float32)
            self.P = np.eye(8) * 1000
        
        # Constant velocity model
        F = np.eye(8)
        F[0, 4] = 1  # cx += vx
        F[1, 5] = 1  # cy += vy
        F[2, 6] = 1  # s += vs
        F[3, 7] = 1  # r += vr
        
        try:
            self.state = F @ self.state
            self.P = F @ self.P @ F.T + self.Q
            
            # Clamp state values
            self.state = np.clip(self.state, -1e6, 1e6)
            
            # Ensure covariance matrix is valid
            self.P = (self.P + self.P.T) / 2  # Make symmetric
            self.P = np.clip(self.P, -1e4, 1e4)
            
            # Check for invalid values after prediction
            if np.any(np.isnan(self.state)) or np.any(np.isinf(self.state)):
                self.state = np.array([0, 0, 100, 1, 0, 0, 0, 0], dtype=np.float32)
            
            if np.any(np.isnan(self.P)) or np.any(np.isinf(self.P)):
                self.P = np.eye(8) * 1000
            
            self.age += 1
        except (ValueError, np.linalg.LinAlgError):
            # Reset on error
            self.state = np.array([0, 0, 100, 1, 0, 0, 0, 0], dtype=np.float32)
            self.P = np.eye(8) * 1000
    
    def update(self, bbox: List[float]):
        """Update tracker with new detection"""
        # Validate bbox
        if len(bbox) < 4:
            return
        
        # Ensure bbox is valid
        bbox = [float(b) for b in bbox]
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            return
        
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        s = max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 1.0)  # Ensure positive
        r = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1] + 1e-6)
        
        # Clamp values to prevent extreme values
        r = np.clip(r, 0.1, 10.0)
        s = np.clip(s, 1.0, 1e6)
        
        z = np.array([cx, cy, s, r], dtype=np.float32)
        
        # Check for invalid values in state
        if np.any(np.isnan(self.state)) or np.any(np.isinf(self.state)):
            # Reset state if invalid
            self.state = np.array([cx, cy, s, r, 0, 0, 0, 0], dtype=np.float32)
            self.P = np.eye(8) * 1000
            return
        
        # Measurement matrix
        H = np.zeros((4, 8))
        H[0, 0] = 1
        H[1, 1] = 1
        H[2, 2] = 1
        H[3, 3] = 1
        
        # Kalman update
        try:
            y = z - H @ self.state  # Innovation
            S = H @ self.P @ H.T + self.R  # Innovation covariance
            
            # Check for invalid values in S
            if np.any(np.isnan(S)) or np.any(np.isinf(S)):
                # Reset covariance if invalid
                self.P = np.eye(8) * 1000
                return
            
            # Use pseudo-inverse for numerical stability
            # Check determinant to avoid singular matrix
            det_S = np.linalg.det(S)
            if abs(det_S) < 1e-6:
                # Matrix is singular, use pseudo-inverse
                K = self.P @ H.T @ np.linalg.pinv(S)
            else:
                # Regular inverse
                K = self.P @ H.T @ np.linalg.inv(S)
            
            # Check Kalman gain for invalid values
            if np.any(np.isnan(K)) or np.any(np.isinf(K)):
                return
            
            self.state = self.state + K @ y
            self.P = (np.eye(8) - K @ H) @ self.P
            
            # Clamp state values to prevent extreme values
            self.state = np.clip(self.state, -1e6, 1e6)
            
            # Ensure covariance matrix is positive definite and has no invalid values
            self.P = (self.P + self.P.T) / 2  # Make symmetric
            self.P = np.clip(self.P, -1e4, 1e4)  # Clamp values
            
            # Check for NaN/Inf in final state and covariance
            if np.any(np.isnan(self.state)) or np.any(np.isinf(self.state)):
                self.state = np.array([cx, cy, s, r, 0, 0, 0, 0], dtype=np.float32)
            
            if np.any(np.isnan(self.P)) or np.any(np.isinf(self.P)):
                self.P = np.eye(8) * 1000
            
            self.hits += 1
            self.time_since_update = 0
        except (np.linalg.LinAlgError, ValueError) as e:
            # If Kalman update fails, reset tracker
            self.state = np.array([cx, cy, s, r, 0, 0, 0, 0], dtype=np.float32)
            self.P = np.eye(8) * 1000
    
    def get_state(self) -> List[float]:
        """Get current bounding box estimate"""
        # Check for invalid state
        if np.any(np.isnan(self.state)) or np.any(np.isinf(self.state)):
            # Return a default bbox if state is invalid
            return [0.0, 0.0, 100.0, 100.0]
        
        cx, cy, s, r = self.state[0], self.state[1], self.state[2], self.state[3]
        
        # Clamp values to reasonable ranges
        cx = np.clip(cx, -1e4, 1e4)
        cy = np.clip(cy, -1e4, 1e4)
        s = np.clip(s, 1.0, 1e6)
        r = np.clip(r, 0.1, 10.0)
        
        # Calculate width and height safely
        # s = width * height, r = width / height
        # So: width = sqrt(s * r), height = sqrt(s / r)
        try:
            w = np.sqrt(max(s * r, 0.1))  # Ensure positive
            h = np.sqrt(max(s / r, 0.1))  # Ensure positive
            
            # Validate w and h
            if np.isnan(w) or np.isinf(w) or np.isnan(h) or np.isinf(h):
                return [0.0, 0.0, 100.0, 100.0]
            
            # Ensure minimum size
            w = max(w, 10.0)
            h = max(h, 10.0)
            
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            
            # Validate bbox
            if x2 <= x1 or y2 <= y1:
                return [0.0, 0.0, 100.0, 100.0]
            
            # Check for NaN/Inf in final bbox
            bbox = [float(x1), float(y1), float(x2), float(y2)]
            for v in bbox:
                if np.isnan(v) or np.isinf(v):
                    return [0.0, 0.0, 100.0, 100.0]
            
            return bbox
        except (ValueError, ZeroDivisionError):
            return [0.0, 0.0, 100.0, 100.0]
    
    def get_velocity(self) -> Tuple[float, float]:
        """Get velocity estimate"""
        return (self.state[4], self.state[5])


class MultiObjectTracker:
    """Multi-object tracker using Kalman filters and Hungarian algorithm"""
    
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        Initialize multi-object tracker
        
        Args:
            max_age: Maximum frames to keep unmatched track
            min_hits: Minimum hits to confirm track
            iou_threshold: IoU threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.trackers: List[KalmanTracker] = []
        self.frame_count = 0
        self.next_id = 1
        
        # Track history for feature extraction
        self.track_history: Dict[int, List[Dict]] = defaultdict(list)
    
    def iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / (union_area + 1e-6)
    
    def associate_detections_to_trackers(self, detections: List[Dict], 
                                        trackers: List[KalmanTracker]) -> Tuple[np.ndarray, List[int], List[int]]:
        """
        Associate detections to trackers using Hungarian algorithm
        
        Returns:
            matches: Array of (detection_idx, tracker_idx) pairs
            unmatched_dets: List of unmatched detection indices
            unmatched_trks: List of unmatched tracker indices
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), list(range(len(detections))), []
        
        # Build cost matrix
        cost_matrix = np.zeros((len(detections), len(trackers)))
        
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                try:
                    trk_bbox = trk.get_state()
                    # Validate bbox values
                    bbox_valid = True
                    for v in trk_bbox:
                        if np.isnan(v) or np.isinf(v):
                            bbox_valid = False
                            break
                    
                    if not bbox_valid:
                        cost_matrix[d, t] = 1.0  # Maximum cost for invalid bbox
                    else:
                        iou_val = self.iou(det['bbox'], trk_bbox)
                        # Validate IOU value
                        if np.isnan(iou_val) or np.isinf(iou_val):
                            cost_matrix[d, t] = 1.0
                        else:
                            cost_matrix[d, t] = 1 - iou_val
                except Exception:
                    # If anything goes wrong, set maximum cost
                    cost_matrix[d, t] = 1.0
        
        # Validate cost matrix before Hungarian algorithm
        # Replace any NaN or Inf values with maximum cost
        if np.any(np.isnan(cost_matrix)) or np.any(np.isinf(cost_matrix)):
            cost_matrix = np.nan_to_num(cost_matrix, nan=1.0, posinf=1.0, neginf=1.0)
        
        # Hungarian algorithm
        if cost_matrix.size > 0:
            try:
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
            except (ValueError, np.linalg.LinAlgError) as e:
                # If Hungarian algorithm fails, return no matches
                return np.empty((0, 2), dtype=int), list(range(len(detections))), list(range(len(trackers)))
            matches = []
            unmatched_dets = []
            unmatched_trks = []
            
            for d in range(len(detections)):
                if d not in row_ind:
                    unmatched_dets.append(d)
            
            for t in range(len(trackers)):
                if t not in col_ind:
                    unmatched_trks.append(t)
            
            for d, t in zip(row_ind, col_ind):
                if cost_matrix[d, t] < (1 - self.iou_threshold):
                    matches.append([d, t])
                else:
                    unmatched_dets.append(d)
                    unmatched_trks.append(t)
            
            return np.array(matches), unmatched_dets, unmatched_trks
        
        return np.empty((0, 2), dtype=int), list(range(len(detections))), list(range(len(trackers)))
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of detections from object detector
            
        Returns:
            List of tracked objects with track_id
        """
        self.frame_count += 1
        
        # Predict trackers
        for trk in self.trackers:
            trk.predict()
        
        # Associate detections to trackers
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(
            detections, self.trackers
        )
        
        # Update matched trackers
        for m in matched:
            det_idx, trk_idx = m
            self.trackers[trk_idx].update(detections[det_idx]['bbox'])
            detections[det_idx]['track_id'] = self.trackers[trk_idx].track_id
        
        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            try:
                bbox = detections[i].get('bbox', [])
                if not bbox or len(bbox) < 4:
                    continue  # Skip invalid detections
                trk = KalmanTracker(bbox, self.next_id)
                self.trackers.append(trk)
                detections[i]['track_id'] = self.next_id
                self.next_id += 1
            except (ValueError, IndexError, KeyError):
                # Skip invalid detections
                continue
        
        # Remove old trackers
        self.trackers = [trk for trk in self.trackers 
                        if trk.time_since_update < self.max_age]
        
        # Update track history
        tracked_objects = []
        for det in detections:
            if 'track_id' in det:
                track_id = det['track_id']
                # Store detection in history
                self.track_history[track_id].append({
                    'frame': self.frame_count,
                    'bbox': det['bbox'],
                    'class_id': det['class_id'],
                    'class_name': det['class_name'],
                    'confidence': det['confidence']
                })
                
                # Keep only recent history
                if len(self.track_history[track_id]) > 100:
                    self.track_history[track_id] = self.track_history[track_id][-100:]
                
                tracked_objects.append(det)
        
        # Mark unmatched trackers
        for i in unmatched_trks:
            self.trackers[i].time_since_update += 1
        
        return tracked_objects
    
    def get_track_history(self, track_id: int) -> List[Dict]:
        """Get history for a specific track"""
        return self.track_history.get(track_id, [])
    
    def get_track_velocity(self, track_id: int, window: int = 10) -> Optional[Tuple[float, float]]:
        """Get velocity estimate for a track"""
        history = self.get_track_history(track_id)
        if len(history) < 2:
            return None
        
        recent = history[-window:]
        if len(recent) < 2:
            return None
        
        try:
            # Calculate velocity from recent positions
            first = recent[0]
            last = recent[-1]
            
            # Validate first and last entries
            if 'bbox' not in first or 'bbox' not in last:
                return None
            if 'frame' not in first or 'frame' not in last:
                return None
            
            first_bbox = first.get('bbox', [])
            last_bbox = last.get('bbox', [])
            
            # Validate bbox length
            if len(first_bbox) < 4 or len(last_bbox) < 4:
                return None
            
            frames_diff = last['frame'] - first['frame']
            if frames_diff == 0:
                return None
            
            cx1 = (first_bbox[0] + first_bbox[2]) / 2
            cy1 = (first_bbox[1] + first_bbox[3]) / 2
            cx2 = (last_bbox[0] + last_bbox[2]) / 2
            cy2 = (last_bbox[1] + last_bbox[3]) / 2
            
            vx = (cx2 - cx1) / frames_diff
            vy = (cy2 - cy1) / frames_diff
            
            # Validate velocity values
            if np.isnan(vx) or np.isinf(vx) or np.isnan(vy) or np.isinf(vy):
                return None
            
            return (vx, vy)
        except (IndexError, KeyError, TypeError, ValueError, ZeroDivisionError):
            return None


