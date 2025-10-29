import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
from dataclasses import dataclass

from pose_estimation.track.utils import CameraParameters, SpringDamperConfig

class Projection3D:
    def __init__(self, camera_params: List[CameraParameters]):
        self.cameras = camera_params
        self.device = camera_params[0].K.device
        self.projection_matrices = {}
        self.view_triangulation_cache = {}
        
        for cam in camera_params:
            RT = torch.cat([cam.R, cam.T.unsqueeze(-1)], dim=-1)
            P = cam.K @ RT
            self.projection_matrices[cam.view_idx] = P.to(dtype=torch.float32)
    
    def world_to_camera(self, point_3d: torch.Tensor, view_idx: int) -> Optional[torch.Tensor]:
        if view_idx not in self.projection_matrices:
            return None
        
        P = self.projection_matrices[view_idx]
        point_3d = point_3d.to(dtype=torch.float32, device=self.device)
        point_h = torch.cat([point_3d, torch.ones(1, device=self.device, dtype=torch.float32)])
        
        projected = P @ point_h
        if projected[2] > 0.1:
            return projected[:2] / projected[2]
        return None
    
    def triangulate_from_views(self, detections: Dict[int, np.ndarray]) -> Optional[torch.Tensor]:
        if len(detections) < 2:
            return None
        
        cache_key = tuple(sorted(detections.keys()))
        if cache_key in self.view_triangulation_cache:
            cached_matrices = self.view_triangulation_cache[cache_key]
        else:
            A = []
            for view_idx, _ in detections.items():
                if view_idx < len(self.cameras):
                    P = self.projection_matrices[view_idx].cpu().numpy()
                    A.append(P)
            self.view_triangulation_cache[cache_key] = A
            cached_matrices = A
        
        A_equations = []
        for i, (view_idx, point_2d) in enumerate(detections.items()):
            if i < len(cached_matrices):
                P = cached_matrices[i]
                x, y = point_2d
                A_equations.append(x * P[2, :] - P[0, :])
                A_equations.append(y * P[2, :] - P[1, :])
        
        if len(A_equations) < 4:
            return None
        
        A_matrix = np.array(A_equations)
        _, _, Vt = np.linalg.svd(A_matrix)
        X = Vt[-1, :]
        
        if abs(X[3]) > 1e-8:
            point_3d = X[:3] / X[3]
            
            if (abs(point_3d[0]) < 15 and abs(point_3d[1]) < 15 and 
                0.05 < point_3d[2] < 12.0):
                
                reprojection_error = self._compute_reprojection_error(point_3d, detections)
                if reprojection_error < 80.0:
                    return torch.tensor(point_3d, device=self.device, dtype=torch.float32)
        
        return None
    
    def triangulate_multi_view_refined(self, detections: Dict[int, np.ndarray], 
                                     confidence_weights: Dict[int, float] = None) -> Optional[torch.Tensor]:
        if len(detections) < 2:
            return None
        
        weights = confidence_weights or {k: 1.0 for k in detections.keys()}
        
        A = []
        for view_idx, point_2d in detections.items():
            if view_idx < len(self.cameras):
                P = self.projection_matrices[view_idx].cpu().numpy()
                weight = weights.get(view_idx, 1.0)
                x, y = point_2d
                A.append(weight * (x * P[2, :] - P[0, :]))
                A.append(weight * (y * P[2, :] - P[1, :]))
        
        if len(A) < 4:
            return None
        
        A = np.array(A)
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1, :]
        
        if abs(X[3]) > 1e-8:
            point_3d = X[:3] / X[3]
            
            if (abs(point_3d[0]) < 15 and abs(point_3d[1]) < 15 and 
                0.05 < point_3d[2] < 12.0):
                
                reprojection_error = self._compute_reprojection_error(point_3d, detections)
                if reprojection_error < 60.0:
                    return torch.tensor(point_3d, device=self.device, dtype=torch.float32)
        
        return None
    
    def _compute_reprojection_error(self, point_3d: np.ndarray, detections: Dict[int, np.ndarray]) -> float:
        total_error = 0.0
        valid_projections = 0
        
        point_3d_tensor = torch.tensor(point_3d, device=self.device, dtype=torch.float32)
        
        for view_idx, original_2d in detections.items():
            projected = self.world_to_camera(point_3d_tensor, view_idx)
            if projected is not None:
                error = np.linalg.norm(projected.cpu().numpy() - original_2d)
                total_error += error
                valid_projections += 1
        
        return total_error / max(valid_projections, 1)
    
    def project_3d_bbox(self, center_3d: torch.Tensor, size_3d: torch.Tensor, view_idx: int) -> Optional[Tuple[int, int, int, int]]:
        center_3d = center_3d.to(dtype=torch.float32, device=self.device)
        size_3d = size_3d.to(dtype=torch.float32, device=self.device)
        
        half_size = size_3d / 2
        corners_3d = torch.stack([
            center_3d + torch.tensor([-1, -1, -1], device=self.device, dtype=torch.float32) * half_size,
            center_3d + torch.tensor([1, -1, -1], device=self.device, dtype=torch.float32) * half_size,
            center_3d + torch.tensor([-1, 1, -1], device=self.device, dtype=torch.float32) * half_size,
            center_3d + torch.tensor([1, 1, -1], device=self.device, dtype=torch.float32) * half_size,
            center_3d + torch.tensor([-1, -1, 1], device=self.device, dtype=torch.float32) * half_size,
            center_3d + torch.tensor([1, -1, 1], device=self.device, dtype=torch.float32) * half_size,
            center_3d + torch.tensor([-1, 1, 1], device=self.device, dtype=torch.float32) * half_size,
            center_3d + torch.tensor([1, 1, 1], device=self.device, dtype=torch.float32) * half_size,
        ])
        
        projected_corners = []
        for corner in corners_3d:
            proj = self.world_to_camera(corner, view_idx)
            if proj is not None:
                projected_corners.append(proj.cpu().numpy())
        
        if len(projected_corners) >= 4:
            projected_corners = np.array(projected_corners)
            x1 = int(np.min(projected_corners[:, 0]))
            y1 = int(np.min(projected_corners[:, 1]))
            x2 = int(np.max(projected_corners[:, 0]))
            y2 = int(np.max(projected_corners[:, 1]))
            return (x1, y1, x2, y2)
        
        return None

class SpringDamperPhysics:
    def __init__(self, config: SpringDamperConfig, projection: Projection3D):
        self.config = config
        self.projection = projection
        self.device = projection.device
        self.position_history = {}
        self.velocity_smoothing = deque(maxlen=5)
        self.position_confidence = {}
    
    def update_track_physics(self, track, detections: Dict[int, Tuple], 
                           confidences: Dict[int, float], dt: float = 1/60.0):
        
        track.position_3d = track.position_3d.to(dtype=torch.float32, device=self.device)
        track.velocity_3d = track.velocity_3d.to(dtype=torch.float32, device=self.device)
        track.size_3d = track.size_3d.to(dtype=torch.float32, device=self.device)
        
        filtered_detections = self._filter_detections_enhanced(detections, confidences)
        
        if filtered_detections:
            centers_2d = {}
            confidence_weights = {}
            valid_detections = {}
            
            for view_idx, bbox in filtered_detections.items():
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                current_center = np.array([center_x, center_y])
                
                is_valid = self._validate_detection_enhanced(track, current_center, view_idx)
                
                if is_valid:
                    centers_2d[view_idx] = current_center
                    confidence_weights[view_idx] = confidences.get(view_idx, 0.5)
                    valid_detections[view_idx] = bbox
            
            if len(centers_2d) >= 2:
                new_position = self.projection.triangulate_multi_view_refined(centers_2d, confidence_weights)
                
                if new_position is not None:
                    new_position = new_position.to(dtype=torch.float32, device=self.device)
                    
                    displacement_3d = torch.norm(new_position - track.position_3d)
                    adaptive_threshold = self._compute_adaptive_threshold(track, len(centers_2d))
                    
                    if displacement_3d < adaptive_threshold:
                        new_velocity = (new_position - track.position_3d) / dt
                        velocity_blend = self._compute_velocity_blend_factor(track, len(centers_2d))
                        track.velocity_3d = (1 - velocity_blend) * track.velocity_3d + velocity_blend * new_velocity
                        
                        position_smoothing = self._compute_position_smoothing(track, displacement_3d)
                        track.position_3d = (1 - position_smoothing) * track.position_3d + position_smoothing * new_position
                        
                        if valid_detections:
                            self._update_size_from_detections_enhanced(track, valid_detections, confidence_weights)
                        
                        self.position_confidence[track.id] = min(1.0, 
                            self.position_confidence.get(track.id, 0.5) + 0.1)
                    else:
                        predicted_pos = track.position_3d + track.velocity_3d * dt
                        blend_factor = min(0.5, 0.1 + displacement_3d / adaptive_threshold * 0.3)
                        track.position_3d = (1 - blend_factor) * predicted_pos + blend_factor * new_position
                        track.velocity_3d *= 0.85
                        
                        self.position_confidence[track.id] = max(0.0, 
                            self.position_confidence.get(track.id, 0.5) - 0.1)
                else:
                    predicted_pos = track.position_3d + track.velocity_3d * dt
                    track.position_3d = predicted_pos
                    track.velocity_3d *= 0.92
                    self.position_confidence[track.id] = max(0.0, 
                        self.position_confidence.get(track.id, 0.5) - 0.05)
            else:
                predicted_pos = track.position_3d + track.velocity_3d * dt
                track.position_3d = predicted_pos
                track.velocity_3d *= 0.92
                self.position_confidence[track.id] = max(0.0, 
                    self.position_confidence.get(track.id, 0.5) - 0.05)
        else:
            predicted_pos = track.position_3d + track.velocity_3d * dt
            track.position_3d = predicted_pos
            track.velocity_3d *= 0.95
            self.position_confidence[track.id] = max(0.0, 
                self.position_confidence.get(track.id, 0.5) - 0.08)
    
    def _filter_detections_enhanced(self, detections: Dict[int, Tuple], 
                                  confidences: Dict[int, float]) -> Dict[int, Tuple]:
        filtered = {}
        
        for view_idx, bbox in detections.items():
            confidence = confidences.get(view_idx, 0.0)
            if confidence < 0.2:
                continue
                
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            if abs(center_x) < 5 and abs(center_y) < 5:
                continue
            
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            if width <= 0 or height <= 0:
                continue
            
            aspect_ratio = height / width
            if not (0.8 < aspect_ratio < 4.0):
                continue
                
            filtered[view_idx] = bbox
        
        return filtered
    
    def _validate_detection_enhanced(self, track, current_center: np.ndarray, view_idx: int) -> bool:
        if track.bboxes.get(view_idx) is not None:
            prev_center = np.array([
                (track.bboxes[view_idx][0] + track.bboxes[view_idx][2]) / 2,
                (track.bboxes[view_idx][1] + track.bboxes[view_idx][3]) / 2
            ])
            
            jump_distance = np.linalg.norm(current_center - prev_center)
            confidence = self.position_confidence.get(track.id, 0.5)
            max_jump_distance = 200.0 - confidence * 50.0
            
            projected_center = self.projection.world_to_camera(track.position_3d, view_idx)
            if projected_center is not None:
                proj_center_np = projected_center.cpu().numpy()
                cube_distance = np.linalg.norm(current_center - proj_center_np)
                max_cube_distance = 250.0 - confidence * 50.0
                
                if jump_distance > max_jump_distance and cube_distance > max_cube_distance:
                    return False
        
        return True
    
    def _compute_adaptive_threshold(self, track, num_views: int) -> float:
        base_threshold = 0.15 if track.time_since_update < 3 else 0.30
        confidence = self.position_confidence.get(track.id, 0.5)
        view_factor = min(1.2, 1.0 + (num_views - 2) * 0.1)
        confidence_factor = 1.0 + (1.0 - confidence) * 0.5
        return base_threshold * view_factor * confidence_factor
    
    def _compute_velocity_blend_factor(self, track, num_views: int) -> float:
        base_blend = 0.8
        confidence = self.position_confidence.get(track.id, 0.5)
        if hasattr(track, 'confidence') and track.confidence > 0.7:
            base_blend = 0.9
        view_factor = min(1.1, 1.0 + (num_views - 2) * 0.05)
        return min(0.95, base_blend * view_factor * (0.5 + confidence * 0.5))
    
    def _compute_position_smoothing(self, track, displacement: float) -> float:
        base_smoothing = 0.1 if hasattr(track, 'confidence') and track.confidence > 0.7 else 0.2
        confidence = self.position_confidence.get(track.id, 0.5)
        displacement_factor = min(2.0, 1.0 + displacement * 2.0)
        return min(0.8, base_smoothing * displacement_factor / (confidence + 0.1))
    
    def _update_size_from_detections_enhanced(self, track, detections: Dict[int, Tuple], 
                                            confidence_weights: Dict[int, float]):
        depth = track.position_3d[2].item()
        
        width_estimates = []
        height_estimates = []
        weights = []
        
        for view_idx, bbox in detections.items():
            if view_idx >= len(self.projection.cameras):
                continue
                
            cam = self.projection.cameras[view_idx]
            fx = cam.K[0, 0].item()
            fy = cam.K[1, 1].item()
            
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            
            world_width = (bbox_width * depth) / fx
            world_height = (bbox_height * depth) / fy
            
            confidence = confidence_weights.get(view_idx, 0.5)
            
            width_estimates.append(world_width)
            height_estimates.append(world_height)
            weights.append(confidence)
        
        if width_estimates:
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            new_width = np.average(width_estimates, weights=weights)
            new_height = np.average(height_estimates, weights=weights)
            new_depth = new_width * 0.5
            
            new_width = np.clip(new_width, 0.3, 1.5)
            new_height = np.clip(new_height, 1.4, 2.5)
            new_depth = np.clip(new_depth, 0.2, 1.0)
            
            target_size = torch.tensor([new_width, new_depth, new_height], 
                                     device=self.device, dtype=torch.float32)
            
            size_confidence = np.mean(weights)
            size_smoothing = 0.8 - size_confidence * 0.2
            track.size_3d = size_smoothing * track.size_3d + (1 - size_smoothing) * target_size