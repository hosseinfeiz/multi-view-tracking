import cv2
import numpy as np
from typing import List, Tuple, Union
from pose_estimation.track.utils import Visualization, PerformanceMonitor
from pose_estimation.track.geometry_3d import Projection3D

class VisualizationEngine(Visualization):
    def __init__(self, projection: Projection3D, target_width: int = 1920, target_height: int = 1080):
        super().__init__(target_width, target_height)
        self.projection = projection
        self.xmem_colors = [
            (0, 255, 0),    # Bright green for XMem-supported tracks
            (255, 255, 0),  # Yellow for mixed support
            (255, 0, 0),    # Red for detection-only tracks
        ]
    
    def draw_tracks(self, mosaic: np.ndarray, tracks: List, scale: float) -> np.ndarray:
        mosaic_h, mosaic_w = mosaic.shape[:2]
        h2, w2 = mosaic_h // 2, mosaic_w // 2
        view_offsets = [(0, 0), (w2, 0), (0, h2), (w2, h2)]
        
        for track in tracks:
            if hasattr(track, 'is_confirmed') and not track.is_confirmed():
                continue
            elif hasattr(track, 'is_active') and not track.is_active():
                continue
            elif hasattr(track, 'confidence') and track.confidence < 0.3:
                continue
            
            if hasattr(track, 'is_xmem_supported') and track.is_xmem_supported():
                base_color = self.colors[track.id % len(self.colors)]
                color = tuple(min(255, int(c * 1.2)) for c in base_color)
                border_color = (0, 255, 0)
                thickness_multiplier = 1.5
            else:
                base_color = self.colors[track.id % len(self.colors)]
                color = tuple(int(c * 0.7) for c in base_color)
                border_color = (255, 100, 100)
                thickness_multiplier = 1.0
            
            if hasattr(track, 'bboxes') and isinstance(track.bboxes, dict):
                track_bboxes = track.bboxes
                num_views = 4
            elif hasattr(track, 'bboxes') and isinstance(track.bboxes, list):
                track_bboxes = {i: bbox for i, bbox in enumerate(track.bboxes) if bbox is not None}
                num_views = len(track.bboxes)
            else:
                continue
            
            for view_idx in range(min(num_views, 4)):
                dx, dy = view_offsets[view_idx]
                
                if hasattr(track, 'position_3d') and hasattr(track, 'size_3d'):
                    projected_bbox = self.projection.project_3d_bbox(track.position_3d, track.size_3d, view_idx)
                    if projected_bbox is not None:
                        x1 = int(projected_bbox[0] * scale) + dx
                        y1 = int(projected_bbox[1] * scale) + dy
                        x2 = int(projected_bbox[2] * scale) + dx
                        y2 = int(projected_bbox[3] * scale) + dy
                        
                        x1, y1 = max(dx, x1), max(dy, y1)
                        x2, y2 = min(dx + w2, x2), min(dy + h2, y2)
                        
                        if hasattr(track, 'is_xmem_supported') and track.is_xmem_supported():
                            dash_color = (0, 200, 0)
                            dash_length = 12
                        else:
                            dash_color = (200, 100, 100)
                            dash_length = 6
                        
                        self._draw_dashed_rectangle(mosaic, (x1, y1, x2, y2), dash_color, 2, dash_length)
                        
                        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                        cv2.circle(mosaic, (center_x, center_y), 4, dash_color, -1)
                
                if view_idx in track_bboxes and track_bboxes[view_idx] is not None:
                    bbox = track_bboxes[view_idx]
                    
                    is_visible = True
                    if hasattr(track, 'occlusion_states') and view_idx < len(track.occlusion_states):
                        is_visible = track.occlusion_states[view_idx] == 'VISIBLE'
                    
                    if is_visible:
                        x1 = int(bbox[0] * scale) + dx
                        y1 = int(bbox[1] * scale) + dy
                        x2 = int(bbox[2] * scale) + dx
                        y2 = int(bbox[3] * scale) + dy
                        
                        x1, y1 = max(dx, x1), max(dy, y1)
                        x2, y2 = min(dx + w2, x2), min(dy + h2, y2)
                        
                        base_thickness = 3
                        if hasattr(track, 'confidence_scores') and view_idx < len(track.confidence_scores):
                            conf = track.confidence_scores[view_idx]
                            thickness = max(1, int(base_thickness * conf * thickness_multiplier))
                        elif hasattr(track, 'confidence'):
                            thickness = max(1, int(base_thickness * track.confidence * thickness_multiplier))
                        else:
                            thickness = base_thickness
                        
                        cv2.rectangle(mosaic, (x1, y1), (x2, y2), color, thickness)
                        
                        if hasattr(track, 'is_xmem_supported') and track.is_xmem_supported():
                            cv2.rectangle(mosaic, (x1-2, y1-2), (x2+2, y2+2), border_color, 2)
                        
                        xmem_conf = 0.0
                        if hasattr(track, 'xmem_confidence'):
                            xmem_conf = track.xmem_confidence
                        
                        if hasattr(track, 'is_xmem_supported') and track.is_xmem_supported():
                            bar_width = int(30 * xmem_conf)
                            bar_y = y2 + 3
                            cv2.rectangle(mosaic, (x1, bar_y), (x1 + bar_width, bar_y + 4), (0, 255, 0), -1)
                            cv2.rectangle(mosaic, (x1 + bar_width, bar_y), (x1 + 30, bar_y + 4), (100, 100, 100), -1)
                            
                            if hasattr(track, 'mask_consistency_scores') and len(track.mask_consistency_scores) > 0:
                                stability = np.std(list(track.mask_consistency_scores))
                                stability_normalized = max(0, 1 - stability)
                                stability_color = (0, 255, 0) if stability_normalized > 0.8 else (255, 255, 0) if stability_normalized > 0.5 else (255, 0, 0)
                                cv2.circle(mosaic, (x2 - 8, y1 + 8), 4, stability_color, -1)
                        
                        elif hasattr(track, 'detection_confidence'):
                            det_conf = track.detection_confidence
                            bar_width = int(30 * det_conf)
                            bar_y = y2 + 3
                            cv2.rectangle(mosaic, (x1, bar_y), (x1 + bar_width, bar_y + 4), (0, 165, 255), -1)
                            cv2.rectangle(mosaic, (x1 + bar_width, bar_y), (x1 + 30, bar_y + 4), (100, 100, 100), -1)
        
        return mosaic
    
    def _draw_dashed_rectangle(self, img: np.ndarray, bbox: Tuple[int, int, int, int], 
                             color: Tuple[int, int, int], thickness: int, dash_length: int = 8):
        x1, y1, x2, y2 = bbox
        
        for x in range(x1, x2, dash_length * 2):
            cv2.line(img, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
            cv2.line(img, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
        
        for y in range(y1, y2, dash_length * 2):
            cv2.line(img, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
            cv2.line(img, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)
    
    def add_performance_info(self, mosaic: np.ndarray, performance_monitor: PerformanceMonitor, 
                           frame_idx: int, tracks: List) -> np.ndarray:
        return mosaic