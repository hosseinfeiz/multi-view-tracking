import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
import time
from pathlib import Path
from dataclasses import dataclass
import argparse
import json
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pose_estimation.track.yolo import YOLOInference, Results

import gc

@dataclass
class TrackingConfig:
    num_persons: int = 2
    num_cameras: int = 4
    detection_confidence: float = 0.4
    assignment_threshold: float = 0.25
    max_missing_frames: int = 60
    occlusion_threshold: float = 0.4
    reidentification_threshold: float = 0.2

@dataclass
class SpringDamperConfig:
    spring_constant: float = 300.0
    damping_coefficient: float = 5.0
    cube_size_base: float = 0.8
    force_threshold: float = 0.5
    max_displacement: float = 2.0
    convergence_epsilon: float = 0.001
    max_iterations: int = 10
    position_smoothing: float = 0.1

@dataclass
class CameraParameters:
    K: torch.Tensor
    R: torch.Tensor
    T: torch.Tensor
    D: torch.Tensor
    view_idx: int = 0
    
    def to_projection_matrix(self) -> torch.Tensor:
        RT = torch.cat([self.R, self.T.unsqueeze(1)], dim=1)
        P = self.K @ RT
        return P

class PerformanceMonitor:
    def __init__(self):
        self.frame_times = deque(maxlen=30)
        self.component_times = defaultdict(list)
        self.start_times = {}
        self.stream = torch.cuda.current_stream() if torch.cuda.is_available() else None
    
    def start_timer(self) -> float:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()
    
    def end_timer(self, start_time: float, component_name: str = None) -> float:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - start_time
        if component_name:
            self.component_times[component_name].append(elapsed)
        return elapsed
    
    def log_frame_time(self, frame_time: float):
        self.frame_times.append(frame_time)
    
    def get_average_fps(self) -> float:
        if not self.frame_times:
            return 0.0
        return 1.0 / (sum(self.frame_times) / len(self.frame_times))

class DetectionEngine:
    def __init__(self, model_path: str = 'yolov8m.pt', confidence: float = 0.4):
        self.model_path = model_path
        self.confidence = confidence
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model()
        self.stream = torch.cuda.current_stream() if torch.cuda.is_available() else None
        self.frame_count = 0
        self.target_size = 320
    
    def _load_model(self):
        self.model = YOLOInference(self.model_path, device=self.device, conf_thresh=self.confidence)
    
    def detect_persons_batch(self, frames: List[torch.Tensor]) -> List[List[Dict]]:
        all_detections = []
        
        with torch.cuda.device(self.device):
            for frame in frames:
                if frame.dim() == 4:
                    frame = frame.squeeze(0)
                if frame.dim() == 3 and frame.shape[0] == 3:
                    frame = frame.permute(1, 2, 0)
                
                img_np = (frame.cpu().numpy() * 255).astype(np.uint8)
                original_h, original_w = img_np.shape[:2]
                
                h, w = img_np.shape[:2]
                if max(h, w) > self.target_size:
                    scale = self.target_size / max(h, w)
                    new_h, new_w = int(h * scale), int(w * scale)
                    img_resized = cv2.resize(img_np, (new_w, new_h))
                    scale_back = True
                else:
                    img_resized = img_np
                    scale_back = False
                    scale = 1.0
                
                if img_resized.shape[2] == 3:
                    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
                
                detections = self.model.detect(img_resized, classes=[0])
                
                view_detections = []
                for det in detections:
                    bbox = det['bbox']
                    if scale_back:
                        x1, y1, x2, y2 = bbox
                        x1 = int(x1 / scale)
                        y1 = int(y1 / scale)
                        x2 = int(x2 / scale)
                        y2 = int(y2 / scale)
                        x1 = max(0, min(x1, original_w))
                        y1 = max(0, min(y1, original_h))
                        x2 = max(0, min(x2, original_w))
                        y2 = max(0, min(y2, original_h))
                        bbox = (x1, y1, x2, y2)
                    
                    view_detections.append({
                        'bbox': bbox,
                        'conf': det['conf']
                    })
                
                all_detections.append(view_detections)
        
        self.frame_count += 1
        if self.frame_count % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()
        
        return all_detections

class XMLOutputManager:
    def __init__(self, output_dir: str, camera_parameters: List[CameraParameters]):
        self.output_dir = Path(output_dir)
        self.cameras = camera_parameters
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.detections = defaultdict(list)
        self.buffer = defaultdict(list)
        self.last_save = 0
    
    def add_track_detections(self, tracks: List, frame_idx: int):
        for track in tracks:
            for view_idx in range(len(track.bboxes)):
                if (track.bboxes[view_idx] is not None and 
                    getattr(track, 'occlusion_states', ['VISIBLE'] * len(track.bboxes))[view_idx] == 'VISIBLE' and
                    getattr(track, 'missing_counts', [0] * len(track.bboxes))[view_idx] <= 5):
                    
                    key = (view_idx, frame_idx)
                    self.buffer[key] = [det for det in self.buffer[key] if det[0] != track.id]
                    self.buffer[key].append((track.id, track.bboxes[view_idx]))
        
        if frame_idx - self.last_save > 100:
            self._flush_buffer()
            self.last_save = frame_idx
    
    def _flush_buffer(self):
        for key, detections in self.buffer.items():
            self.detections[key] = detections
        self.buffer.clear()
    
    def save_xml_files(self):
        self._flush_buffer()
        for camera_idx in range(len(self.cameras)):
            self._save_camera_xml(camera_idx)
    
    def _save_camera_xml(self, camera_idx: int):
        cam_params = self.cameras[camera_idx]
        
        root = ET.Element('root')
        root.set('K', self._format_matrix(cam_params.K))
        root.set('R', self._format_matrix(cam_params.R))
        root.set('T', self._format_matrix(cam_params.T))
        root.set('D', self._format_matrix(cam_params.D))
        
        frames_dict = defaultdict(list)
        for (cam_idx, frame_idx), detections in self.detections.items():
            if cam_idx == camera_idx:
                frames_dict[frame_idx].extend(detections)
        
        for frame_idx in sorted(frames_dict.keys()):
            keyframe_elem = ET.SubElement(root, 'keyframe')
            keyframe_elem.set('key', str(frame_idx))
            
            detections = frames_dict[frame_idx]
            for person_id, bbox in detections:
                key_elem = ET.SubElement(keyframe_elem, 'key')
                key_elem.set('personID', str(person_id))
                
                x1, y1, x2, y2 = bbox
                bbox_str = f"{int(x1)} {int(y1)} {int(x2)} {int(y2)} 1.0"
                key_elem.set('bbox', bbox_str)
        
        self._save_xml_safely(root, self.output_dir / f'{camera_idx + 1}.xml')
    
    def _format_matrix(self, matrix: torch.Tensor) -> str:
        if isinstance(matrix, torch.Tensor):
            matrix = matrix.cpu().numpy()
        return ' '.join(map(str, matrix.flatten()))
    
    def _save_xml_safely(self, root: ET.Element, filename: Path):
        rough_string = ET.tostring(root, 'unicode')
        reparsed = minidom.parseString(rough_string)
        pretty_lines = reparsed.toprettyxml(indent="  ").split('\n')[1:]
        pretty_lines = [line for line in pretty_lines if line.strip()]
        xml_content = '\n'.join(pretty_lines)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(xml_content)

class Visualization:
    def __init__(self, target_width: int = 1280, target_height: int = 720):
        self.target_width = target_width
        self.target_height = target_height
        self.colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255)
        ]
        self.mosaic_buffer = None
        self.scale_cache = 1.0
    
    def create_mosaic(self, frames: List[torch.Tensor]) -> Tuple[np.ndarray, float]:
        if not frames or len(frames) == 0:
            return np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8), 1.0
        
        frame_arrays = []
        for frame in frames:
            with torch.cuda.device(frame.device):
                if frame.dim() == 4:
                    frame = frame.squeeze(0)
                if frame.dim() == 3 and frame.shape[0] == 3:
                    frame = frame.permute(1, 2, 0)
                
                img = (frame.cpu().numpy() * 255).astype(np.uint8)
                
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                frame_arrays.append(img)
        
        while len(frame_arrays) < 4:
            h, w = frame_arrays[0].shape[:2]
            black_frame = np.zeros((h, w, 3), dtype=np.uint8)
            frame_arrays.append(black_frame)
        
        h, w = frame_arrays[0].shape[:2]
        target_w = self.target_width // 2
        target_h = self.target_height // 2
        scale_w = target_w / w
        scale_h = target_h / h
        scale = min(scale_w, scale_h, 1.0)
        
        if abs(scale - self.scale_cache) > 0.01:
            self.scale_cache = scale
        
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized_frames = []
        for img in frame_arrays[:4]:
            resized = cv2.resize(img, (new_w, new_h))
            resized_frames.append(resized)
        
        top_row = np.hstack([resized_frames[0], resized_frames[1]])
        bottom_row = np.hstack([resized_frames[2], resized_frames[3]])
        mosaic = np.vstack([top_row, bottom_row])
        
        return mosaic, scale
    
    def draw_tracks(self, mosaic: np.ndarray, tracks: List, scale: float) -> np.ndarray:
        mosaic_h, mosaic_w = mosaic.shape[:2]
        h2, w2 = mosaic_h // 2, mosaic_w // 2
        view_offsets = [(0, 0), (w2, 0), (0, h2), (w2, h2)]
        
        for track in tracks:
            color = self.colors[track.id % len(self.colors)]
            
            for view_idx in range(min(len(track.bboxes), 4)):
                if track.bboxes[view_idx] is not None:
                    dx, dy = view_offsets[view_idx]
                    bbox = track.bboxes[view_idx]
                    
                    x1 = int(bbox[0] * scale) + dx
                    y1 = int(bbox[1] * scale) + dy
                    x2 = int(bbox[2] * scale) + dx
                    y2 = int(bbox[3] * scale) + dy
                    
                    x1, y1 = max(dx, x1), max(dy, y1)
                    x2, y2 = min(dx + w2, x2), min(dy + h2, y2)
                    
                    thickness = 3
                    if hasattr(track, 'occlusion_states') and view_idx < len(track.occlusion_states):
                        if track.occlusion_states[view_idx] == 'OCCLUDED':
                            thickness = 1
                        elif hasattr(track, 'missing_counts') and track.missing_counts[view_idx] > 5:
                            thickness = 2
                    
                    cv2.rectangle(mosaic, (x1, y1), (x2, y2), color, thickness)
                    
                    cv2.putText(mosaic, f"T{track.id}", (x1 + 2, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return mosaic
    
    def add_performance_info(self, mosaic: np.ndarray, frame_idx: int, num_tracks: int, fps: float) -> np.ndarray:
        info_lines = [
            f"Frame: {frame_idx}",
            f"FPS: {fps:.1f}",
            f"Active Tracks: {num_tracks}"
        ]
        
        y_offset = 30
        for info in info_lines:
            cv2.putText(mosaic, info, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30
        
        return mosaic

class AnnotationTool:
    def __init__(self, num_cameras: int = 4):
        self.num_cameras = num_cameras
        self.annotations = {}
        self.current_person_id = 1
        self.colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    def annotate_frames(self, video_paths: List[str], num_frames: int = 40) -> Dict[int, Dict[int, Dict[int, List]]]:
        caps = [cv2.VideoCapture(path) for path in video_paths]
        
        frame_annotations = {}
        
        for frame_idx in range(num_frames):
            frames = []
            for cap in caps:
                ret, frame = cap.read()
                if ret:
                    h, w = frame.shape[:2]
                    if max(h, w) > 640:
                        scale = 640 / max(h, w)
                        new_h, new_w = int(h * scale), int(w * scale)
                        frame = cv2.resize(frame, (new_w, new_h))
                    frames.append(frame)
                else:
                    break
            
            if len(frames) != len(video_paths):
                break
            
            frame_annotations[frame_idx] = self._annotate_single_frame(frames, frame_idx)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()
        
        return frame_annotations
    
    def _annotate_single_frame(self, frames: List[np.ndarray], frame_idx: int) -> Dict[int, Dict[int, List]]:
        frame_data = {}
        
        mosaic = self._create_mosaic(frames)
        
        while True:
            display_mosaic = mosaic.copy()
            
            for person_id, views_data in frame_data.items():
                color = self.colors[person_id % len(self.colors)]
                for view_idx, bbox in views_data.items():
                    self._draw_bbox_on_mosaic(display_mosaic, bbox, view_idx, person_id, color, frames[0].shape)
            
            cv2.putText(display_mosaic, f"Frame {frame_idx} - Person {self.current_person_id}", 
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(display_mosaic, "Click views to annotate. 'n': next person, 'f': next frame, 'q': quit", 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow("Annotation Tool", display_mosaic)
            
            cv2.setMouseCallback("Annotation Tool", self._mouse_callback, 
                               (frames, frame_data, frames[0].shape))
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('n'):
                if self.current_person_id in frame_data:
                    self.current_person_id += 1
            elif key == ord('f'):
                break
            elif key == ord('q'):
                return frame_data
        
        return frame_data
    
    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            frames, frame_data, frame_shape = param
            view_idx = self._get_view_from_coordinates(x, y, frame_shape)
            
            if view_idx is not None:
                self._annotate_person_in_view(frames[view_idx], view_idx, frame_data)
    
    def _get_view_from_coordinates(self, x: int, y: int, frame_shape: Tuple) -> Optional[int]:
        h, w = frame_shape[:2]
        scale = 0.5
        view_w, view_h = int(w * scale), int(h * scale)
        
        if y < view_h:
            if x < view_w:
                return 0
            elif x < view_w * 2:
                return 1
        else:
            if x < view_w:
                return 2
            elif x < view_w * 2:
                return 3
        
        return None
    
    def _annotate_person_in_view(self, frame: np.ndarray, view_idx: int, frame_data: Dict):
        window_name = f"Annotate Person {self.current_person_id} - View {view_idx + 1}"
        
        display_frame = frame.copy()
        if self.current_person_id in frame_data and view_idx in frame_data[self.current_person_id]:
            bbox = frame_data[self.current_person_id][view_idx]
            cv2.rectangle(display_frame, (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        
        cv2.putText(display_frame, f"Select person {self.current_person_id}", 
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        bbox = cv2.selectROI(window_name, display_frame, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(window_name)
        
        if bbox[2] > 0 and bbox[3] > 0:
            x, y, w, h = bbox
            bbox_xyxy = [x, y, x + w, y + h]
            
            if self.current_person_id not in frame_data:
                frame_data[self.current_person_id] = {}
            
            frame_data[self.current_person_id][view_idx] = bbox_xyxy
    
    def _create_mosaic(self, frames: List[np.ndarray]) -> np.ndarray:
        if len(frames) < 4:
            return frames[0]
        
        h, w = frames[0].shape[:2]
        scale = 0.5
        resized_frames = [cv2.resize(frame, (int(w * scale), int(h * scale))) for frame in frames[:4]]
        
        top = np.hstack((resized_frames[0], resized_frames[1]))
        bottom = np.hstack((resized_frames[2], resized_frames[3]))
        mosaic = np.vstack((top, bottom))
        
        return mosaic
    
    def _draw_bbox_on_mosaic(self, mosaic: np.ndarray, bbox: List, view_idx: int, 
                           person_id: int, color: Tuple, original_shape: Tuple):
        h, w = original_shape[:2]
        scale = 0.5
        
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)
        
        if view_idx == 1:
            x1 += int(w * scale)
            x2 += int(w * scale)
        elif view_idx == 2:
            y1 += int(h * scale)
            y2 += int(h * scale)
        elif view_idx == 3:
            x1 += int(w * scale)
            x2 += int(w * scale)
            y1 += int(h * scale)
            y2 += int(h * scale)
        
        cv2.rectangle(mosaic, (x1, y1), (x2, y2), color, 2)
        cv2.putText(mosaic, f"P{person_id}", (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def parse_camera_parameters(param_type: str = 'old') -> List[CameraParameters]:
    camera_data_sets = {
        'old': [
            {'R': np.array([[-0.675, 0.738, 0.014], [0.156, 0.161, -0.975], [-0.721, -0.656, -0.224]], dtype=np.float32),
             'T': np.array([-1.819, 0.456, 8.807], dtype=np.float32),
             'K': np.array([[987.592, 0., 939.441], [0., 990.636, 490.471], [0., 0., 1.]], dtype=np.float32),
             'D': np.array([-0.288, 0.093, 0.001, 0.000, 0.000], dtype=np.float32)},
            {'R': np.array([[-0.733, -0.680, -0.029], [-0.110, 0.160, -0.981], [0.672, -0.716, -0.192]], dtype=np.float32),
             'T': np.array([1.354, 0.865, 9.156], dtype=np.float32),
             'K': np.array([[996.485, 0., 948.831], [0., 1000.034, 486.320], [0., 0., 1.]], dtype=np.float32),
             'D': np.array([-0.281, 0.073, 0.001, -0.000, 0.000], dtype=np.float32)},
            {'R': np.array([[0.633, -0.774, 0.018], [-0.266, -0.239, -0.934], [0.727, 0.586, -0.358]], dtype=np.float32),
             'T': np.array([-0.295, 0.409, 5.797], dtype=np.float32),
             'K': np.array([[983.350, 0., 937.236], [0., 987.666, 584.016], [0., 0., 1.]], dtype=np.float32),
             'D': np.array([-0.278, 0.073, -0.001, -0.000, 0.000], dtype=np.float32)},
            {'R': np.array([[0.996, -0.071, 0.052], [0.021, -0.383, -0.923], [0.085, 0.921, -0.380]], dtype=np.float32),
             'T': np.array([-0.233, 0.928, 5.0344], dtype=np.float32),
             'K': np.array([[1211.137, 0., 993.070], [0., 1222.058, 578.002], [0., 0., 1.]], dtype=np.float32),
             'D': np.array([-0.363, 0.118, -0.004, -0.005, 0.000], dtype=np.float32)}
        ],
        'new': [
            {'R': np.array([[0.738, 0.675, 0.016], [0.190, -0.185, -0.964], [-0.648, 0.714, -0.265]], dtype=np.float32),
             'T': np.array([-0.114, 0.298, 8.838], dtype=np.float32),
             'K': np.array([[979.692, 0., 935.499], [0., 980.944, 496.320], [0., 0., 1.]], dtype=np.float32),
             'D': np.array([-0.298, 0.099, -0.000, -0.002, 0.000], dtype=np.float32)},
            {'R': np.array([[0.652, -0.758, 0.017], [-0.189, -0.184, -0.965], [0.734, 0.626, -0.263]], dtype=np.float32),
             'T': np.array([-1.818, 0.491, 7.494], dtype=np.float32),
             'K': np.array([[1000.697, 0., 1002.663], [0., 1003.962, 553.965], [0., 0., 1.]], dtype=np.float32),
             'D': np.array([-0.282, 0.079, -0.001, -0.002, 0.000], dtype=np.float32)},
            {'R': np.array([[-0.840, 0.542, -0.018], [0.155, 0.207, -0.966], [-0.520, -0.814, -0.258]], dtype=np.float32),
             'T': np.array([1.907, 0.630, 6.971], dtype=np.float32),
             'K': np.array([[982.444, 0., 952.889], [0., 981.987, 521.117], [0., 0., 1.]], dtype=np.float32),
             'D': np.array([-0.253, 0.052, 0.001, 0.000, 0.000], dtype=np.float32)},
            {'R': np.array([[-0.999, 0.041, 0.033], [-0.021, 0.276, -0.961], [-0.049, -0.960, -0.275]], dtype=np.float32),
             'T': np.array([0.934, 1.096, 5.234], dtype=np.float32),
             'K': np.array([[1028.197, 0., 972.071], [0., 1030.518, 521.682], [0., 0., 1.]], dtype=np.float32),
             'D': np.array([-0.248, 0.048, -0.001, 0.000, 0.000], dtype=np.float32)}
        ]
    }
    
    camera_data = camera_data_sets.get(param_type, camera_data_sets['old'])
    camera_params = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for idx, cam_data in enumerate(camera_data):
        params = CameraParameters(
            K=torch.tensor(cam_data['K'], dtype=torch.float32, device=device),
            R=torch.tensor(cam_data['R'], dtype=torch.float32, device=device),
            T=torch.tensor(cam_data['T'], dtype=torch.float32, device=device),
            D=torch.tensor(cam_data['D'], dtype=torch.float32, device=device),
            view_idx=idx
        )
        camera_params.append(params)
    
    return camera_params

def get_user_choice_for_initialization(json_file_path: str) -> str:
    json_exists = os.path.exists(json_file_path)
    
    if json_exists:
        choice = input(f"Found existing config {json_file_path}. Use it? (y/n): ").strip().lower()
        return "json" if choice == 'y' else "annotation"
    else:
        choice = input("No config found. Do manual annotation? (y/n): ").strip().lower()
        return "annotation" if choice == 'y' else "cancel"

def load_initial_configuration(config_file: str) -> Optional[Dict[int, Dict[int, Tuple]]]:
    if not config_file or not os.path.exists(config_file):
        return None
    
    with open(config_file, 'r') as f:
        config_data = json.load(f)
    
    if 'init_bboxes' not in config_data:
        return None
    
    init_bboxes = {}
    for track_id_str, bbox_data in config_data['init_bboxes'].items():
        track_id = int(track_id_str)
        
        if isinstance(bbox_data, dict):
            first_frame_bboxes = bbox_data.get('0') or bbox_data.get(0)
            if first_frame_bboxes:
                valid_bboxes = {}
                for i, bbox in enumerate(first_frame_bboxes):
                    if bbox is not None:
                        valid_bboxes[i] = tuple(bbox)
                if valid_bboxes:
                    init_bboxes[track_id] = valid_bboxes
        else:
            valid_bboxes = {}
            for i, bbox in enumerate(bbox_data):
                if bbox is not None:
                    valid_bboxes[i] = tuple(bbox)
            if valid_bboxes:
                init_bboxes[track_id] = valid_bboxes
    
    return init_bboxes

def do_manual_annotation(video_paths: List[str], num_cameras: int) -> Dict[int, Dict[int, Tuple]]:
    annotation_tool = AnnotationTool(num_cameras)
    frame_annotations = annotation_tool.annotate_frames(video_paths, num_frames=5)
    
    init_bboxes = {}
    if 0 in frame_annotations:
        for person_id, views_data in frame_annotations[0].items():
            valid_bboxes = {}
            for view_idx, bbox in views_data.items():
                if bbox is not None:
                    valid_bboxes[view_idx] = tuple(bbox)
            if valid_bboxes:
                init_bboxes[person_id] = valid_bboxes
    
    return init_bboxes

def save_initialization_config(init_bboxes: Dict[int, Dict[int, Tuple]], config_file: str):
    config_data = {
        "init_bboxes": {}
    }
    
    for person_id, view_bboxes in init_bboxes.items():
        config_data["init_bboxes"][str(person_id)] = {
            "0": [None] * 4
        }
        
        for view_idx, bbox in view_bboxes.items():
            if view_idx < 4:
                config_data["init_bboxes"][str(person_id)]["0"][view_idx] = list(bbox)
    
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)

def get_initialization_data(config_file: str, video_paths: List[str], num_cameras: int) -> Optional[Dict[int, Dict[int, Tuple]]]:
    choice = get_user_choice_for_initialization(config_file)
    
    if choice == "cancel":
        return None
    elif choice == "json":
        return load_initial_configuration(config_file)
    elif choice == "annotation":
        init_bboxes = do_manual_annotation(video_paths, num_cameras)
        if init_bboxes:
            save_initialization_config(init_bboxes, config_file)
        return init_bboxes
    
    return None

def load_video_streams(dataset_path: str, num_cameras: int) -> List[str]:
    dataset_dir = Path(dataset_path)
    video_dir = dataset_dir / "videos"
    
    video_paths = []
    for i in range(1, num_cameras + 1):
        video_file = video_dir / f"{i}.mp4"
        if video_file.exists():
            video_paths.append(str(video_file))
        else:
            alt_video_file = video_dir / f"camera_{i:02d}.mp4"
            if alt_video_file.exists():
                video_paths.append(str(alt_video_file))
    
    if len(video_paths) != num_cameras:
        raise ValueError(f"Expected {num_cameras} videos, found {len(video_paths)} in {video_dir}")
    
    return video_paths

def setup_video_captures(video_paths: List[str]) -> List[cv2.VideoCapture]:
    caps = []
    for i, path in enumerate(video_paths):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video {i+1}: {path}")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        caps.append(cap)
    return caps

def read_frame_batch(caps: List[cv2.VideoCapture], device: torch.device) -> Optional[List[torch.Tensor]]:
    frames = []
    
    with torch.cuda.device(device):
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                return None
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frames.append(frame_tensor.to(device, non_blocking=True))
    
    return frames

def create_argument_parser():
    parser = argparse.ArgumentParser(description='Super Robust Multi-View Tracker')
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--config_file', type=str, default=None)
    parser.add_argument('--num_persons', type=int, default=2)
    parser.add_argument('--num_cameras', type=int, default=4)
    parser.add_argument('--detection_conf', type=float, default=0.5)
    parser.add_argument('--camera_params', type=str, default='new', choices=['old', 'new'])
    parser.add_argument('--spring_constant', type=float, default=300.0)
    parser.add_argument('--damping_coefficient', type=float, default=5.0)
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--output_dir', type=str, default=None)
    return parser