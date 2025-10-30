import torch
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import gc

from track.utils import TrackingConfig, SpringDamperConfig, CameraParameters, PerformanceMonitor, DetectionEngine
from track.geometry_3d import Projection3D, SpringDamperPhysics
from track.mobile_sam_wrapper import MobileSAMWrapper
from memory_manager import MemoryManager
from xmem_network import XMem

import torch.nn.functional as F

def aggregate(prob, dim, return_logits=False):
    new_prob = torch.cat([
        torch.prod(1-prob, dim=dim, keepdim=True),
        prob
    ], dim).clamp(1e-7, 1-1e-7)
    logits = torch.log((new_prob /(1-new_prob)))
    prob = F.softmax(logits, dim=dim)
    if return_logits:
        return logits, prob
    else:
        return prob

def pad_divide_by(in_img, d):
    h, w = in_img.shape[-2:]
    if h % d > 0:
        new_h = h + d - h % d
    else:
        new_h = h
    if w % d > 0:
        new_w = w + d - w % d
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pad_array = (int(lw), int(uw), int(lh), int(uh))
    out = F.pad(in_img, pad_array)
    return out, pad_array

def unpad(img, pad):
    if len(img.shape) == 4:
        if pad[2]+pad[3] > 0:
            img = img[:,:,pad[2]:-pad[3],:]
        if pad[0]+pad[1] > 0:
            img = img[:,:,:,pad[0]:-pad[1]]
    elif len(img.shape) == 3:
        if pad[2]+pad[3] > 0:
            img = img[:,pad[2]:-pad[3],:]
        if pad[0]+pad[1] > 0:
            img = img[:,:,pad[0]:-pad[1]]
    return img

def im_normalization(img):
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device, dtype=img.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device, dtype=img.dtype).view(3, 1, 1)
    return (img - mean) / std

def image_to_torch_optimized(frame: np.ndarray, device='cuda'):
    if len(frame.shape) == 2:
        frame = np.stack([frame, frame, frame], axis=2)
    frame = frame.transpose(2, 0, 1)
    frame = torch.from_numpy(frame).float().to(device) / 255.0
    frame_norm = im_normalization(frame)
    return frame_norm

def index_numpy_to_one_hot_torch(mask, num_classes):
    mask = torch.from_numpy(mask).long()
    mask = torch.clamp(mask, 0, num_classes - 1)
    one_hot = torch.nn.functional.one_hot(mask, num_classes=num_classes)
    return one_hot.permute(2, 0, 1).float()

class InferenceCore:
    def __init__(self, network: XMem, config):
        self.config = config
        self.network = network
        self.mem_every = config['mem_every']
        self.deep_update_every = config['deep_update_every']
        self.enable_long_term = config['enable_long_term']
        self.deep_update_sync = (self.deep_update_every < 0)
        self.clear_memory()
        self.all_labels = None

    def clear_memory(self):
        self.curr_ti = -1
        self.last_mem_ti = 0
        if not self.deep_update_sync:
            self.last_deep_update_ti = -self.deep_update_every
        self.memory = MemoryManager(config=self.config)

    def set_all_labels(self, all_labels):
        self.all_labels = all_labels

    def step(self, image, mask=None, valid_labels=None, end=False):
        self.curr_ti += 1
        image, self.pad = pad_divide_by(image, 16)
        image = image.unsqueeze(0)

        is_mem_frame = ((self.curr_ti-self.last_mem_ti >= self.mem_every) or (mask is not None)) and (not end)
        need_segment = (self.curr_ti > 0) and ((valid_labels is None) or (len(self.all_labels) != len(valid_labels)))
        is_deep_update = (
            (self.deep_update_sync and is_mem_frame) or
            (not self.deep_update_sync and self.curr_ti-self.last_deep_update_ti >= self.deep_update_every)
        ) and (not end)
        is_normal_update = (not self.deep_update_sync or not is_deep_update) and (not end)

        key, shrinkage, selection, f16, f8, f4 = self.network.encode_key(image, 
                                                    need_ek=(self.enable_long_term or need_segment), 
                                                    need_sk=is_mem_frame)
        multi_scale_features = (f16, f8, f4)

        if need_segment:
            memory_readout = self.memory.match_memory(key, selection).unsqueeze(0)
            hidden, _, pred_prob_with_bg = self.network.segment(multi_scale_features, memory_readout, 
                                    self.memory.get_hidden(), h_out=is_normal_update, strip_bg=False)
            pred_prob_with_bg = pred_prob_with_bg[0]
            pred_prob_no_bg = pred_prob_with_bg[1:]
            if is_normal_update:
                self.memory.set_hidden(hidden)
        else:
            pred_prob_no_bg = pred_prob_with_bg = None

        if mask is not None:
            mask, _ = pad_divide_by(mask, 16)
            if pred_prob_no_bg is not None:
                mask_regions = (mask.sum(0) > 0.5)
                pred_prob_no_bg[:, mask_regions] = 0
                mask = mask.type_as(pred_prob_no_bg)
                if valid_labels is not None:
                    shift_by_one_non_labels = [i for i in range(pred_prob_no_bg.shape[0]) if (i+1) not in valid_labels]
                    mask[shift_by_one_non_labels] = pred_prob_no_bg[shift_by_one_non_labels]
            pred_prob_with_bg = aggregate(mask, dim=0)
            self.memory.create_hidden_state(len(self.all_labels), key)

        if is_mem_frame:
            value, hidden = self.network.encode_value(image, f16, self.memory.get_hidden(), 
                                    pred_prob_with_bg[1:].unsqueeze(0), is_deep_update=is_deep_update)
            self.memory.add_memory(key, shrinkage, value, self.all_labels, 
                                    selection=selection if self.enable_long_term else None)
            self.last_mem_ti = self.curr_ti
            if is_deep_update:
                self.memory.set_hidden(hidden)
                self.last_deep_update_ti = self.curr_ti
                
        return unpad(pred_prob_with_bg, self.pad)

class StableTrack:
    def __init__(self, track_id: int, position_3d: torch.Tensor, size_3d: torch.Tensor, device: torch.device):
        self.id = track_id
        self.position_3d = position_3d.to(device)
        self.velocity_3d = torch.zeros(3, device=device)
        self.size_3d = size_3d.to(device)
        
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.confidence = 0.8
        
        self.bboxes = {}
        self.xmem_masks = {}
        self.mask_centroids = {}
        self.view_confidences = {}
        self.geometric_consistency = 1.0
        self.occlusion_states = {}  # Track occlusion per view
        self.detection_count = {}   # Count detections per view
        
        self.device = device
        self.last_seen_frame = 0
        
        # Initialize view states
        for view_idx in range(4):
            self.occlusion_states[view_idx] = 'VISIBLE'
            self.detection_count[view_idx] = 0
        
    def is_active(self) -> bool:
        return self.time_since_update < 60 and self.confidence > 0.1  # 60fps * 1sec
    
    def update_bbox(self, view_idx: int, bbox: Tuple, confidence: float):
        self.bboxes[view_idx] = bbox
        self.view_confidences[view_idx] = confidence
        self.occlusion_states[view_idx] = 'VISIBLE'
        self.detection_count[view_idx] = self.detection_count.get(view_idx, 0) + 1
        
    def update_mask(self, view_idx: int, mask: torch.Tensor):
        if mask.sum() > 100:
            self.xmem_masks[view_idx] = mask
            y_coords, x_coords = torch.where(mask > 0.5)
            centroid_x = x_coords.float().mean().item()
            centroid_y = y_coords.float().mean().item()
            self.mask_centroids[view_idx] = (centroid_x, centroid_y)
    
    def mark_occluded(self, view_idx: int):
        self.occlusion_states[view_idx] = 'OCCLUDED'
        # Remove bbox when occluded - don't show anything
        if view_idx in self.bboxes:
            del self.bboxes[view_idx]
        if view_idx in self.view_confidences:
            del self.view_confidences[view_idx]
    
    def mark_missing(self, view_idx: int):
        self.occlusion_states[view_idx] = 'MISSING'
        # Remove bbox when missing - don't show anything
        if view_idx in self.bboxes:
            del self.bboxes[view_idx]
        if view_idx in self.view_confidences:
            del self.view_confidences[view_idx]
    
    def is_visible_in_view(self, view_idx: int) -> bool:
        return (self.occlusion_states.get(view_idx, 'MISSING') == 'VISIBLE' and 
                view_idx in self.bboxes and 
                self.bboxes[view_idx] is not None)

class XMemProcessor:
    def __init__(self, xmem_config, max_obj_cnt, device, view_idx):
        self.device = device
        self.view_idx = view_idx
        self.max_obj_cnt = max_obj_cnt
        self.initialized = False
        
        try:
            if device.lower() != 'cpu':
                self.network = XMem(xmem_config, './saves/XMem.pth').eval().to('cuda')
            else:
                self.network = XMem(xmem_config, './saves/XMem.pth', map_location='cpu').eval()
            
            self.processor = InferenceCore(self.network, config=xmem_config)
            self.processor.set_all_labels(range(1, max_obj_cnt + 1))
            self.initialized = True
        except:
            self.processor = None
            self.initialized = False
        
        # SAM for initial masks
        try:
            from mobile_sam import SamPredictor, sam_model_registry
            sam = sam_model_registry['vit_t'](checkpoint='./saves/mobile_sam.pt')
            if device.lower() != 'cpu':
                sam.to(device='cuda')
            else:
                sam.to(device='cpu')
            self.sam_predictor = SamPredictor(sam)
            self.sam_available = True
        except:
            self.sam_available = False
    
    def create_mask_from_img(self, image, yolov_bboxes):
        """Create mask using SAM - following original approach"""
        if not self.sam_available or not yolov_bboxes:
            # Fallback to bbox masks
            result = np.zeros(image.shape[:2], dtype=np.uint8)
            for i, bbox in enumerate(yolov_bboxes[:self.max_obj_cnt]):
                x1, y1, x2, y2 = [int(c) for c in bbox]
                result[y1:y2, x1:x2] = i + 1
            return result
        
        self.sam_predictor.set_image(image)
        input_boxes = torch.tensor(yolov_bboxes, device=self.sam_predictor.device)
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
        
        masks = []
        for box in transformed_boxes:
            mask, _, _ = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=box.unsqueeze(0),
                multimask_output=False,
            )
            # Clean mask values
            values, counts = torch.unique(mask, return_counts=True)
            value_count = [(v.item(), c.item()) for v, c in zip(values, counts)]
            value_count = sorted(value_count, key=lambda x: x[1], reverse=True)
            mask[mask != 0] = value_count[0][0] if value_count[0][0] != 0 else value_count[1][0]
            masks.append(mask)
        
        # Create result mask like original
        result = np.zeros(image.shape[:2], dtype=np.uint8)
        for i, mask in enumerate(masks):
            binary_mask = mask.cpu().squeeze().numpy().astype(np.uint8)
            result[binary_mask > 0] = i + 1
        
        # # Filter small segments - keep only largest areas per ID
        # if len(np.unique(result)) > len(yolov7_bboxes) + 1:
        #     mask_uniq_values = np.unique(result).tolist()
        #     class_pixel_cnts = [np.sum(result == val) for val in mask_uniq_values]
        #     sorted_indices = np.argsort(class_pixel_cnts)[::-1].tolist()
            
        #     filtered_values = []
        #     for index in sorted_indices:
        #         filtered_values.append(mask_uniq_values[index])
        #         if len(filtered_values) == len(yolov7_bboxes) + 1:
        #             break
            
        #     for pixel_val in mask_uniq_values:
        #         if pixel_val not in filtered_values:
        #             result[result == pixel_val] = 0
        
        return result
    
    def add_mask(self, image, mask):
        """Add mask to XMem - following original approach"""
        if not self.initialized:
            return None
        
        if self.device.lower() != 'cpu':
            frame_torch = image_to_torch_optimized(image, device='cuda')
            mask_torch = index_numpy_to_one_hot_torch(mask, self.max_obj_cnt + 1).to('cuda')
        else:
            frame_torch = image_to_torch_optimized(image, device='cpu')
            mask_torch = index_numpy_to_one_hot_torch(mask, self.max_obj_cnt + 1).to('cpu')
        
        return self.processor.step(frame_torch, mask_torch[1:])
    
    def predict(self, image):
        """Predict masks - following original approach"""
        if not self.initialized:
            return None
        
        if self.device.lower() != 'cpu':
            frame_torch = image_to_torch_optimized(image, device='cuda')
        else:
            frame_torch = image_to_torch_optimized(image, device='cpu')
        
        return self.processor.step(frame_torch)
    
    def masks_to_boxes_with_ids(self, mask_tensor):
        """Extract bboxes with IDs - following original approach"""
        unique_values = torch.unique(mask_tensor[mask_tensor != 0])
        bbox_list = []
        
        for unique_value in unique_values:
            binary_mask = (mask_tensor == unique_value).byte()
            nonzero_coords = torch.nonzero(binary_mask, as_tuple=False)
            
            if nonzero_coords.numel() > 0:
                min_x = torch.min(nonzero_coords[:, 2])
                min_y = torch.min(nonzero_coords[:, 1])
                max_x = torch.max(nonzero_coords[:, 2])
                max_y = torch.max(nonzero_coords[:, 1])
                
                bbox = [unique_value.item(), min_x.item(), min_y.item(), max_x.item(), max_y.item()]
                bbox_list.append(bbox)
        
        return bbox_list

class GeometricTracker:
    def __init__(self, projection: Projection3D, physics: SpringDamperPhysics):
        self.projection = projection
        self.physics = physics
        self.consistency_threshold = 0.6
        
    def triangulate_position(self, detections: Dict[int, Tuple], 
                           confidences: Dict[int, float]) -> Optional[torch.Tensor]:
        if len(detections) < 2:
            return None
        
        centers_2d = {}
        for view_idx, bbox in detections.items():
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            centers_2d[view_idx] = np.array([center_x, center_y])
        
        return self.projection.triangulate_multi_view_refined(centers_2d, confidences)
    
    def compute_geometric_consistency(self, track: StableTrack, 
                                    detections: Dict[int, Tuple]) -> float:
        if len(detections) < 2:
            return 0.0
        
        projected_positions = {}
        for view_idx in detections.keys():
            projected = self.projection.world_to_camera(track.position_3d, view_idx)
            if projected is not None:
                projected_positions[view_idx] = projected.cpu().numpy()
        
        if len(projected_positions) < 2:
            return 0.0
        
        total_error = 0.0
        count = 0
        
        for view_idx, bbox in detections.items():
            if view_idx in projected_positions:
                det_center = np.array([(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2])
                proj_center = projected_positions[view_idx]
                error = np.linalg.norm(det_center - proj_center)
                total_error += error
                count += 1
        
        if count == 0:
            return 0.0
        
        avg_error = total_error / count
        consistency = max(0.0, 1.0 - avg_error / 100.0)
        return consistency

class StableIDManager:
    def __init__(self, max_tracks: int = 2):
        self.max_tracks = max_tracks
        self.active_tracks = {}
        self.track_features = {}
        self.available_ids = list(range(1, max_tracks + 1))
        self.used_ids = set()
        
    def create_track(self, position_3d: torch.Tensor, size_3d: torch.Tensor, 
                    device: torch.device) -> StableTrack:
        if not self.available_ids:
            return None
        
        track_id = self.available_ids.pop(0)
        self.used_ids.add(track_id)
        
        track = StableTrack(track_id, position_3d, size_3d, device)
        self.active_tracks[track_id] = track
        self.track_features[track_id] = {
            'position_history': deque(maxlen=10),
            'confidence_history': deque(maxlen=10)
        }
        
        return track
    
    def remove_track(self, track: StableTrack):
        if track.id in self.active_tracks:
            del self.active_tracks[track.id]
            del self.track_features[track.id]
            self.used_ids.discard(track.id)
            self.available_ids.append(track.id)
            self.available_ids.sort()
    
    def get_active_tracks(self) -> List[StableTrack]:
        return list(self.active_tracks.values())

class StableTracker:
    def __init__(self, tracking_config: TrackingConfig, spring_config: SpringDamperConfig, 
                 camera_params: List[CameraParameters]):
        self.config = tracking_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.projection = Projection3D(camera_params)
        self.physics = SpringDamperPhysics(spring_config, self.projection)
        self.geometric_tracker = GeometricTracker(self.projection, self.physics)
        self.detection_engine = DetectionEngine('yolov8m.pt', tracking_config.detection_confidence)
        self.performance_monitor = PerformanceMonitor()
        
        self.id_manager = StableIDManager(tracking_config.num_persons)
        self.frame_idx = 0
        
        # Simple XMem config - like the original working version
        xmem_config = {
            'mem_every': 5,
            'deep_update_every': 100,
            'enable_long_term': True,
            'enable_long_term_count_usage': True,
            'key_dim': 64,
            'value_dim': 256,
            'hidden_dim': 32,
            'top_k': 20
        }
        
        self.xmem_processors = []
        for view_idx in range(tracking_config.num_cameras):
            processor = XMemProcessor(xmem_config, tracking_config.num_persons, self.device, view_idx)
            self.xmem_processors.append(processor)
        
        self.association_threshold = 0.8
        self.reid_threshold = 0.6
        
    @property
    def tracks(self) -> List[StableTrack]:
        return self.id_manager.get_active_tracks()
    
    def initialize_from_config(self, initial_detections: Dict[int, Dict[int, Tuple]]) -> Dict[int, int]:
        track_mapping = {}
        
        # Only create tracks from initial config, use config IDs as track IDs
        for config_id, view_detections in initial_detections.items():
            if len(view_detections) >= 2 and config_id <= self.config.num_persons:
                centers_2d = {}
                confidences = {}
                for view_idx, bbox in view_detections.items():
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    centers_2d[view_idx] = np.array([center_x, center_y])
                    confidences[view_idx] = 0.8
                
                world_pos = self.projection.triangulate_multi_view_refined(centers_2d, confidences)
                if world_pos is not None:
                    # Use config_id directly as track_id (must be 1 or 2)
                    track_id = config_id
                    track = StableTrack(track_id, world_pos, 
                                      torch.tensor([0.4, 0.4, 0.9], device=self.device),  # Half the size
                                      self.device)
                    
                    # Add to id_manager manually
                    self.id_manager.active_tracks[track_id] = track
                    self.id_manager.track_features[track_id] = {
                        'position_history': deque(maxlen=10),
                        'confidence_history': deque(maxlen=10)
                    }
                    if track_id in self.id_manager.available_ids:
                        self.id_manager.available_ids.remove(track_id)
                    self.id_manager.used_ids.add(track_id)
                    
                    for view_idx, bbox in view_detections.items():
                        track.update_bbox(view_idx, bbox, 0.8)
                    
                    track_mapping[config_id] = track_id
        
        return track_mapping
    
    def add_xmem_masks(self, frames_np: List[np.ndarray], initial_detections: Dict[int, Dict[int, Tuple]]):
        """Initialize XMem like the original - simple and effective"""
        for view_idx, frame_np in enumerate(frames_np):
            if view_idx < len(self.xmem_processors) and self.xmem_processors[view_idx].initialized:
                # Collect bboxes for this view in correct format
                view_bboxes = []
                for config_id in sorted(initial_detections.keys()):
                    if view_idx in initial_detections[config_id] and config_id <= self.config.num_persons:
                        view_bboxes.append(initial_detections[config_id][view_idx])
                
                if view_bboxes:
                    # Create mask using SAM like original
                    mask = self.xmem_processors[view_idx].create_mask_from_img(frame_np, view_bboxes)
                    # Add to XMem
                    self.xmem_processors[view_idx].add_mask(frame_np, mask)
    
    def process_frame(self, frames: List[torch.Tensor], scales: List[float] = None) -> Dict[str, Any]:
        start_time = self.performance_monitor.start_timer()
        
        frames_np = []
        for frame in frames:
            frame_np = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
            frames_np.append(frame_np)
        
        # XMem prediction
        xmem_results = []
        for view_idx, frame_np in enumerate(frames_np):
            if view_idx < len(self.xmem_processors) and self.xmem_processors[view_idx].initialized:
                result = self.xmem_processors[view_idx].predict(frame_np)
                xmem_results.append(result)
            else:
                xmem_results.append(None)
        
        # Detection
        all_detections = self.detection_engine.detect_persons_batch(frames)
        if scales:
            all_detections = self._rescale_detections(all_detections, scales)
        
        # Tracking pipeline
        self._predict_tracks()
        assignments = self._associate_detections(all_detections, xmem_results)
        
        self._update_tracks(assignments, all_detections, xmem_results)
        self._handle_unassigned_detections(all_detections, assignments)
        self._cleanup_tracks()
        
        total_time = self.performance_monitor.end_timer(start_time, 'total_frame')
        self.frame_idx += 1
        
        return {
            'detections': all_detections,
            'tracks': self.tracks,
            'frame_time': total_time
        }
    
    def _predict_tracks(self):
        for track in self.tracks:
            # Update physics with 60fps
            dt = 1/60.0
            track.position_3d += track.velocity_3d * dt
            track.velocity_3d *= 0.98
            
            # Clear all bboxes - only show actual detections, not predictions
            # This ensures occluded/missing tracks don't show frozen bboxes
            track.bboxes.clear()
            track.view_confidences.clear()
    
    def _associate_detections(self, all_detections: List[List[Dict]], 
                            xmem_results: List[torch.Tensor]) -> Dict[int, Dict[int, int]]:
        assignments = {}
        used_detections = set()
        
        active_tracks = [t for t in self.tracks if t.confidence > 0.1]
        
        for view_idx, detections in enumerate(all_detections):
            if not detections or not active_tracks:
                continue
            
            cost_matrix = np.full((len(active_tracks), len(detections)), 1.0)
            
            for track_idx, track in enumerate(active_tracks):
                for det_idx, detection in enumerate(detections):
                    if (view_idx, det_idx) in used_detections:
                        continue
                    
                    cost = self._compute_association_cost(
                        track, detection, view_idx, xmem_results
                    )
                    
                    if cost < self.association_threshold:
                        cost_matrix[track_idx, det_idx] = cost
            
            # Hungarian assignment
            if np.any(cost_matrix < 1.0):
                from scipy.optimize import linear_sum_assignment
                row_indices, col_indices = linear_sum_assignment(cost_matrix)
                
                for r, c in zip(row_indices, col_indices):
                    if cost_matrix[r, c] < self.association_threshold:
                        track = active_tracks[r]
                        if track.id not in assignments:
                            assignments[track.id] = {}
                        assignments[track.id][view_idx] = c
                        used_detections.add((view_idx, c))
        
        return assignments
    
    def _compute_association_cost(self, track: StableTrack, detection: Dict, 
                                view_idx: int, xmem_results: List[torch.Tensor]) -> float:
        det_bbox = detection['bbox']
        det_center = np.array([(det_bbox[0] + det_bbox[2])/2, (det_bbox[1] + det_bbox[3])/2])
        
        # Use 3D projection for association (but not for display)
        projected_bbox = self.projection.project_3d_bbox(track.position_3d, track.size_3d, view_idx)
        
        # Geometric cost based on 3D projection
        geometric_cost = 1.0
        if projected_bbox is not None:
            iou = self._compute_iou(projected_bbox, det_bbox)
            geometric_cost = 1.0 - iou
        
        # Position consistency cost
        position_cost = 1.0
        projected_center = self.projection.world_to_camera(track.position_3d, view_idx)
        if projected_center is not None:
            proj_center = projected_center.cpu().numpy()
            distance = np.linalg.norm(det_center - proj_center)
            position_cost = min(1.0, distance / 90.0)
        
        # XMem mask cost - simplified like original
        mask_cost = 1
        if (view_idx < len(xmem_results) and xmem_results[view_idx] is not None):
            # Convert XMem result to mask bboxes like original
            from pose_estimation.track.xmem_interactive_utils import torch_prob_to_numpy_mask
            
            mask_np = torch_prob_to_numpy_mask(xmem_results[view_idx])
            mask_tensor = torch.tensor(mask_np).unsqueeze(0)
            
            # Get bboxes from XMem masks
            mask_bboxes = self.xmem_processors[view_idx].masks_to_boxes_with_ids(mask_tensor)
            
            # Find matching bbox for this track
            for bbox_info in mask_bboxes:
                if len(bbox_info) >= 5 and bbox_info[0] == track.id:
                    mask_bbox = bbox_info[1:5]  # [x1, y1, x2, y2]
                    # Compute IoU with detection
                    iou = self._compute_iou(mask_bbox, det_bbox)
                    mask_cost = 1.0 - iou
                    break
        
        # Check for occlusion - if IoU with other tracks is high, increase cost
        occlusion_penalty = 0.0
        for other_track in self.tracks:
            if other_track.id != track.id:
                other_projected = self.projection.project_3d_bbox(other_track.position_3d, other_track.size_3d, view_idx)
                if other_projected is not None:
                    overlap_iou = self._compute_iou(projected_bbox, other_projected)
                    if overlap_iou > 0.3:  # High overlap indicates occlusion
                        occlusion_penalty = 0.1
                        break
        
        # Combined cost - simple like original
        total_cost = 0.35 * geometric_cost + 0.35 * position_cost + 0.35 * mask_cost + occlusion_penalty
        return total_cost
    
    def _update_tracks(self, assignments: Dict[int, Dict[int, int]], 
                      all_detections: List[List[Dict]], xmem_results: List[torch.Tensor]):
        
        for track_id, view_assignments in assignments.items():
            track = next((t for t in self.tracks if t.id == track_id), None)
            if track is None:
                continue
            
            current_detections = {}
            current_confidences = {}
            
            for view_idx, det_idx in view_assignments.items():
                detection = all_detections[view_idx][det_idx]
                bbox = detection['bbox']
                confidence = detection['conf']
                
                track.update_bbox(view_idx, bbox, confidence)
                current_detections[view_idx] = bbox
                current_confidences[view_idx] = confidence
                
                # Update XMem mask - fix index mapping
                if (view_idx < len(xmem_results) and xmem_results[view_idx] is not None):
                    xmem_channel = track.id - 1  # Convert 1-based track ID to 0-based XMem channel
                    if xmem_channel < xmem_results[view_idx].shape[0]:
                        track.update_mask(view_idx, xmem_results[view_idx][xmem_channel])
            
            # Update 3D position
            if len(current_detections) >= 2:
                new_position = self.geometric_tracker.triangulate_position(
                    current_detections, current_confidences
                )
                
                if new_position is not None:
                    # Update velocity and position with 60fps
                    dt = 1/60.0
                    velocity = (new_position - track.position_3d) / dt
                    
                    # Smooth velocity update
                    alpha = 0.2  # Lower for 60fps
                    track.velocity_3d = alpha * velocity + (1 - alpha) * track.velocity_3d
                    
                    # Update position with physics
                    track.position_3d = new_position
                    
                    # Update geometric consistency
                    track.geometric_consistency = self.geometric_tracker.compute_geometric_consistency(
                        track, current_detections
                    )
            
            track.hits += 1
            track.time_since_update = 0
            track.confidence = min(1.0, track.confidence + 0.1)
            track.last_seen_frame = self.frame_idx
        
        # Update unassigned tracks and handle occlusion/missing
        assigned_ids = set(assignments.keys())
        for track in self.tracks:
            if track.id not in assigned_ids:
                track.time_since_update += 1
                track.confidence = max(0.0, track.confidence - 0.02)
                
                # For each view, determine if track is occluded or just missing
                for view_idx in range(self.config.num_cameras):
                    is_occluded = False
                    projected_bbox = self.projection.project_3d_bbox(track.position_3d, track.size_3d, view_idx)
                    
                    if projected_bbox is not None:
                        # Check occlusion against other assigned tracks
                        for other_track in self.tracks:
                            if other_track.id != track.id and other_track.id in assigned_ids:
                                other_projected = self.projection.project_3d_bbox(other_track.position_3d, other_track.size_3d, view_idx)
                                if other_projected is not None:
                                    overlap_iou = self._compute_iou(projected_bbox, other_projected)
                                    if overlap_iou > 0.2:  # High overlap indicates occlusion
                                        is_occluded = True
                                        break
                    
                    if is_occluded:
                        track.mark_occluded(view_idx)
                    else:
                        # Not occluded, just missing detection
                        track.mark_missing(view_idx)
    
    def _handle_unassigned_detections(self, all_detections: List[List[Dict]], 
                                    assignments: Dict[int, Dict[int, int]]):
        # Only use initial tracks from JSON - no new track creation
        pass
    
    def _cleanup_tracks(self):
        # Only remove tracks that have been missing too long from ALL views
        tracks_to_remove = []
        
        for track in self.tracks:
            track.age += 1
            
            # Check if track is completely lost (missing from all views for too long)
            all_views_missing = True
            for view_idx in range(self.config.num_cameras):
                if track.occlusion_states.get(view_idx, 'MISSING') == 'VISIBLE':
                    all_views_missing = False
                    break
            
            # Only remove if track has been missing from all views for a long time
            if (all_views_missing and track.time_since_update > 180):  # 3 seconds at 60fps
                tracks_to_remove.append(track)
        
        for track in tracks_to_remove:
            self.id_manager.remove_track(track)
    
    def _rescale_detections(self, all_detections: List[List[Dict]], 
                          scales: List[float]) -> List[List[Dict]]:
        rescaled = []
        for view_idx, detections in enumerate(all_detections):
            if view_idx < len(scales):
                scale = scales[view_idx]
                rescaled_dets = []
                for det in detections:
                    bbox = det['bbox']
                    rescaled_bbox = tuple(int(coord * scale) for coord in bbox)
                    rescaled_dets.append({**det, 'bbox': rescaled_bbox})
                rescaled.append(rescaled_dets)
            else:
                rescaled.append(detections)
        return rescaled
    
    def _compute_iou(self, box1: Tuple, box2: Tuple) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / max(union, 1e-8)
