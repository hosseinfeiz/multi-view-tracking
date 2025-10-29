import numpy as np
import torch
import torch.nn.functional as F
# XMem utility functions
def im_normalization(img):
    """ImageNet normalization"""
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device, dtype=img.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device, dtype=img.dtype).view(3, 1, 1)
    return (img - mean) / std

def image_to_torch_optimized(frame: np.ndarray, device='cuda'):
    """Convert numpy image to normalized torch tensor with optimization"""
    if len(frame.shape) == 2:
        frame = np.stack([frame, frame, frame], axis=2)
    
    h, w = frame.shape[:2]
    max_size = 320
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        frame = cv2.resize(frame, (new_w, new_h))
    
    frame = frame.transpose(2, 0, 1)
    frame = torch.from_numpy(frame).float().to(device, non_blocking=True) / 255.0
    frame_norm = im_normalization(frame)
    return frame_norm

def index_numpy_to_one_hot_torch(mask, num_classes):
    """Convert indexed mask to one-hot tensor"""
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    mask = mask.astype(np.int64)
    mask = torch.from_numpy(mask).long()
    mask = torch.clamp(mask, 0, num_classes - 1)
    
    one_hot = torch.nn.functional.one_hot(mask, num_classes=num_classes)
    return one_hot.permute(2, 0, 1).float()

def keep_largest_connected_components(mask):
    """Keep largest connected components using OpenCV"""
    if mask is None:
        return None
    
    try:
        if torch.is_tensor(mask):
            mask_np = mask.squeeze().cpu().numpy()
            is_tensor = True
            original_device = mask.device
        else:
            mask_np = mask
            is_tensor = False
        
        unique_values = np.unique(mask_np)
        unique_values = unique_values[unique_values != 0]
        
        new_mask = np.zeros_like(mask_np)
        
        for class_value in unique_values:
            binary_mask = (mask_np == class_value).astype(np.uint8)
            
            if np.sum(binary_mask) < 100:
                continue
            
            h, w = binary_mask.shape
            kernel_size = max(3, min(11, min(h, w) // 30))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                binary_mask, connectivity=8
            )
            
            if num_labels > 1:
                component_areas = stats[1:, cv2.CC_STAT_AREA]
                largest_component_idx = 1 + np.argmax(component_areas)
                largest_component_mask = (labels == largest_component_idx)
                new_mask[largest_component_mask] = class_value
        
        if is_tensor:
            return torch.from_numpy(new_mask).unsqueeze(0).to(original_device)
        else:
            return new_mask
            
    except Exception as e:
        print(f"Connected components filtering error: {e}")
        return mask

# Simplified InferenceCore for internal use
class InferenceCore:
    def __init__(self, network, config):
        self.config = config
        self.network = network
        self.mem_every = max(1, config.get('mem_every', 5))
        self.enable_long_term = config.get('enable_long_term', False)
        self.max_spatial_size = 320
        self.clear_memory()
        self.all_labels = None
        self.stable_id_mapping = {}
        
    def clear_memory(self):
        self.curr_ti = -1
        self.last_mem_ti = 0
        # Use a simple memory placeholder
        self.memory = None
        torch.cuda.empty_cache()
        gc.collect()
    
    def set_all_labels(self, all_labels):
        self.all_labels = all_labels
        for i, label in enumerate(all_labels):
            if label not in self.stable_id_mapping:
                self.stable_id_mapping[label] = i + 1
    
    def step(self, image, mask=None, valid_labels=None, end=False):
        """Simplified step function"""
        self.curr_ti += 1
        
        if mask is not None:
            # When mask is provided, return it as probability
            if mask.dim() == 3:
                # Convert one-hot to probability
                return torch.softmax(mask, dim=0)
            else:
                # Convert index mask to one-hot then probability
                num_classes = len(self.all_labels) + 1 if self.all_labels else 3
                one_hot = index_numpy_to_one_hot_torch(mask.cpu().numpy(), num_classes)
                return torch.softmax(one_hot.to(mask.device), dim=0)
        else:
            # Prediction mode - return empty result for now
            h, w = image.shape[-2:]
            num_classes = len(self.all_labels) + 1 if self.all_labels else 3
            result = torch.zeros(num_classes, h, w, device=image.device)
            result[0] = 1.0  # Background
            return result# File: pose_estimation/track/tracker_core.pyimport torch
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import gc

from pose_estimation.track.utils import TrackingConfig, SpringDamperConfig, CameraParameters, PerformanceMonitor, DetectionEngine
from pose_estimation.track.geometry_3d import Projection3D, SpringDamperPhysics
from pose_estimation.track.xmem_network import XMem
from pose_estimation.track.mobile_sam_wrapper import MobileSAMWrapper

class StableTrack:
    def __init__(self, track_id: int, position_3d: torch.Tensor, size_3d: torch.Tensor, device: torch.device):
        self.id = track_id  # Stable, never-changing ID
        self.position_3d = position_3d.to(device)
        self.velocity_3d = torch.zeros(3, device=device)
        self.size_3d = size_3d.to(device)
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.confidence = 0.8
        self.bboxes = {}
        self.device = device
        
        # XMem integration
        self.xmem_masks = {}  # Store XMem masks per view
        self.mask_centroids = {}  # Store mask centroids for tracking
        self.feature_buffer = deque(maxlen=5)
        self.centroid_history = deque(maxlen=10)
        self.last_seen_frame = 0
        
        # Per-view state
        self.view_states = {}
        for view_idx in range(4):
            self.view_states[view_idx] = {
                'confidence': 0.0,
                'missing_count': 0,
                'last_centroid': None,
                'has_mask': False
            }
        
    def is_active(self) -> bool:
        return self.time_since_update < 25 and self.confidence > 0.2
    
    def update_view_state(self, view_idx: int, bbox: Tuple, confidence: float, mask_centroid=None):
        if view_idx in self.view_states:
            bbox_centroid = ((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2)
            self.view_states[view_idx]['confidence'] = confidence
            self.view_states[view_idx]['missing_count'] = 0
            self.view_states[view_idx]['last_centroid'] = bbox_centroid
            
            if mask_centroid is not None:
                self.mask_centroids[view_idx] = mask_centroid
                self.view_states[view_idx]['has_mask'] = True
            
            self.centroid_history.append((view_idx, bbox_centroid))
    
    def miss_view(self, view_idx: int):
        if view_idx in self.view_states:
            self.view_states[view_idx]['missing_count'] += 1
            if self.view_states[view_idx]['missing_count'] > 5:
                self.view_states[view_idx]['has_mask'] = False
    
    def get_predicted_centroid(self, view_idx: int) -> Optional[Tuple]:
        if view_idx in self.view_states:
            return self.view_states[view_idx]['last_centroid']
        return None

class StableIDManager:
    def __init__(self):
        self.next_id = 1
        self.active_tracks = {}  # Maps stable ID to track object
        self.id_history = set()  # All IDs ever used
        
    def create_track(self, position_3d: torch.Tensor, size_3d: torch.Tensor, device: torch.device) -> StableTrack:
        stable_id = self.next_id
        self.next_id += 1
        
        track = StableTrack(stable_id, position_3d, size_3d, device)
        self.active_tracks[stable_id] = track
        self.id_history.add(stable_id)
        
        return track
    
    def remove_track(self, track: StableTrack):
        if track.id in self.active_tracks:
            del self.active_tracks[track.id]
    
    def get_track_by_id(self, stable_id: int) -> Optional[StableTrack]:
        return self.active_tracks.get(stable_id, None)
    
    def get_active_tracks(self) -> List[StableTrack]:
        return list(self.active_tracks.values())
    
    def get_active_ids(self) -> List[int]:
        return sorted(list(self.active_tracks.keys()))

class XMemProcessor:
    """XMem processor for individual camera views"""
    
    def __init__(self, xmem_config, max_obj_cnt, device, view_idx):
        self.device = device
        self.view_idx = view_idx
        self.max_obj_cnt = max_obj_cnt
        self.frame_count = 0
        self.max_processing_size = 320
        
        # Stable ID management
        self.stable_id_mapping = {}
        self.next_stable_id = 1
        self.id_history = {}
        self.missing_counts = {}
        
        # Initialize XMem
        try:
            optimized_config = self._optimize_xmem_config(xmem_config)
            if device.lower() != 'cpu':
                self.network = XMem(optimized_config, './saves/XMem.pth').eval().to('cuda')
            else:
                self.network = XMem(optimized_config, './saves/XMem.pth', map_location='cpu').eval()
            
            self.processor = InferenceCore(self.network, config=optimized_config)
            self._initialize_stable_labels()
            self.initialized = True
            print(f"XMem initialized successfully for view {view_idx}")
        except Exception as e:
            print(f"XMem initialization failed for view {view_idx}: {e}")
            self.processor = None
            self.initialized = False
        
        # SAM integration
        self.mobile_sam = MobileSAMWrapper()
        self.sam_use_count = 0
        self.max_sam_uses = 100
    
    def _optimize_xmem_config(self, config):
        """Optimize XMem configuration for memory efficiency"""
        optimized = config.copy()
        optimized.update({
            'mem_every': 8,
            'deep_update_every': -1,
            'enable_long_term': False,
            'enable_long_term_count_usage': False,
            'max_mid_term_frames': 6,
            'min_mid_term_frames': 3,
            'num_prototypes': 32,
            'max_long_term_elements': 400,
            'top_k': 20,
            'hidden_dim': 16,
            'key_dim': 16,
            'value_dim': 32,
        })
        return optimized
    
    def _initialize_stable_labels(self):
        """Initialize stable label mapping"""
        stable_labels = []
        for i in range(1, self.max_obj_cnt + 1):
            stable_id = self._get_or_create_stable_id(i)
            stable_labels.append(stable_id)
            self.missing_counts[stable_id] = 0
            
        self.processor.set_all_labels(stable_labels)
    
    def _get_or_create_stable_id(self, internal_id):
        """Get or create stable ID for internal ID"""
        if internal_id not in self.stable_id_mapping:
            self.stable_id_mapping[internal_id] = self.next_stable_id
            self.next_stable_id += 1
        return self.stable_id_mapping[internal_id]
    
    def create_mask_from_img(self, image, yolov7_bboxes):
        """Create mask with stable IDs"""
        yolov7_bboxes = yolov7_bboxes[:min(len(yolov7_bboxes), self.max_obj_cnt)]
        
        if not yolov7_bboxes:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        
        self.sam_use_count += 1
        if self.sam_use_count > self.max_sam_uses or not self.mobile_sam.is_available:
            return self._fallback_segmentation(image, yolov7_bboxes)
        
        try:
            masks = self.mobile_sam.segment_from_boxes(image, yolov7_bboxes)
            
            if masks:
                result_masks = []
                for i, mask in enumerate(masks):
                    stable_id = self._get_or_create_stable_id(i + 1)
                    mask_cleaned = self._clean_mask(mask)
                    mask_cleaned[mask_cleaned > 0] = stable_id
                    result_masks.append(mask_cleaned)
                
                result = self._combine_masks(result_masks)
                result = self._filter_small_segments(result, yolov7_bboxes)
                return result
            else:
                return self._fallback_segmentation(image, yolov7_bboxes)
                
        except Exception as e:
            print(f"SAM segmentation failed for view {self.view_idx}: {e}")
            return self._fallback_segmentation(image, yolov7_bboxes)
    
    def _fallback_segmentation(self, image, bboxes):
        """Fallback segmentation using bounding boxes"""
        result = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for i, bbox in enumerate(bboxes[:self.max_obj_cnt]):
            x1, y1, x2, y2 = [max(0, int(coord)) for coord in bbox]
            x2 = min(x2, image.shape[1])
            y2 = min(y2, image.shape[0])
            
            if x2 > x1 and y2 > y1:
                stable_id = self._get_or_create_stable_id(i + 1)
                result[y1:y2, x1:x2] = stable_id
        
        return result
    
    def _clean_mask(self, mask):
        """Clean individual mask using OpenCV"""
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        mask = mask.astype(np.uint8)
        if mask.sum() < 50:
            return mask
        
        # Morphological operations
        kernel_size = max(3, min(7, int(np.sqrt(mask.sum()) / 10)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find largest connected component using OpenCV
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        if num_labels > 1:
            # Find largest component (excluding background label 0)
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = (labels == largest_label).astype(np.uint8) * 255
        
        return mask
    
    def _combine_masks(self, masks):
        """Combine individual masks"""
        if not masks:
            return np.zeros((100, 100), dtype=np.uint8)
        
        result = np.zeros_like(masks[0])
        for mask in masks:
            result[mask > 0] = mask[mask > 0]
        
        return result
    
    def _filter_small_segments(self, mask, original_bboxes):
        """Filter out small segments"""
        unique_values = np.unique(mask)
        unique_values = unique_values[unique_values != 0]
        
        if len(unique_values) <= len(original_bboxes):
            return mask
        
        # Keep largest segments
        value_sizes = [(val, np.sum(mask == val)) for val in unique_values]
        value_sizes.sort(key=lambda x: x[1], reverse=True)
        keep_values = {val for val, _ in value_sizes[:len(original_bboxes)]}
        
        filtered_mask = np.zeros_like(mask)
        for val in keep_values:
            filtered_mask[mask == val] = val
        
        return filtered_mask
    
    def predict(self, image):
        """Predict masks for current frame"""
        self.frame_count += 1
        
        if not self.initialized or self.processor is None:
            return self._create_empty_result(image.shape[:2])
        
        try:
            # Optimize image size
            original_size = image.shape[:2]
            h, w = original_size
            
            if max(h, w) > self.max_processing_size:
                scale = self.max_processing_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                image_small = cv2.resize(image, (new_w, new_h))
            else:
                image_small = image
                scale = 1.0
            
            # Convert to torch
            if self.device.lower() != 'cpu':
                frame_torch = image_to_torch_optimized(image_small, device='cuda')
            else:
                frame_torch = image_to_torch_optimized(image_small, device='cpu')
            
            # Process with XMem
            result = self.processor.step(frame_torch)
            
            # Scale result back if needed
            if scale != 1.0 and result is not None:
                result = self._upsample_result(result, original_size)
            
            # Apply ID stability
            result = self._stabilize_ids(result)
            
            return result
            
        except torch.cuda.OutOfMemoryError:
            print(f"CUDA OOM in XMem view {self.view_idx}")
            self._emergency_cleanup()
            return self._create_empty_result(image.shape[:2])
            
        except Exception as e:
            print(f"XMem prediction error for view {self.view_idx}: {e}")
            return self._create_empty_result(image.shape[:2])
    
    def add_mask(self, image, mask):
        """Add mask to XMem"""
        if not self.initialized or self.processor is None:
            return None
        
        try:
            # Update ID history
            self._update_id_history_from_mask(mask)
            
            # Optimize sizes
            original_size = image.shape[:2]
            h, w = original_size
            
            if max(h, w) > self.max_processing_size:
                scale = self.max_processing_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                image_small = cv2.resize(image, (new_w, new_h))
                mask_small = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            else:
                image_small = image
                mask_small = mask
                scale = 1.0
            
            # Convert to torch
            if self.device.lower() != 'cpu':
                frame_torch = image_to_torch_optimized(image_small, device='cuda')
                mask_torch = index_numpy_to_one_hot_torch(
                    mask_small, self.max_obj_cnt + 1).to('cuda')
            else:
                frame_torch = image_to_torch_optimized(image_small, device='cpu')
                mask_torch = index_numpy_to_one_hot_torch(
                    mask_small, self.max_obj_cnt + 1).to('cpu')
            
            print(f'Added mask to view {self.view_idx} with IDs: {np.unique(mask)}')
            
            # Process with XMem
            result = self.processor.step(frame_torch, mask_torch[1:])
            
            # Scale result back if needed
            if scale != 1.0 and result is not None:
                result = self._upsample_result(result, original_size)
            
            return result
            
        except Exception as e:
            print(f"Add mask error for view {self.view_idx}: {e}")
            return None
    
    def _update_id_history_from_mask(self, mask):
        """Update ID history from mask"""
        unique_ids = np.unique(mask)
        unique_ids = unique_ids[unique_ids != 0]
        
        for obj_id in unique_ids:
            obj_mask = (mask == obj_id)
            if np.sum(obj_mask) > 100:
                y_coords, x_coords = np.where(obj_mask)
                centroid_y = np.mean(y_coords)
                centroid_x = np.mean(x_coords)
                
                if obj_id not in self.id_history:
                    self.id_history[obj_id] = []
                
                self.id_history[obj_id].append((centroid_x, centroid_y))
                if len(self.id_history[obj_id]) > 10:
                    self.id_history[obj_id] = self.id_history[obj_id][-5:]
                
                self.missing_counts[obj_id] = 0
    
    def _stabilize_ids(self, result):
        """Apply ID stabilization"""
        if result is None:
            return result
        
        # Update missing counts
        for stable_id in self.missing_counts:
            if stable_id < result.shape[0]:
                mask_sum = result[stable_id].sum().item()
                if mask_sum < 100:
                    self.missing_counts[stable_id] += 1
                else:
                    self.missing_counts[stable_id] = 0
        
        return result
    
    def _upsample_result(self, result, original_size):
        """Upsample result to original size"""
        if result is None:
            return None
        
        if result.dim() == 3:
            result = result.unsqueeze(0)
            upsampled = torch.nn.functional.interpolate(result, size=original_size, mode='bilinear', align_corners=False)
            return upsampled.squeeze(0)
        else:
            return torch.nn.functional.interpolate(result, size=original_size, mode='bilinear', align_corners=False)
    
    def _create_empty_result(self, size):
        """Create empty result tensor"""
        h, w = size
        result = torch.zeros(self.max_obj_cnt + 1, h, w)
        result[0] = 1.0  # Background
        return result
    
    def _emergency_cleanup(self):
        """Emergency cleanup"""
        if self.processor:
            try:
                self.processor.clear_memory()
            except:
                pass
        
        self.id_history.clear()
        torch.cuda.empty_cache()
        gc.collect()

class Tracker:
    """Unified tracker combining geometric tracking with XMem segmentation"""
    
    def __init__(self, tracking_config: TrackingConfig, spring_config: SpringDamperConfig, 
                 camera_params: List[CameraParameters], xmem_config: dict = None):
        self.config = tracking_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Geometric tracking components
        self.projection = Projection3D(camera_params)
        self.physics = SpringDamperPhysics(spring_config, self.projection)
        self.detection_engine = DetectionEngine('yolov8m.pt', tracking_config.detection_confidence)
        self.performance_monitor = PerformanceMonitor()
        
        # Stable ID management
        self.id_manager = StableIDManager()
        self.frame_idx = 0
        
        # XMem integration
        self.xmem_processors = []
        if xmem_config is not None:
            for view_idx in range(tracking_config.num_cameras):
                try:
                    processor = XMemProcessor(
                        xmem_config, tracking_config.num_persons, self.device, view_idx
                    )
                    self.xmem_processors.append(processor)
                    print(f"Initialized XMem for view {view_idx}")
                except Exception as e:
                    print(f"Failed to initialize XMem for view {view_idx}: {e}")
                    self.xmem_processors.append(None)
        
        # Re-identification parameters
        self.reid_threshold = 0.6
        self.max_missing_frames = 20
        
    @property
    def tracks(self) -> List[StableTrack]:
        return self.id_manager.get_active_tracks()
    
    def initialize_from_config(self, initial_detections: Dict[int, Dict[int, Tuple]]) -> Dict[int, int]:
        """Initialize tracks from configuration"""
        track_mapping = {}
        
        for config_id, view_detections in initial_detections.items():
            centers_2d = {}
            for view_idx, bbox in view_detections.items():
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                centers_2d[view_idx] = np.array([center_x, center_y])
            
            if len(centers_2d) >= 2:
                world_pos = self.projection.triangulate_from_views(centers_2d)
                if world_pos is not None:
                    track = self.id_manager.create_track(
                        position_3d=world_pos,
                        size_3d=torch.tensor([0.8, 0.8, 1.8], device=self.device),
                        device=self.device
                    )
                    
                    for view_idx, bbox in view_detections.items():
                        track.bboxes[view_idx] = bbox
                        track.update_view_state(view_idx, bbox, 0.8)
                    
                    track_mapping[config_id] = track.id
        
        # Initialize XMem processors with ground truth
        if self.xmem_processors:
            self._initialize_xmem_with_ground_truth(initial_detections)
        
        return track_mapping
    
    def _initialize_xmem_with_ground_truth(self, initial_detections: Dict[int, Dict[int, Tuple]]):
        """Initialize XMem processors with ground truth masks"""
        for view_idx, processor in enumerate(self.xmem_processors):
            if processor is None:
                continue
            
            # Collect bboxes for this view
            view_bboxes = []
            for config_id, view_detections in initial_detections.items():
                if view_idx in view_detections:
                    view_bboxes.append(view_detections[view_idx])
            
            if view_bboxes:
                # Create dummy image for initialization (we'll replace with real image later)
                dummy_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
                
                try:
                    # Create initial mask
                    mask = processor.create_mask_from_img(dummy_image, view_bboxes)
                    print(f"Created initial mask for view {view_idx} with IDs: {np.unique(mask)}")
                except Exception as e:
                    print(f"Failed to create initial mask for view {view_idx}: {e}")
    
    def process_frame(self, frames: List[torch.Tensor], scales: List[float] = None) -> Dict[str, Any]:
        """Process frame with both geometric tracking and XMem"""
        start_time = self.performance_monitor.start_timer()
        
        # Convert frames to numpy for XMem
        frames_np = []
        for frame in frames:
            frame_np = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
            frames_np.append(frame_np)
        
        # Process with XMem if available
        xmem_results = []
        if self.xmem_processors:
            for view_idx, frame_np in enumerate(frames_np):
                if view_idx < len(self.xmem_processors) and self.xmem_processors[view_idx] is not None:
                    try:
                        if self.frame_idx == 0:
                            # Skip first frame prediction, will be initialized elsewhere
                            xmem_results.append(None)
                        else:
                            result = self.xmem_processors[view_idx].predict(frame_np)
                            xmem_results.append(result)
                    except Exception as e:
                        print(f"XMem error for view {view_idx}: {e}")
                        xmem_results.append(None)
                else:
                    xmem_results.append(None)
        
        # Geometric tracking
        all_detections = self.detection_engine.detect_persons_batch(frames)
        
        if scales:
            all_detections = self._rescale_detections_to_original(all_detections, scales)
        
        # Predict track positions
        self._predict_tracks()
        
        # Associate detections to tracks (enhanced with XMem)
        assignments = self._associate_detections_with_xmem(all_detections, xmem_results)
        
        # Update tracks
        self._update_tracks_with_xmem(assignments, all_detections, xmem_results)
        
        # Handle re-identification
        self._handle_reidentification(all_detections, assignments)
        
        # Initialize new tracks
        self._initialize_new_tracks(all_detections, assignments)
        
        # Cleanup tracks
        self._cleanup_tracks()
        
        total_time = self.performance_monitor.end_timer(start_time, 'total_frame')
        self.frame_idx += 1
        
        # Periodic memory cleanup
        if self.frame_idx % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()
        
        return {
            'detections': all_detections,
            'tracks': self.tracks,
            'frame_time': total_time
        }
    
    def add_xmem_masks(self, frames_np: List[np.ndarray], initial_detections: Dict[int, Dict[int, Tuple]]):
        """Add initial masks to XMem processors"""
        if not self.xmem_processors:
            return
        
        for view_idx, frame_np in enumerate(frames_np):
            if view_idx < len(self.xmem_processors) and self.xmem_processors[view_idx] is not None:
                # Collect bboxes for this view
                view_bboxes = []
                for config_id, view_detections in initial_detections.items():
                    if view_idx in view_detections:
                        view_bboxes.append(view_detections[view_idx])
                
                if view_bboxes:
                    try:
                        # Create and add mask
                        mask = self.xmem_processors[view_idx].create_mask_from_img(frame_np, view_bboxes)
                        result = self.xmem_processors[view_idx].add_mask(frame_np, mask)
                        print(f"Added initial mask to view {view_idx}")
                    except Exception as e:
                        print(f"Failed to add mask to view {view_idx}: {e}")
    
    def _associate_detections_with_xmem(self, all_detections: List[List[Dict]], 
                                       xmem_results: List[torch.Tensor]) -> Dict[int, Dict[int, int]]:
        """Enhanced detection association using XMem masks"""
        assignments = {}
        
        for view_idx, detections in enumerate(all_detections):
            if not detections:
                continue
            
            active_tracks = [t for t in self.tracks if t.is_active()]
            if not active_tracks:
                continue
            
            # Get XMem masks for this view
            xmem_mask = None
            if (view_idx < len(xmem_results) and xmem_results[view_idx] is not None):
                xmem_mask = xmem_results[view_idx]
            
            cost_matrix = np.full((len(active_tracks), len(detections)), 1.5)
            
            for track_idx, track in enumerate(active_tracks):
                predicted_bbox = track.bboxes.get(view_idx, None)
                predicted_centroid = track.get_predicted_centroid(view_idx)
                
                for det_idx, detection in enumerate(detections):
                    det_bbox = detection['bbox']
                    det_centroid = ((det_bbox[0] + det_bbox[2])/2, (det_bbox[1] + det_bbox[3])/2)
                    
                    # Geometric cost
                    geometric_cost = 1.0
                    if predicted_bbox is not None:
                        iou = self._compute_iou(predicted_bbox, det_bbox)
                        geometric_cost = 1.0 - iou
                    
                    # Centroid distance cost
                    centroid_cost = 1.0
                    if predicted_centroid is not None:
                        distance = np.sqrt((predicted_centroid[0] - det_centroid[0])**2 + 
                                         (predicted_centroid[1] - det_centroid[1])**2)
                        max_distance = 150
                        centroid_cost = min(1.0, distance / max_distance)
                    
                    # XMem mask cost
                    mask_cost = 0.5  # Default neutral cost
                    if xmem_mask is not None and track.id < xmem_mask.shape[0]:
                        mask_cost = self._compute_mask_detection_cost(
                            xmem_mask[track.id], det_bbox
                        )
                    
                    # Combined cost with XMem weighting
                    total_cost = 0.4 * geometric_cost + 0.3 * centroid_cost + 0.3 * mask_cost
                    
                    if total_cost < 1.2:
                        cost_matrix[track_idx, det_idx] = total_cost
            
            # Solve assignment
            if np.any(cost_matrix < 1.5):
                from scipy.optimize import linear_sum_assignment
                row_indices, col_indices = linear_sum_assignment(cost_matrix)
                
                for r, c in zip(row_indices, col_indices):
                    if cost_matrix[r, c] < 1.0:
                        track = active_tracks[r]
                        if track.id not in assignments:
                            assignments[track.id] = {}
                        assignments[track.id][view_idx] = c
        
        return assignments
    
    def _compute_mask_detection_cost(self, mask: torch.Tensor, bbox: Tuple) -> float:
        """Compute cost between XMem mask and detection bbox"""
        try:
            if mask.sum() < 50:  # Empty or very small mask
                return 1.0
            
            # Get mask centroid
            y_coords, x_coords = torch.where(mask > 0.5)
            if len(y_coords) == 0:
                return 1.0
            
            mask_centroid_x = x_coords.float().mean().item()
            mask_centroid_y = y_coords.float().mean().item()
            
            # Get detection centroid
            det_centroid_x = (bbox[0] + bbox[2]) / 2
            det_centroid_y = (bbox[1] + bbox[3]) / 2
            
            # Calculate distance
            distance = np.sqrt((mask_centroid_x - det_centroid_x)**2 + 
                             (mask_centroid_y - det_centroid_y)**2)
            
            # Normalize by image size (assume reasonable image dimensions)
            max_distance = 200
            cost = min(1.0, distance / max_distance)
            
            return cost
            
        except Exception:
            return 1.0  # High cost on error
    
    def _update_tracks_with_xmem(self, assignments: Dict[int, Dict[int, int]], 
                                all_detections: List[List[Dict]],
                                xmem_results: List[torch.Tensor]):
        """Update tracks with XMem mask information"""
        
        # Update assigned tracks
        for stable_id, view_assignments in assignments.items():
            track = self.id_manager.get_track_by_id(stable_id)
            if track is None:
                continue
            
            current_detections = {}
            
            for view_idx, det_idx in view_assignments.items():
                detection = all_detections[view_idx][det_idx]
                bbox = detection['bbox']
                track.bboxes[view_idx] = bbox
                current_detections[view_idx] = bbox
                
                # Extract XMem mask centroid if available
                mask_centroid = None
                if (view_idx < len(xmem_results) and xmem_results[view_idx] is not None and
                    track.id < xmem_results[view_idx].shape[0]):
                    
                    mask = xmem_results[view_idx][track.id]
                    if mask.sum() > 50:
                        y_coords, x_coords = torch.where(mask > 0.5)
                        if len(y_coords) > 0:
                            mask_centroid = (x_coords.float().mean().item(), 
                                           y_coords.float().mean().item())
                
                track.update_view_state(view_idx, bbox, detection['conf'], mask_centroid)
            
            # Update 3D position
            if len(current_detections) >= 2:
                centers_2d = {}
                for view_idx, bbox in current_detections.items():
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    centers_2d[view_idx] = np.array([center_x, center_y])
                
                world_pos = self.projection.triangulate_from_views(centers_2d)
                if world_pos is not None:
                    track.position_3d = world_pos.to(device=self.device)
            
            track.hits += 1
            track.time_since_update = 0
            track.confidence = min(1.0, track.confidence + 0.1)
            track.last_seen_frame = self.frame_idx
        
        # Update missing tracks
        assigned_ids = set(assignments.keys())
        for track in self.tracks:
            if track.id not in assigned_ids:
                track.time_since_update += 1
                track.confidence = max(0.0, track.confidence - 0.05)
                
                for view_idx in range(self.config.num_cameras):
                    track.miss_view(view_idx)
    
    def _predict_tracks(self):
        """Predict track positions"""
        for track in self.tracks:
            if track.is_active():
                # Update predicted positions for all views
                for view_idx in range(self.config.num_cameras):
                    projected_bbox = self.projection.project_3d_bbox(
                        track.position_3d, track.size_3d, view_idx
                    )
                    if projected_bbox is not None:
                        track.bboxes[view_idx] = projected_bbox
    
    def _handle_reidentification(self, all_detections: List[List[Dict]], assignments: Dict[int, Dict[int, int]]):
        """Handle re-identification using XMem and geometric info"""
        # Find unassigned detections
        used_detections = set()
        for view_assignments in assignments.values():
            for view_idx, det_idx in view_assignments.items():
                used_detections.add((view_idx, det_idx))
        
        # Find recently lost tracks
        lost_tracks = []
        for track in self.tracks:
            if (track.id not in assignments and 
                track.time_since_update > 1 and track.time_since_update < self.max_missing_frames and
                track.confidence > 0.3):
                lost_tracks.append(track)
        
        # Try to re-identify
        for track in lost_tracks:
            best_match = None
            best_score = 0.7
            
            for view_idx, detections in enumerate(all_detections):
                for det_idx, detection in enumerate(detections):
                    if (view_idx, det_idx) in used_detections:
                        continue
                    
                    score = self._calculate_reid_score(track, detection['bbox'], view_idx)
                    
                    if score > best_score:
                        best_score = score
                        best_match = (track.id, view_idx, det_idx)
            
            # Apply best match
            if best_match:
                track_id, view_idx, det_idx = best_match
                if track_id not in assignments:
                    assignments[track_id] = {}
                assignments[track_id][view_idx] = det_idx
                used_detections.add((view_idx, det_idx))
    
    def _calculate_reid_score(self, track: StableTrack, det_bbox: Tuple, view_idx: int) -> float:
        """Calculate re-identification score"""
        predicted_centroid = track.get_predicted_centroid(view_idx)
        if predicted_centroid is None:
            return 0.0
        
        det_centroid = ((det_bbox[0] + det_bbox[2])/2, (det_bbox[1] + det_bbox[3])/2)
        
        # Distance-based score
        distance = np.sqrt((predicted_centroid[0] - det_centroid[0])**2 + 
                          (predicted_centroid[1] - det_centroid[1])**2)
        
        max_movement = 200 * track.time_since_update
        if distance > max_movement:
            return 0.0
        
        score = max(0.0, 1.0 - distance / max_movement)
        
        # Boost for high confidence tracks
        if track.confidence > 0.7:
            score *= 1.2
        
        return min(1.0, score)
    
    def _initialize_new_tracks(self, all_detections: List[List[Dict]], assignments: Dict[int, Dict[int, int]]):
        """Initialize new tracks from unassigned detections"""
        if len(self.tracks) >= self.config.num_persons:
            return
        
        # Find unassigned detections
        used_detections = set()
        for view_assignments in assignments.values():
            for view_idx, det_idx in view_assignments.items():
                used_detections.add((view_idx, det_idx))
        
        # Collect candidates
        candidates = []
        for view_idx, detections in enumerate(all_detections):
            for det_idx, detection in enumerate(detections):
                if ((view_idx, det_idx) not in used_detections and 
                    detection['conf'] > 0.5):
                    
                    bbox = detection['bbox']
                    center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
                    candidates.append((view_idx, det_idx, center, detection))
        
        # Try to create new tracks
        for i, (view1, det1_idx, center1, det1) in enumerate(candidates):
            if len(self.tracks) >= self.config.num_persons:
                break
            
            for j, (view2, det2_idx, center2, det2) in enumerate(candidates[i+1:], i+1):
                if view1 == view2:
                    continue
                
                centers_2d = {view1: center1, view2: center2}
                world_pos = self.projection.triangulate_from_views(centers_2d)
                
                if world_pos is not None:
                    track = self.id_manager.create_track(
                        position_3d=world_pos,
                        size_3d=torch.tensor([0.8, 0.8, 1.8], device=self.device),
                        device=self.device
                    )
                    
                    track.bboxes[view1] = det1['bbox']
                    track.bboxes[view2] = det2['bbox']
                    track.update_view_state(view1, det1['bbox'], det1['conf'])
                    track.update_view_state(view2, det2['bbox'], det2['conf'])
                    track.last_seen_frame = self.frame_idx
                    
                    used_detections.add((view1, det1_idx))
                    used_detections.add((view2, det2_idx))
                    break
    
    def _cleanup_tracks(self):
        """Remove inactive tracks"""
        tracks_to_remove = []
        
        for track in self.tracks:
            track.age += 1
            
            if (track.time_since_update > self.max_missing_frames or 
                (track.confidence < 0.1 and track.age > 30)):
                tracks_to_remove.append(track)
        
        for track in tracks_to_remove:
            self.id_manager.remove_track(track)
    
    def _rescale_detections_to_original(self, all_detections: List[List[Dict]], scales: List[float]) -> List[List[Dict]]:
        """Rescale detections to original image size"""
        rescaled_detections = []
        
        for view_idx, detections in enumerate(all_detections):
            if view_idx < len(scales):
                scale = scales[view_idx]
                rescaled_view_detections = []
                
                for detection in detections:
                    bbox = detection['bbox']
                    x1, y1, x2, y2 = bbox
                    
                    rescaled_bbox = (
                        int(x1 * scale), int(y1 * scale),
                        int(x2 * scale), int(y2 * scale)
                    )
                    
                    rescaled_detection = {
                        'bbox': rescaled_bbox,
                        'conf': detection['conf']
                    }
                    rescaled_view_detections.append(rescaled_detection)
                
                rescaled_detections.append(rescaled_view_detections)
            else:
                rescaled_detections.append(detections)
        
        return rescaled_detections
    
    def _compute_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Compute IoU between two bounding boxes"""
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
    
    def keep_largest_connected_components(self, mask):
        """Keep largest connected components with optimization"""
        return keep_largest_connected_components(mask)
    
    def masks_to_boxes_with_ids(self, mask_tensor):
        """Extract bounding boxes with stable IDs"""
        if mask_tensor is None:
            return []
        
        try:
            if torch.is_tensor(mask_tensor):
                mask_np = mask_tensor.squeeze().cpu().numpy()
            else:
                mask_np = mask_tensor
            
            unique_values = np.unique(mask_np)
            unique_values = unique_values[unique_values != 0]
            
            bbox_list = []
            for unique_value in unique_values:
                binary_mask = (mask_np == unique_value)
                y_coords, x_coords = np.where(binary_mask)
                
                if len(y_coords) > 0:
                    min_x, max_x = np.min(x_coords), np.max(x_coords)
                    min_y, max_y = np.min(y_coords), np.max(y_coords)
                    
                    bbox = [int(unique_value), int(min_x), int(min_y), int(max_x), int(max_y)]
                    bbox_list.append(bbox)
            
            return bbox_list
            
        except Exception as e:
            print(f"Box extraction error: {e}")
            return []
    
    def create_mask_from_img(self, image, yolov7_bboxes, sam_checkpoint='./saves/mobile_sam.pt', model_type='vit_t', device='0'):
        """Create mask from image and bboxes - compatibility method"""
        if hasattr(self, 'xmem_processors') and len(self.xmem_processors) > 0:
            # Use first available XMem processor
            for processor in self.xmem_processors:
                if processor is not None and processor.initialized:
                    return processor.create_mask_from_img(image, yolov7_bboxes)
        
        # Fallback: create simple bbox masks
        result = np.zeros(image.shape[:2], dtype=np.uint8)
        for i, bbox in enumerate(yolov7_bboxes[:3]):
            x1, y1, x2, y2 = [max(0, int(coord)) for coord in bbox]
            x2 = min(x2, image.shape[1])
            y2 = min(y2, image.shape[0])
            if x2 > x1 and y2 > y1:
                result[y1:y2, x1:x2] = i + 1
        return result
    
    def predict(self, image):
        """Predict masks - compatibility method"""
        # This method is for compatibility with the original API
        # The actual prediction is handled by process_frame
        return np.zeros(image.shape[:2], dtype=np.uint8)
    
    def add_mask(self, image, mask):
        """Add mask - compatibility method"""
        # This is handled internally by the unified tracker
        return None
    
    def keep_largest_connected_components(self, mask):
        """Keep largest connected components with optimization"""
        return keep_largest_connected_components(mask)
    
    def masks_to_boxes_with_ids(self, mask_tensor):
        """Extract bounding boxes with stable IDs"""
        if mask_tensor is None:
            return []
        
        try:
            if torch.is_tensor(mask_tensor):
                mask_np = mask_tensor.squeeze().cpu().numpy()
            else:
                mask_np = mask_tensor
            
            unique_values = np.unique(mask_np)
            unique_values = unique_values[unique_values != 0]
            
            bbox_list = []
            for unique_value in unique_values:
                binary_mask = (mask_np == unique_value)
                y_coords, x_coords = np.where(binary_mask)
                
                if len(y_coords) > 0:
                    min_x, max_x = np.min(x_coords), np.max(x_coords)
                    min_y, max_y = np.min(y_coords), np.max(y_coords)
                    
                    bbox = [int(unique_value), int(min_x), int(min_y), int(max_x), int(max_y)]
                    bbox_list.append(bbox)
            
            return bbox_list
            
        except Exception as e:
            print(f"Box extraction error: {e}")
            return []