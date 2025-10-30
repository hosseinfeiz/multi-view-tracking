import torch
import torch.nn.functional as F
import numpy as np
import gc
import torch

def pad_divide_by(x, divisor):
    h, w = x.shape[-2:]
    new_h = ((h - 1) // divisor + 1) * divisor
    new_w = ((w - 1) // divisor + 1) * divisor
    
    pad_h = new_h - h
    pad_w = new_w - w
    
    pad = (0, pad_w, 0, pad_h)
    padded = F.pad(x, pad)
    
    return padded, (pad_h, pad_w)

def unpad(x, pad_info):
    pad_h, pad_w = pad_info
    if pad_h > 0:
        x = x[..., :-pad_h, :]
    if pad_w > 0:
        x = x[..., :, :-pad_w]
    return x

def aggregate(mask, dim=0):
    """Aggregate mask to create probability distribution"""
    # Add background channel
    background = 1.0 - mask.sum(dim=dim, keepdim=True)
    background = torch.clamp(background, 0, 1)
    
    # Concatenate background and object masks
    prob = torch.cat([background, mask], dim=dim)
    
    # Normalize to ensure valid probability distribution
    prob = prob / (prob.sum(dim=dim, keepdim=True) + 1e-8)
    
    return prob

def im_normalization(img):
    """ImageNet normalization"""
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device, dtype=img.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device, dtype=img.dtype).view(3, 1, 1)
    return (img - mean) / std

def image_to_torch_optimized(frame: np.ndarray, device='cuda'):
    """Convert numpy image to normalized torch tensor"""
    # frame: H*W*3 numpy array
    if len(frame.shape) == 2:
        frame = np.stack([frame, frame, frame], axis=2)
    
    frame = frame.transpose(2, 0, 1)
    frame = torch.from_numpy(frame).float().to(device) / 255
    frame_norm = im_normalization(frame)
    return frame_norm, frame

def index_numpy_to_one_hot_torch(mask, num_classes):
    """Convert indexed mask to one-hot tensor"""
    mask = torch.from_numpy(mask).long()
    return F.one_hot(mask, num_classes=num_classes).permute(2, 0, 1).float()

class InferenceCore:
    def __init__(self, network, config):
        self.config = config
        self.network = network
        
        # Optimized configuration
        self.mem_every = max(1, config.get('mem_every', 5))
        self.deep_update_every = config.get('deep_update_every', -1)
        self.enable_long_term = config.get('enable_long_term', False)
        
        # Memory management
        self.max_spatial_size = 320  # Limit spatial dimensions
        self.memory_cleanup_interval = 50
        self.frame_reset_interval = 200
        
        # Deep update synchronization
        self.deep_update_sync = (self.deep_update_every < 0)
        
        self.clear_memory()
        self.all_labels = None
        self.stable_id_mapping = {}
        
    def clear_memory(self):
        """Clear all memory states"""
        self.curr_ti = -1
        self.last_mem_ti = 0
        if not self.deep_update_sync:
            self.last_deep_update_ti = -self.deep_update_every
        
        # Use optimized memory manager
        from pose_estimation.track.memory_manager import MemoryManager
        self.memory = MemoryManager(config=self.config)
        
        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    def update_config(self, config):
        """Update configuration"""
        self.mem_every = max(1, config.get('mem_every', 5))
        self.deep_update_every = config.get('deep_update_every', -1)
        self.enable_long_term = config.get('enable_long_term', False)
        
        self.deep_update_sync = (self.deep_update_every < 0)
        if hasattr(self.memory, 'update_config'):
            self.memory.update_config(config)
    
    def set_all_labels(self, all_labels):
        """Set object labels with stable ID mapping"""
        self.all_labels = all_labels
        
        # Create stable ID mapping
        for i, label in enumerate(all_labels):
            if label not in self.stable_id_mapping:
                self.stable_id_mapping[label] = i + 1  # +1 for background
    
    def _downsample_if_needed(self, tensor, max_size=None):
        """Downsample tensor if too large"""
        if max_size is None:
            max_size = self.max_spatial_size
            
        if tensor is None:
            return tensor, 1.0
        
        h, w = tensor.shape[-2:]
        if max(h, w) <= max_size:
            return tensor, 1.0
        
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
            resized = F.interpolate(tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
            return resized.squeeze(0), 1.0 / scale
        else:
            resized = F.interpolate(tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
            return resized, 1.0 / scale
    
    def _upsample_result(self, result, scale, original_size):
        """Upsample result back to original size"""
        if result is None or scale == 1.0:
            return result
        
        if result.dim() == 3:
            result = result.unsqueeze(0)
            upsampled = F.interpolate(result, size=original_size, mode='bilinear', align_corners=False)
            return upsampled.squeeze(0)
        else:
            return F.interpolate(result, size=original_size, mode='bilinear', align_corners=False)
    
    def step(self, image, mask=None, valid_labels=None, end=False):
        """Process one frame with memory and ID management"""
        self.curr_ti += 1
        original_size = image.shape[-2:]
        
        # Downsample for processing
        image, scale = self._downsample_if_needed(image)
        if mask is not None:
            mask, _ = self._downsample_if_needed(mask)
        
        # Pad to multiples of 16
        image, self.pad = pad_divide_by(image, 16)
        image = image.unsqueeze(0)  # Add batch dimension
        
        # Determine processing flags
        is_mem_frame = ((self.curr_ti - self.last_mem_ti >= self.mem_every) or 
                       (mask is not None)) and (not end)
        need_segment = (self.curr_ti > 0) and (
            (valid_labels is None) or 
            (len(self.all_labels) != len(valid_labels))
        )
        is_deep_update = (
            (self.deep_update_sync and is_mem_frame) or
            (not self.deep_update_sync and 
             self.curr_ti - self.last_deep_update_ti >= self.deep_update_every)
        ) and (not end)
        is_normal_update = (not self.deep_update_sync or not is_deep_update) and (not end)
        
        try:
            # Encode key features
            key, shrinkage, selection, f16, f8, f4 = self.network.encode_key(
                image, 
                need_ek=(self.enable_long_term or need_segment), 
                need_sk=is_mem_frame
            )
            multi_scale_features = (f16, f8, f4)
            
            # Segment current frame if needed
            if need_segment:
                memory_readout = self.memory.match_memory(key, selection).unsqueeze(0)
                hidden, _, pred_prob_with_bg = self.network.segment(
                    multi_scale_features, memory_readout, 
                    self.memory.get_hidden(), h_out=is_normal_update, strip_bg=False
                )
                
                # Remove batch dimension
                pred_prob_with_bg = pred_prob_with_bg[0]
                pred_prob_no_bg = pred_prob_with_bg[1:]
                
                if is_normal_update:
                    self.memory.set_hidden(hidden)
            else:
                pred_prob_no_bg = pred_prob_with_bg = None
            
            # Use input mask if provided
            if mask is not None:
                mask, _ = pad_divide_by(mask, 16)
                
                if pred_prob_no_bg is not None:
                    # Make predicted mask consistent with input
                    mask_regions = (mask.sum(0) > 0.5)
                    pred_prob_no_bg[:, mask_regions] = 0
                    mask = mask.type_as(pred_prob_no_bg)
                    
                    if valid_labels is not None:
                        shift_by_one_non_labels = [
                            i for i in range(pred_prob_no_bg.shape[0]) 
                            if (i+1) not in valid_labels
                        ]
                        mask[shift_by_one_non_labels] = pred_prob_no_bg[shift_by_one_non_labels]
                
                pred_prob_with_bg = aggregate(mask, dim=0)
                
                # Create new hidden states
                self.memory.create_hidden_state(len(self.all_labels), key)
            
            # Save to memory if needed
            if is_mem_frame and pred_prob_with_bg is not None:
                value, hidden = self.network.encode_value(
                    image, f16, self.memory.get_hidden(), 
                    pred_prob_with_bg[1:].unsqueeze(0), is_deep_update=is_deep_update
                )
                
                self.memory.add_memory(
                    key, shrinkage, value, self.all_labels, 
                    selection=selection if self.enable_long_term else None
                )
                self.last_mem_ti = self.curr_ti
                
                if is_deep_update:
                    self.memory.set_hidden(hidden)
                    self.last_deep_update_ti = self.curr_ti
            
            # Prepare result
            if pred_prob_with_bg is None:
                # Create default result
                h, w = image.shape[-2:]
                pred_prob_with_bg = torch.zeros(len(self.all_labels) + 1, h, w, device=image.device)
                pred_prob_with_bg[0] = 1.0  # Background
            
            # Unpad and upsample result
            result = unpad(pred_prob_with_bg, self.pad)
            result = self._upsample_result(result, scale, original_size)
            
            # Apply ID stability
            result = self._apply_id_stability(result)
            
            # Periodic cleanup
            if self.curr_ti % self.memory_cleanup_interval == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
            # Full reset if needed
            if self.curr_ti % self.frame_reset_interval == 0:
                self._partial_memory_reset()
            
            return result
            
        except torch.cuda.OutOfMemoryError:
            print("CUDA OOM during inference, clearing memory...")
            self._emergency_cleanup()
            # Return fallback result
            h, w = original_size
            fallback = torch.zeros(len(self.all_labels) + 1, h, w)
            fallback[0] = 1.0
            return fallback
            
        except Exception as e:
            print(f"Inference error: {e}")
            # Return safe fallback
            h, w = original_size
            fallback = torch.zeros(len(self.all_labels) + 1, h, w)
            fallback[0] = 1.0
            return fallback
    
    def _apply_id_stability(self, result):
        """Apply stable ID mapping to results"""
        if result is None or not self.stable_id_mapping:
            return result
        
        # Create stable result tensor
        stable_result = torch.zeros_like(result)
        stable_result[0] = result[0]  # Background stays the same
        
        # Map channels according to stable IDs
        for original_label, stable_channel in self.stable_id_mapping.items():
            if stable_channel < result.shape[0] and stable_channel < stable_result.shape[0]:
                # Find best matching channel in result
                best_channel = self._find_best_matching_channel(result, original_label)
                if best_channel is not None and best_channel < result.shape[0]:
                    stable_result[stable_channel] = result[best_channel]
        
        return stable_result
    
    def _find_best_matching_channel(self, result, target_label):
        """Find channel that best matches target label"""
        # Simple heuristic: return the label index if valid
        if target_label < result.shape[0]:
            return target_label
        return None
    
    def _partial_memory_reset(self):
        """Partial memory reset to prevent accumulation"""
        if hasattr(self.memory, 'work_mem'):
            if hasattr(self.memory.work_mem, 'clear'):
                self.memory.work_mem.clear()
        
        # Reset hidden state
        self.memory.hidden = None
        
        # Reset counters
        self.last_mem_ti = self.curr_ti - 10  # Keep recent context
        
        torch.cuda.empty_cache()
        gc.collect()
    
    def _emergency_cleanup(self):
        """Emergency cleanup on OOM"""
        self.clear_memory()
        torch.cuda.empty_cache()
        gc.collect()
        
        # Reduce memory usage
        self.max_spatial_size = min(256, self.max_spatial_size)
        self.mem_every = max(10, self.mem_every * 2)


class StabilizedInferenceCore(InferenceCore):
    """Extended inference core with advanced ID stability"""
    
    def __init__(self, network, config):
        super().__init__(network, config)
        self.id_history = {}  # Track ID centroids and features
        self.missing_counts = {}  # Track missing frames per ID
        self.reid_threshold = 0.6
        self.max_missing_frames = 20
    
    def _apply_id_stability(self, result):
        """Enhanced ID stability with re-identification"""
        if result is None:
            return result
        
        # Update ID history
        self._update_id_history(result)
        
        # Handle re-identification
        result = self._handle_reidentification(result)
        
        # Apply stable mapping
        return super()._apply_id_stability(result)
    
    def _update_id_history(self, result):
        """Update ID history with current frame information"""
        for i in range(1, result.shape[0]):  # Skip background
            mask = result[i]
            if mask.sum() > 100:  # Valid mask
                # Calculate centroid
                y_coords, x_coords = torch.where(mask > 0.5)
                if len(y_coords) > 0:
                    centroid_y = y_coords.float().mean().item()
                    centroid_x = x_coords.float().mean().item()
                    
                    if i not in self.id_history:
                        self.id_history[i] = []
                    
                    self.id_history[i].append((centroid_x, centroid_y))
                    
                    # Keep only recent history
                    if len(self.id_history[i]) > 5:
                        self.id_history[i] = self.id_history[i][-5:]
                    
                    # Reset missing count
                    self.missing_counts[i] = 0
            else:
                # Increment missing count
                self.missing_counts[i] = self.missing_counts.get(i, 0) + 1
    
    def _handle_reidentification(self, result):
        """Handle re-identification of lost objects"""
        # Find missing objects
        missing_ids = []
        for obj_id, missing_count in self.missing_counts.items():
            if 5 < missing_count < self.max_missing_frames:
                missing_ids.append(obj_id)
        
        # Find active masks without stable IDs
        active_channels = []
        for i in range(1, result.shape[0]):
            if (result[i].sum() > 100 and 
                self.missing_counts.get(i, 0) == 0 and
                i not in self.id_history):
                active_channels.append(i)
        
        # Try to match missing IDs to active channels
        for missing_id in missing_ids:
            if not active_channels:
                break
            
            if missing_id in self.id_history and self.id_history[missing_id]:
                last_centroid = self.id_history[missing_id][-1]
                
                best_channel = None
                min_distance = float('inf')
                
                for channel in active_channels:
                    mask = result[channel]
                    y_coords, x_coords = torch.where(mask > 0.5)
                    if len(y_coords) > 0:
                        centroid_y = y_coords.float().mean().item()
                        centroid_x = x_coords.float().mean().item()
                        
                        distance = np.sqrt(
                            (centroid_x - last_centroid[0])**2 + 
                            (centroid_y - last_centroid[1])**2
                        )
                        
                        if distance < min_distance and distance < 100:
                            min_distance = distance
                            best_channel = channel
                
                # Re-assign if good match found
                if best_channel is not None:
                    if missing_id < result.shape[0]:
                        result[missing_id] = result[best_channel]
                        result[best_channel] = 0  # Clear original
                        active_channels.remove(best_channel)
                        self.missing_counts[missing_id] = 0
        
        return result