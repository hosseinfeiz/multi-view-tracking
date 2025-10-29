import cv2
import numpy as np
import torch

def keep_largest_connected_components(mask):
    """Keep largest connected components using OpenCV instead of skimage"""
    if mask is None:
        return None
    
    try:
        # Convert to numpy if tensor
        if torch.is_tensor(mask):
            mask_np = mask.squeeze().cpu().numpy()
            is_tensor = True
            original_device = mask.device
        else:
            mask_np = mask
            is_tensor = False
        
        # Get unique values (object IDs)
        unique_values = np.unique(mask_np)
        unique_values = unique_values[unique_values != 0]  # Skip background
        
        new_mask = np.zeros_like(mask_np)
        
        for class_value in unique_values:
            # Create binary mask for this class
            binary_mask = (mask_np == class_value).astype(np.uint8)
            
            if np.sum(binary_mask) < 100:  # Skip very small masks
                continue
            
            # Adaptive morphological operations
            h, w = binary_mask.shape
            kernel_size = max(3, min(11, min(h, w) // 30))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            # Close holes and remove noise
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            
            # Find connected components using OpenCV
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                binary_mask, connectivity=8
            )
            
            if num_labels > 1:  # Found components (label 0 is background)
                # Find largest component (excluding background label 0)
                component_areas = stats[1:, cv2.CC_STAT_AREA]  # Skip background
                largest_component_idx = 1 + np.argmax(component_areas)  # +1 to account for background
                
                # Keep only the largest component
                largest_component_mask = (labels == largest_component_idx)
                new_mask[largest_component_mask] = class_value
        
        # Convert back to tensor if needed
        if is_tensor:
            return torch.from_numpy(new_mask).unsqueeze(0).to(original_device)
        else:
            return new_mask
            
    except Exception as e:
        print(f"Connected components filtering error: {e}")
        return mask  # Return original mask on error