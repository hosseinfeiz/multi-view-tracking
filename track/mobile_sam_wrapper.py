import torch
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import torch.nn.functional as F

try:
    from mobile_sam import SamPredictor, sam_model_registry
    MOBILE_SAM_AVAILABLE = True
except ImportError:
    MOBILE_SAM_AVAILABLE = False

class MobileSAMWrapper:
    def __init__(self, checkpoint_path: str = 'mobile_sam.pt', model_type: str = 'vit_t', device: str = 'cuda'):
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.device = device
        self.predictor = None
        self.max_input_size = 512
        self.batch_cache = {}
        self.mask_quality_cache = {}
        
        if MOBILE_SAM_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        try:
            sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
            sam.to(device=self.device)
            sam.eval()
            self.predictor = SamPredictor(sam)
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Failed to load MobileSAM: {e}")
            self.predictor = None
    
    def segment_from_boxes(self, image: np.ndarray, bboxes: List[Tuple[int, int, int, int]], 
                          track_ids: List[int] = None) -> List[np.ndarray]:
        if not MOBILE_SAM_AVAILABLE or self.predictor is None or not bboxes:
            return self._fallback_segmentation(image, bboxes)
        
        if len(bboxes) > 4:
            bboxes = bboxes[:4]
        
        h, w = image.shape[:2]
        original_size = (h, w)
        
        processed_image, scale, bboxes_scaled = self._preprocess_for_sam(image, bboxes)
        
        try:
            self.predictor.set_image(processed_image)
            
            input_boxes = torch.tensor(bboxes_scaled, device=self.predictor.device)
            transformed_boxes = self.predictor.transform.apply_boxes_torch(input_boxes, processed_image.shape[:2])
            
            masks, scores, _ = self.predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            
            processed_masks = self._postprocess_masks(masks, scores, original_size, scale, track_ids)
            
            del input_boxes, transformed_boxes, masks
            torch.cuda.empty_cache()
            
            return processed_masks
            
        except Exception as e:
            print(f"SAM segmentation failed: {e}")
            return self._fallback_segmentation(image, bboxes)
    
    def segment_from_boxes_batch(self, images: List[np.ndarray], 
                                bboxes_list: List[List[Tuple[int, int, int, int]]], 
                                track_ids_list: List[List[int]] = None) -> List[List[np.ndarray]]:
        if not MOBILE_SAM_AVAILABLE or self.predictor is None:
            return [self._fallback_segmentation(img, bboxes) for img, bboxes in zip(images, bboxes_list)]
        
        all_masks = []
        for i, (image, bboxes) in enumerate(zip(images, bboxes_list)):
            track_ids = track_ids_list[i] if track_ids_list else None
            masks = self.segment_from_boxes(image, bboxes, track_ids)
            all_masks.append(masks)
        
        return all_masks
    
    def _preprocess_for_sam(self, image: np.ndarray, bboxes: List[Tuple[int, int, int, int]]) -> Tuple[np.ndarray, float, List[Tuple[int, int, int, int]]]:
        h, w = image.shape[:2]
        
        if max(h, w) > self.max_input_size:
            scale = self.max_input_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            bboxes_scaled = []
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                bboxes_scaled.append((
                    max(0, int(x1 * scale)),
                    max(0, int(y1 * scale)),
                    min(new_w-1, int(x2 * scale)),
                    min(new_h-1, int(y2 * scale))
                ))
            
            return image_resized, scale, bboxes_scaled
        else:
            return image, 1.0, bboxes
    
    def _postprocess_masks(self, masks: torch.Tensor, scores: torch.Tensor, 
                          original_size: Tuple[int, int], scale: float, 
                          track_ids: List[int] = None) -> List[np.ndarray]:
        processed_masks = []
        
        for i, mask in enumerate(masks):
            mask_np = mask.cpu().squeeze().numpy().astype(np.uint8)
            
            if scale != 1.0:
                mask_np = cv2.resize(mask_np, (original_size[1], original_size[0]), 
                                   interpolation=cv2.INTER_NEAREST)
            
            mask_np = (mask_np * 255).astype(np.uint8)
            
            mask_refined = self._refine_mask(mask_np)
            
            if track_ids and i < len(track_ids):
                quality_score = scores[i].item() if scores is not None else 0.5
                self.mask_quality_cache[track_ids[i]] = quality_score
            
            processed_masks.append(mask_refined)
        
        return processed_masks
    
    def _refine_mask(self, mask: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask_refined = np.zeros_like(mask)
            cv2.fillPoly(mask_refined, [largest_contour], 255)
            return mask_refined
        
        return mask_cleaned
    
    def _fallback_segmentation(self, image: np.ndarray, bboxes: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
        masks = []
        for bbox in bboxes:
            x1, y1, x2, y2 = [max(0, int(coord)) for coord in bbox]
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            width, height = x2 - x1, y2 - y1
            
            cv2.ellipse(mask, (center_x, center_y), (width//2, height//2), 0, 0, 360, 255, -1)
            masks.append(mask)
        return masks
    
    def get_mask_quality(self, track_id: int) -> float:
        return self.mask_quality_cache.get(track_id, 0.5)
    
    def segment_with_points(self, image: np.ndarray, points: List[Tuple[int, int]], 
                           labels: List[int], bbox: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        if not MOBILE_SAM_AVAILABLE or self.predictor is None:
            return self._fallback_segmentation(image, [bbox] if bbox else [(0, 0, image.shape[1], image.shape[0])])[0]
        
        try:
            self.predictor.set_image(image)
            
            point_coords = torch.tensor(points, device=self.predictor.device, dtype=torch.float32)
            point_labels = torch.tensor(labels, device=self.predictor.device)
            
            input_box = None
            if bbox:
                input_box = torch.tensor([bbox], device=self.predictor.device, dtype=torch.float32)
                input_box = self.predictor.transform.apply_boxes_torch(input_box, image.shape[:2])
            
            masks, _, _ = self.predictor.predict_torch(
                point_coords=point_coords.unsqueeze(0),
                point_labels=point_labels.unsqueeze(0),
                boxes=input_box,
                multimask_output=False,
            )
            
            mask = masks[0].cpu().squeeze().numpy().astype(np.uint8) * 255
            return self._refine_mask(mask)
            
        except Exception as e:
            print(f"Point-based segmentation failed: {e}")
            return self._fallback_segmentation(image, [bbox] if bbox else [(0, 0, image.shape[1], image.shape[0])])[0]
    
    def clear_cache(self):
        self.batch_cache.clear()
        self.mask_quality_cache.clear()
        torch.cuda.empty_cache()
    
    @property
    def is_available(self) -> bool:
        return MOBILE_SAM_AVAILABLE and self.predictor is not None