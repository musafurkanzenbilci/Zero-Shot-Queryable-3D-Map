from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class SegmentMask:
    """Single segmentation mask with metadata."""
    mask: np.ndarray  # HxW binary mask
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    area: int
    confidence: float
    mask_id: int


@dataclass
class FrameSegmentation:
    """Segmentation results for a single frame"""
    masks: List[SegmentMask]
    image_shape: Tuple[int, int]  # H, W
    
    def get_mask_at_pixel(self, u: int, v: int) -> Optional[SegmentMask]:
        """Get the mask containing a pixel"""
        for mask in self.masks:
            if mask.mask[v, u]:
                return mask
        return None
    
    def get_all_masks_at_pixel(self, u: int, v: int) -> List[SegmentMask]:
        """Get all masks containing a pixel"""
        return [m for m in self.masks if m.mask[v, u]]


class FastSAMSegmenter:
    """
    FastSAM-based instance segmentation.
    
    FastSAM uses a YOLOv8-based architecture to generate class-agnostic instance masks.
    """
    
    def __init__(self, model_name: str = "FastSAM-s", device: Optional[str] = None,
                 config: Optional[dict] = None):
        self.config = config or {}
        seg_config = self.config.get('segmentation', {})
        
        # Config parameters
        self.conf_threshold = seg_config.get('conf_threshold', 0.4)
        self.iou_threshold = seg_config.get('iou_threshold', 0.9)
        self.max_masks = seg_config.get('max_masks_per_frame', 100)
        
        # Determine device
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device
        
        model_paths = {
            "FastSAM-s": "FastSAM-s.pt",
            "FastSAM-x": "FastSAM-x.pt"
        }
        
        self.model_name = model_name
        self.model = None
        self._model_loaded = False
        
        print(f"[FastSAMSegmenter] Initialized")
        print(f"  Model: {model_name}")
        print(f"  Device: {device}")
        print(f"  Confidence threshold: {self.conf_threshold}")
    
    def _ensure_model_loaded(self):
        if self._model_loaded:
            return
        
        try:
            from ultralytics import YOLO

            # Try to load from local path first, then download
            model_path = self.model_name + ".pt"
            
            # Check common locations
            local_paths = [
                Path(__file__).parent.parent / "models" / model_path,
                Path.home() / ".cache" / "fastsam" / model_path,
            ]
            
            model_file = None
            for p in local_paths:
                if p.exists():
                    model_file = str(p)
                    break
            
            if model_file is None:
                # Use ultralytics to download
                print(f"  Downloading {self.model_name}...")
                model_file = model_path
            
            self.model = YOLO(model_file)
            self._model_loaded = True
            print(f"  Model loaded successfully")
            
        except ImportError:
            raise ImportError(
                "FastSAM requires ultralytics. Install with: pip install ultralytics"
            )
    
    def segment_frame(self, image: np.ndarray) -> FrameSegmentation:
        self._ensure_model_loaded()
        
        H, W = image.shape[:2]
        
        # Run FastSAM inference
        results = self.model(
            image,
            device=self.device,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            retina_masks=True,
            verbose=False
        )
        
        masks = []
        
        if results and len(results) > 0 and results[0].masks is not None:
            result = results[0]
            
            # Get masks and boxes
            mask_data = result.masks.data.cpu().numpy()  # NxHxW
            
            # Get confidence scores if available
            if result.boxes is not None and result.boxes.conf is not None:
                confidences = result.boxes.conf.cpu().numpy()
            else:
                confidences = np.ones(len(mask_data))
            
            # Get bounding boxes if available
            if result.boxes is not None and result.boxes.xyxy is not None:
                boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            else:
                boxes = [None] * len(mask_data)
            
            for i, (mask, conf) in enumerate(zip(mask_data, confidences)):
                if i >= self.max_masks:
                    break
                
                # Resize mask if needed
                if mask.shape[0] != H or mask.shape[1] != W:
                    import cv2
                    mask = cv2.resize(mask.astype(np.uint8), (W, H), 
                                     interpolation=cv2.INTER_NEAREST)
                
                mask_binary = mask > 0.5
                area = mask_binary.sum()
                
                # Skip very small masks
                if area < 100:
                    continue
                
                # Compute bbox from mask if not provided
                if boxes[i] is not None:
                    bbox = tuple(boxes[i])
                else:
                    ys, xs = np.where(mask_binary)
                    if len(xs) > 0:
                        bbox = (xs.min(), ys.min(), xs.max(), ys.max())
                    else:
                        continue
                
                masks.append(SegmentMask(
                    mask=mask_binary,
                    bbox=bbox,
                    area=area,
                    confidence=float(conf),
                    mask_id=i
                ))
        
        # Sort by area (largest first)
        masks.sort(key=lambda m: m.area, reverse=True)
        
        return FrameSegmentation(masks=masks, image_shape=(H, W))
    
    def get_mask_crops(self, image: np.ndarray, 
                       segmentation: FrameSegmentation) -> List[Tuple[np.ndarray, SegmentMask]]:
        crops = []
        
        for mask in segmentation.masks:
            x1, y1, x2, y2 = mask.bbox
            
            # Ensure bounds are valid
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Crop image and mask
            crop = image[y1:y2, x1:x2].copy()
            mask_crop = mask.mask[y1:y2, x1:x2]
            
            # Apply mask (set background to neutral gray)
            # This helps CLIP focus on the object
            bg_color = np.array([128, 128, 128], dtype=np.uint8)
            crop[~mask_crop] = bg_color
            
            crops.append((crop, mask))
        
        return crops


def test_segmentation():
    from pathlib import Path

    import cv2

    # Load a test image
    dataset_path = Path(__file__).parent.parent / "data" / "rgbd_dataset_freiburg3_long_office_household"
    rgb_path = dataset_path / "rgb" / "1341847980.722988.png"
    
    image = cv2.imread(str(rgb_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print(f"Loaded image: {image.shape}")
    
    # Test FastSAM
    print("\nTesting FastSAMSegmenter...")
    try:
        segmenter = FastSAMSegmenter(model_name="FastSAM-s")
        result = segmenter.segment_frame(image)
        print(f"  Generated {len(result.masks)} instance masks")
        
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt

        def show_masks_on_image(image, masks):
            plt.figure(figsize=(8, 8))
            plt.imshow(image)
            ax = plt.gca()
            for m in masks:
                # Show mask as transparent overlay
                colored_mask = np.zeros((*m.mask.shape, 4), dtype=np.float32)
                colored_mask[..., 0] = 1.0  # Red channel
                colored_mask[..., 3] = 0.25 * m.mask  # Alpha channel, only where mask==1
                plt.imshow(colored_mask)
                # Draw bounding box
                x1, y1, x2, y2 = m.bbox
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, 
                                        edgecolor='lime', facecolor='none')
                ax.add_patch(rect)
            plt.title(f"Segmentation masks ({len(masks)})")
            plt.axis('off')
            plt.show()

        show_masks_on_image(image, result.masks)
        # Show mask statistics
        areas = [m.area for m in result.masks]
        print(f"  Mask areas: min={min(areas)}, max={max(areas)}, mean={np.mean(areas):.0f}")

        mask_crops = segmenter.get_mask_crops(image, result)
        show_masks_on_image(mask_crops[10][0], [mask_crops[10][1]])
        
    except Exception as e:
        print(f"  FastSAM not available: {e}")
    
    print("\n✅ Segmentation test completed!")


if __name__ == "__main__":
    test_segmentation()
