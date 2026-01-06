from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class ImageFeatures:
    global_embedding: np.ndarray  # 1xD global feature
    local_embeddings: np.ndarray  # NxD local features (per mask)
    fused_embeddings: np.ndarray  # NxD fused features
    mask_ids: List[int]  # Corresponding mask IDs


class CLIPFeatureExtractor:
    """
    CLIP-based feature extraction.
    
    Implements the ConceptFusion-style pixel alignment:
    1. Extract global feature f_G for entire image
    2. Extract local features f_L for each masked region
    3. Fuse features: f_P = o * f_G + (1-o) * f_L weighted by similarity
    """
    
    def __init__(self, model_name: str = "ViT-B/32", device: Optional[str] = None, config: Optional[dict] = None):
        self.config = config or {}
        feat_config = self.config.get('features', {})
        
        # Fusion weights from config
        self.global_weight = feat_config.get('global_weight', 0.3)
        self.local_weight = feat_config.get('local_weight', 0.7)
        
        # Determine device
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = device
        
        self.model_name = model_name
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self._model_loaded = False
        self.embedding_dim = 512  # Default, updated after loading
        
        print(f"[CLIPFeatureExtractor] Initialized")
        print(f"  Model: {model_name}")
        print(f"  Device: {device}")
        print(f"  Fusion weights: global={self.global_weight}, local={self.local_weight}")
    
    def _ensure_model_loaded(self):
        if self._model_loaded:
            return
        
        try:
            import clip
            
            print(f"  Loading CLIP model {self.model_name}...")
            self.model, self.preprocess = clip.load(self.model_name, device=self.device)
            self.model.eval()
            
            # Get embedding dimension
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224, device=self.device)
                self.embedding_dim = self.model.encode_image(dummy).shape[-1]
            
            self._model_loaded = True
            print(f"  CLIP model loaded (embedding dim: {self.embedding_dim})")
            
        except ImportError:
            print("  OpenAI CLIP not available, trying transformers...")
            self._load_transformers_clip()
    
    def _load_transformers_clip(self):
        """Load CLIP via Hugging Face transformers."""
        from transformers import CLIPModel, CLIPProcessor
        
        model_mapping = {
            "ViT-B/32": "openai/clip-vit-base-patch32",
            "ViT-B/16": "openai/clip-vit-base-patch16",
            "ViT-L/14": "openai/clip-vit-large-patch14",
        }
        
        hf_model = model_mapping.get(self.model_name, "openai/clip-vit-base-patch32")
        
        print(f"  Loading {hf_model} from transformers...")
        self.model = CLIPModel.from_pretrained(hf_model).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(hf_model)
        self.model.eval()
        
        self.embedding_dim = self.model.config.projection_dim
        self._model_loaded = True
        self._use_transformers = True
        print(f"  Model loaded (embedding dim: {self.embedding_dim})")
    
    def _preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if hasattr(self, '_use_transformers') and self._use_transformers:
            inputs = self.processor(images=image, return_tensors="pt")
            return inputs['pixel_values'].to(self.device)
        else:
            return self.preprocess(image).unsqueeze(0).to(self.device)
    
    @torch.no_grad()
    def encode_image(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        self._ensure_model_loaded()
        
        image_input = self._preprocess_image(image)
        
        if hasattr(self, '_use_transformers') and self._use_transformers:
            embedding = self.model.get_image_features(pixel_values=image_input)
        else:
            embedding = self.model.encode_image(image_input)
        
        # Normalize
        embedding = F.normalize(embedding, dim=-1)
        
        return embedding.cpu().numpy().squeeze()
    
    @torch.no_grad()
    def encode_images_batch(self, images: List[Union[np.ndarray, Image.Image]], batch_size: int = 32) -> np.ndarray:
        self._ensure_model_loaded()
        
        all_embeddings = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            # Preprocess batch
            if hasattr(self, '_use_transformers') and self._use_transformers:
                pil_images = [Image.fromarray(img) if isinstance(img, np.ndarray) else img 
                             for img in batch]
                inputs = self.processor(images=pil_images, return_tensors="pt", padding=True)
                image_input = inputs['pixel_values'].to(self.device)
                embeddings = self.model.get_image_features(pixel_values=image_input)
            else:
                tensors = [self._preprocess_image(img) for img in batch]
                image_input = torch.cat(tensors, dim=0)
                embeddings = self.model.encode_image(image_input)
            
            # Normalize
            embeddings = F.normalize(embeddings, dim=-1)
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    @torch.no_grad()
    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        self._ensure_model_loaded()
        
        if isinstance(text, str):
            text = [text]
        
        if hasattr(self, '_use_transformers') and self._use_transformers:
            inputs = self.processor(text=text, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items() if k != 'pixel_values'}
            embeddings = self.model.get_text_features(**inputs)
        else:
            import clip
            text_tokens = clip.tokenize(text).to(self.device)
            embeddings = self.model.encode_text(text_tokens)
        
        # Normalize
        embeddings = F.normalize(embeddings, dim=-1)
        result = embeddings.cpu().numpy()
        
        return result.squeeze() if len(text) == 1 else result
    
    def extract_frame_features(self, image: np.ndarray, mask_crops: List[Tuple[np.ndarray, any]]) -> ImageFeatures:
        # Extract global feature
        global_embedding = self.encode_image(image)
        
        if len(mask_crops) == 0:
            return ImageFeatures(
                global_embedding=global_embedding,
                local_embeddings=np.empty((0, self.embedding_dim)),
                fused_embeddings=np.empty((0, self.embedding_dim)),
                mask_ids=[]
            )
        
        # Extract local features for each crop
        crops = [crop for crop, _ in mask_crops]
        local_embeddings = self.encode_images_batch(crops)
        
        # Compute fused features using similarity-based weighting
        # From ConceptFusion: higher similarity = more global context
        similarities = np.dot(local_embeddings, global_embedding)
        
        # Adaptive fusion based on similarity
        # Objects similar to global context get more global weight
        weight = self.global_weight * (1 + similarities.reshape(-1, 1)) / 2
        weight = np.clip(weight, 0.1, 0.9)  # Keep some of both
        
        fused_embeddings = weight * global_embedding + (1 - weight) * local_embeddings
        
        # Renormalize fused embeddings
        fused_embeddings = fused_embeddings / np.linalg.norm(fused_embeddings, axis=1, keepdims=True)
        
        mask_ids = [mask.mask_id for _, mask in mask_crops]
        
        return ImageFeatures(
            global_embedding=global_embedding,
            local_embeddings=local_embeddings,
            fused_embeddings=fused_embeddings,
            mask_ids=mask_ids
        )


class MobileCLIPFeatureExtractor(CLIPFeatureExtractor):
    """
    Apple MobileCLIP for efficient inference on Apple Silicon.
    
    MobileCLIP offers better latency on M-series chips while
    maintaining competitive accuracy with standard CLIP.
    
    Reference: https://github.com/apple/ml-mobileclip
    """
    
    def __init__(self, model_name: str = "mobileclip_s1", device: Optional[str] = None,
                 config: Optional[dict] = None):
        """
        Initialize MobileCLIP.
        
        Args:
            model_name: One of "mobileclip_s0", "mobileclip_s1", "mobileclip_s2"
        """
        # Don't call parent init, we override everything
        self.config = config or {}
        feat_config = self.config.get('features', {})
        
        self.global_weight = feat_config.get('global_weight', 0.3)
        self.local_weight = feat_config.get('local_weight', 0.7)
        
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device
        
        self.model_name = model_name
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self._model_loaded = False
        self.embedding_dim = 512
        
        print(f"[MobileCLIPFeatureExtractor] Initialized")
        print(f"  Model: {model_name}")
        print(f"  Device: {device}")
    
    def _ensure_model_loaded(self):
        """Load MobileCLIP model."""
        if self._model_loaded:
            return
        
        try:
            import mobileclip
            
            model_configs = {
                "mobileclip_s0": "mobileclip_s0",
                "mobileclip_s1": "mobileclip_s1",
                "mobileclip_s2": "mobileclip_s2",
            }
            
            config_name = model_configs.get(self.model_name, "mobileclip_s1")
            
            print(f"  Loading MobileCLIP {config_name}...")
            self.model, _, self.preprocess = mobileclip.create_model_and_transforms(
                config_name, pretrained=f'checkpoints/{config_name}.pt'
            )
            self.tokenizer = mobileclip.get_tokenizer(config_name)
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self._model_loaded = True
            print(f"  MobileCLIP loaded successfully")
            
        except (ImportError, Exception) as e:
            print(f"  MobileCLIP not available ({e}), falling back to standard CLIP")
            # Fall back to standard CLIP via transformers
            self._load_transformers_clip()


def create_feature_extractor(model_type: str = "clip", 
                             config: Optional[dict] = None) -> CLIPFeatureExtractor:
    """
    Factory function to create appropriate feature extractor.
    
    Args:
        model_type: "clip" for OpenAI CLIP, "mobileclip" for Apple MobileCLIP
        config: Optional configuration
        
    Returns:
        Feature extractor instance
    """
    feat_config = (config or {}).get('features', {})
    model_name = feat_config.get('model', 'ViT-B/32')
    
    if model_type == "mobileclip" or "mobileclip" in model_name.lower():
        return MobileCLIPFeatureExtractor(model_name=model_name, config=config)
    else:
        return CLIPFeatureExtractor(model_name=model_name, config=config)


def test_feature_extractor():
    """Test feature extraction."""
    from pathlib import Path

    import cv2

    # Load test image
    dataset_path = Path(__file__).parent.parent / "data" / "rgbd_dataset_freiburg3_long_office_household"
    rgb_path = dataset_path / "rgb" / "1341847980.722988.png"
    
    image = cv2.imread(str(rgb_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print(f"Loaded image: {image.shape}")
    
    # Test CLIP feature extractor
    print("\nTesting CLIPFeatureExtractor...")
    extractor = CLIPFeatureExtractor(model_name="ViT-B/32")
    
    # Test image encoding
    embedding = extractor.encode_image(image)
    print(f"  Image embedding shape: {embedding.shape}")
    print(f"  Embedding norm: {np.linalg.norm(embedding):.4f}")
    
    # Test text encoding
    text_embedding = extractor.encode_text("a photo of an office desk")
    print(f"  Text embedding shape: {text_embedding.shape}")
    
    # Test similarity
    similarity = np.dot(embedding, text_embedding)
    print(f"  Image-text similarity: {similarity:.4f}")
    
    # Test with some queries
    queries = [
        "a yellow dice",
        "a purple office chair", 
        "water bottles",
        "a globe",
        "a computer monitor"
    ]
    
    print("\nQuery similarities:")
    similarities = []
    query_embeddings = []
    for query in queries:
        text_emb = extractor.encode_text(query)
        query_embeddings.append(text_emb)
        sim = np.dot(embedding, text_emb)
        similarities.append(sim)
        print(f"  '{query}': {sim:.4f}")

    # Show the segmentation of the most similar embedding on the image
    print("\nFinding segmentation mask most similar to query...")

    from segmentation import FastSAMSegmenter

    segmenter = FastSAMSegmenter(model_name="FastSAM-s")
    segmentation = segmenter.segment_frame(image)
    print(f"  Generated {len(segmentation.masks)} instance masks via FastSAM.")

    # Encode CLIP embedding for each segment
    crops = segmenter.get_mask_crops(image, segmentation)
    mask_embeddings = []
    for crop, mask in crops:
        mask_embeddings.append(extractor.encode_image(crop))

    # Find the most similar mask for the most similar query
    best_query_idx = int(np.argmax(similarities))
    best_query = queries[best_query_idx]
    best_query_emb = query_embeddings[best_query_idx]
    print(f"\nBest matching query: '{best_query}' (similarity={similarities[best_query_idx]:.4f})")

    mask_scores = [np.dot(mask_emb, best_query_emb) for mask_emb in mask_embeddings]
    if len(mask_scores) > 0:
        best_mask_idx = int(np.argmax(mask_scores))
        best_mask = crops[best_mask_idx][1]
        print(f"  Best segmentation mask for '{best_query}': mask_id={best_mask.mask_id}, score={mask_scores[best_mask_idx]:.4f}")

        # Plot the mask on the image
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        ax = plt.gca()
        # show mask as transparent overlay
        colored_mask = np.zeros((*best_mask.mask.shape, 4), dtype=np.float32)
        colored_mask[..., 0] = 1.0  # Red channel
        colored_mask[..., 3] = 0.30 * best_mask.mask  # Alpha
        plt.imshow(colored_mask)
        # Overlay bbox
        x1, y1, x2, y2 = best_mask.bbox
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        plt.title(f"Most similar segment for: '{best_query}'")
        plt.axis("off")
        plt.show()
    else:
        print("Warning: No mask segments produced.")
    
    print("\n✅ Feature extraction test completed!")


if __name__ == "__main__":
    test_feature_extractor()
