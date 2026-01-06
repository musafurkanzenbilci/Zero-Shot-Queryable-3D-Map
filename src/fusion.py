import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from tqdm import tqdm

from data_loader import Frame, CameraIntrinsics, TUMDatasetLoader
from point_cloud import PointCloudFrame, SemanticPointCloud, PointCloudGenerator
from segmentation import FastSAMSegmenter, FrameSegmentation
from feature_extractor import CLIPFeatureExtractor, ImageFeatures

"""
Pixel Fusion Module

Implements the core ConceptFusion algorithm to fuse 2D semantic features
into 3D point cloud embeddings.

For each 3D point, we:
1. Back-project to all frames where it was visible
2. Look up the mask containing that pixel
3. Aggregate semantic embeddings weighted by visibility/confidence
"""

@dataclass
class ProcessedFrame:
    """Frame with all extracted information."""
    frame_idx: int
    timestamp: float
    
    # Point cloud for this frame
    point_cloud: PointCloudFrame
    
    # Segmentation
    segmentation: FrameSegmentation
    
    # Features
    features: ImageFeatures
    
    # Camera pose (4x4 matrix)
    pose_matrix: np.ndarray


class SemanticFusion:
    """
    Fuses 2D semantic features into 3D point cloud.
    
    Implements multi-view feature aggregation following ConceptFusion:
    - Each 3D point accumulates features from all frames where visible
    - Features are weighted by depth confidence and mask uniqueness
    - Final embedding is L2-normalized
    """
    
    def __init__(self, intrinsics: CameraIntrinsics, config: Optional[dict] = None):
        self.intrinsics = intrinsics
        self.config = config or {}
        
        # Processing parameters
        proc_config = self.config.get('processing', {})
        self.max_depth = proc_config.get('max_depth', 8.0)
        self.min_depth = proc_config.get('min_depth', 0.1)
        
        print(f"[SemanticFusion] Initialized")
    
    def pixel_to_embedding(self, pixel_indices: np.ndarray, 
                           segmentation: FrameSegmentation,
                           features: ImageFeatures) -> np.ndarray:
        """
        Map pixel coordinates to semantic embeddings.
        """
        N = len(pixel_indices)
        D = features.global_embedding.shape[0]
        
        embeddings = np.zeros((N, D))
        
        # Create mask ID lookup map for efficiency
        mask_id_to_idx = {mid: idx for idx, mid in enumerate(features.mask_ids)}
        
        # Build a pixel-to-mask lookup (H x W -> mask_idx or -1)
        H, W = segmentation.image_shape
        pixel_mask_map = np.full((H, W), -1, dtype=np.int32)
        
        # Fill in mask regions (later masks override earlier - smallest on top)
        for mask in reversed(segmentation.masks):
            pixel_mask_map[mask.mask] = mask.mask_id
        
        # Look up embeddings for each pixel
        for i, (u, v) in enumerate(pixel_indices):
            u, v = int(u), int(v)
            
            if 0 <= u < W and 0 <= v < H:
                mask_id = pixel_mask_map[v, u]
                
                if mask_id >= 0 and mask_id in mask_id_to_idx:
                    # Use fused embedding for this mask
                    idx = mask_id_to_idx[mask_id]
                    embeddings[i] = features.fused_embeddings[idx]
                else:
                    # Use global embedding
                    embeddings[i] = features.global_embedding
            else:
                embeddings[i] = features.global_embedding
        
        return embeddings
    
    def fuse_frame_to_pointcloud(self, processed_frame: ProcessedFrame, semantic_pc: SemanticPointCloud) -> None:
        """
        Fuse a single frame's features into the point cloud.
        """
        pc_frame = processed_frame.point_cloud
        
        if len(pc_frame.points) == 0:
            return
        
        # Get embeddings for each point based on its pixel location
        embeddings = self.pixel_to_embedding(
            pc_frame.pixel_indices,
            processed_frame.segmentation,
            processed_frame.features
        )
        
        # Find which points in semantic_pc correspond to this frame
        # For now, we add points with their embeddings directly
        # TODO: track point-to-frame visibility
        
        # Update point cloud with embeddings
        start_idx = sum(semantic_pc.point_counts[:-1]) if semantic_pc.point_counts else 0
        end_idx = start_idx + len(embeddings)
        
        if end_idx <= len(semantic_pc.embeddings):
            semantic_pc.embeddings[start_idx:end_idx] = embeddings
    
    def create_semantic_map(self, loader: TUMDatasetLoader,
                            segmenter,
                            feature_extractor: CLIPFeatureExtractor,
                            frame_skip: int = 10,
                            max_frames: Optional[int] = None,
                            subsample: int = 4) -> SemanticPointCloud:
        """
        Create a complete semantic point cloud map.
        
        Full pipeline:
        1. Load frames with RGB and depth
        2. Generate point clouds
        3. Segment each frame
        4. Extract CLIP features
        5. Fuse features into 3D points
        
        Args:
            loader: TUM dataset loader
            segmenter: Instance segmenter
            feature_extractor: CLIP feature extractor
            frame_skip: Process every nth frame
            max_frames: Maximum frames to process
            subsample: Pixel subsampling for point cloud
            
        Returns:
            SemanticPointCloud with all points and embeddings
        """
        # Initialize point cloud generator
        pc_generator = PointCloudGenerator(loader.intrinsics, self.config)
        
        # Determine frames to process
        total_frames = len(loader)
        if max_frames is not None:
            total_frames = min(total_frames, max_frames * frame_skip)
        
        frame_indices = list(range(0, total_frames, frame_skip))
        n_frames = len(frame_indices)
        
        print(f"\n{'='*60}")
        print(f"Creating Semantic 3D Map")
        print(f"{'='*60}")
        print(f"  Total dataset frames: {len(loader)}")
        print(f"  Processing every {frame_skip}th frame")
        print(f"  Frames to process: {n_frames}")
        print(f"  Pixel subsampling: {subsample}x")
        print()
        
        # Initialize semantic point cloud
        semantic_pc = SemanticPointCloud()
        
        # Process each frame
        for i, frame_idx in enumerate(tqdm(frame_indices, desc="Processing frames")):
            # Load frame
            frame = loader[frame_idx]
            frame = loader.load_frame_images(frame)
            
            # Generate point cloud for this frame
            pc_frame = pc_generator.depth_to_pointcloud(frame, subsample=subsample)
            pc_frame.frame_idx = i
            
            # Segment frame
            segmentation = segmenter.segment_frame(frame.rgb_image)
            
            # Get mask crops for feature extraction
            mask_crops = segmenter.get_mask_crops(frame.rgb_image, segmentation) \
                if hasattr(segmenter, 'get_mask_crops') else []
            
            # Extract features
            features = feature_extractor.extract_frame_features(
                frame.rgb_image, mask_crops
            )
            
            # Get embeddings for each point
            embeddings = self.pixel_to_embedding(
                pc_frame.pixel_indices,
                segmentation,
                features
            )
            
            # Add to semantic point cloud
            semantic_pc.add_frame(pc_frame, embeddings)
        
        print(f"\n{'='*60}")
        print(f"Semantic Map Created")
        print(f"{'='*60}")
        print(f"  Total 3D points: {len(semantic_pc):,}")
        print(f"  Embedding dimension: {semantic_pc.embeddings.shape[1]}")
        print(f"  Memory usage: {semantic_pc.embeddings.nbytes / 1e6:.1f} MB (embeddings)")

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        semantic_pc.save(f'output/semantic_pc_{timestamp}.npz')
        
        return semantic_pc


class MultiViewFusion(SemanticFusion):
    """
    Advanced multi-view fusion with point tracking.
    
    Tracks which 3D points are visible across multiple frames
    and aggregates their embeddings with confidence weighting.
    """
    
    def __init__(self, intrinsics: CameraIntrinsics, config: Optional[dict] = None,
                 voxel_size: float = 0.02):
        super().__init__(intrinsics, config)
        self.voxel_size = voxel_size
        
        # Voxel grid for tracking visibility
        self.voxel_embeddings = {}  # voxel_key -> list of embeddings
        self.voxel_counts = {}  # voxel_key -> observation count
        self.voxel_positions = {}  # voxel_key -> average position
        self.voxel_colors = {}  # voxel_key -> average color
    
    def _point_to_voxel_key(self, point: np.ndarray) -> Tuple[int, int, int]:
        """Convert 3D point to voxel grid key."""
        voxel_idx = np.floor(point / self.voxel_size).astype(int)
        return tuple(voxel_idx)
    
    def accumulate_frame(self, pc_frame: PointCloudFrame,
                         segmentation: FrameSegmentation,
                         features: ImageFeatures) -> None:
        """
        Accumulate frame observations into voxel grid.
        
        Points in the same voxel from different views are aggregated.
        """
        if len(pc_frame.points) == 0:
            return
        
        # Get embeddings for all points
        embeddings = self.pixel_to_embedding(
            pc_frame.pixel_indices,
            segmentation,
            features
        )
        
        # Accumulate into voxels
        for i, (point, color, embedding) in enumerate(zip(
            pc_frame.points, pc_frame.colors, embeddings
        )):
            key = self._point_to_voxel_key(point)
            
            if key not in self.voxel_embeddings:
                self.voxel_embeddings[key] = []
                self.voxel_counts[key] = 0
                self.voxel_positions[key] = np.zeros(3)
                self.voxel_colors[key] = np.zeros(3)
            
            self.voxel_embeddings[key].append(embedding)
            self.voxel_counts[key] += 1
            self.voxel_positions[key] += point
            self.voxel_colors[key] += color
    
    def finalize(self) -> SemanticPointCloud:
        """
        Finalize the voxel grid into a semantic point cloud.
        
        Averages accumulated embeddings for each voxel.
        """
        n_voxels = len(self.voxel_embeddings)
        
        if n_voxels == 0:
            return SemanticPointCloud()
        
        # Get embedding dimension
        sample_key = next(iter(self.voxel_embeddings))
        embedding_dim = len(self.voxel_embeddings[sample_key][0])
        
        # Allocate arrays
        points = np.zeros((n_voxels, 3))
        colors = np.zeros((n_voxels, 3))
        embeddings = np.zeros((n_voxels, embedding_dim))
        
        for i, key in enumerate(self.voxel_embeddings):
            count = self.voxel_counts[key]
            
            # Average position and color
            points[i] = self.voxel_positions[key] / count
            colors[i] = self.voxel_colors[key] / count
            
            # Average embedding (can also use weighted average based on depth)
            voxel_embs = np.array(self.voxel_embeddings[key])
            embeddings[i] = voxel_embs.mean(axis=0)
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-8)
        
        # Create semantic point cloud
        semantic_pc = SemanticPointCloud()
        semantic_pc.points = points
        semantic_pc.colors = colors
        semantic_pc.embeddings = embeddings
        
        return semantic_pc


def test_fusion():
    """Test the fusion module."""
    import yaml
    from pathlib import Path
    
    # Load config
    config_path = Path(__file__).parent.parent / "config" / "tum_freiburg3.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    dataset_path = Path(__file__).parent.parent / "data" / "rgbd_dataset_freiburg3_long_office_household"
    loader = TUMDatasetLoader(str(dataset_path), config)
    
    # Use simple segmenter for testing
    segmenter = FastSAMSegmenter(config=config)
    
    # Initialize feature extractor
    feature_extractor = CLIPFeatureExtractor(model_name="ViT-B/32", config=config)
    
    # Initialize fusion
    fusion = SemanticFusion(loader.intrinsics, config)
    
    # Create semantic map (small test)
    semantic_pc = fusion.create_semantic_map(
        loader=loader,
        segmenter=segmenter,
        feature_extractor=feature_extractor,
        frame_skip=25,
        max_frames=50,
        subsample=8
    )

    from visualizer import RerunVisualizer

    visualizer = RerunVisualizer(app_name="Test Visualization")
    # visualizer.log_point_cloud(semantic_pc, point_size=0.02)
    
    # Test heatmap
    similarities = np.random.rand(len(semantic_pc))
    visualizer.log_similarity_heatmap(semantic_pc, similarities)
    
    print(f"\n✅ Fusion test completed!")
    print(f"  Points: {len(semantic_pc):,}")
    print(f"  Embedding shape: {semantic_pc.embeddings.shape}")


if __name__ == "__main__":
    test_fusion()
