from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from data_loader import CameraIntrinsics, Frame


@dataclass
class PointCloudFrame:
    """Point cloud data for a single frame."""
    timestamp: float
    points: np.ndarray  # Nx3 world coordinates
    colors: np.ndarray  # Nx3 RGB colors
    pixel_indices: np.ndarray  # Nx2 original pixel coordinates [u, v]
    depths: np.ndarray  # N depth values in meters
    
    # Frame reference for back-tracking
    frame_idx: int = -1


@dataclass
class SemanticPointCloud:
    """
    Full semantic point cloud with embeddings.
    
    The main data structure for the queryable map.
    """
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))  # Nx3
    colors: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))  # Nx3
    embeddings: np.ndarray = field(default_factory=lambda: np.empty((0, 512)))  # NxD (CLIP dim)
    
    # Metadata
    frame_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    point_counts: List[int] = field(default_factory=list)  # Points per frame
    
    def __len__(self) -> int:
        return len(self.points)
    
    def add_frame(self, pc_frame: PointCloudFrame, embeddings: Optional[np.ndarray] = None):
        """Append points from a processed frame."""
        if len(pc_frame.points) == 0:
            return
        
        # Concatenate points and colors
        self.points = np.vstack([self.points, pc_frame.points]) if len(self.points) > 0 else pc_frame.points
        self.colors = np.vstack([self.colors, pc_frame.colors]) if len(self.colors) > 0 else pc_frame.colors
        
        # Add embeddings (zeros if not provided)
        if embeddings is None:
            embeddings = np.zeros((len(pc_frame.points), self.embeddings.shape[1] if len(self.embeddings) > 0 else 512))
        self.embeddings = np.vstack([self.embeddings, embeddings]) if len(self.embeddings) > 0 else embeddings
        
        # Metadata
        frame_indices = np.full(len(pc_frame.points), pc_frame.frame_idx, dtype=np.int32)
        self.frame_indices = np.concatenate([self.frame_indices, frame_indices]) if len(self.frame_indices) > 0 else frame_indices
        self.point_counts.append(len(pc_frame.points))
    
    def save(self, path: str):
        np.savez_compressed(
            path,
            points=self.points,
            colors=self.colors,
            embeddings=self.embeddings,
            frame_indices=self.frame_indices,
            point_counts=np.array(self.point_counts)
        )
        print(f"Saved semantic point cloud to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'SemanticPointCloud': # forward reference
        """Load saved point cloud from file"""
        data = np.load(path)
        pc = cls()
        pc.points = data['points']
        pc.colors = data['colors']
        pc.embeddings = data['embeddings']
        pc.frame_indices = data['frame_indices']
        pc.point_counts = data['point_counts'].tolist()
        return pc


class PointCloudGenerator:
    """
    Generates 3D point clouds from RGB-D frames.
    
    Implements depth back-projection using the pinhole camera model:
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    Z = depth
    
    Then transforms points to world coordinates using camera poses.
    """
    
    def __init__(self, intrinsics: CameraIntrinsics, config: Optional[dict] = None):
        self.intrinsics = intrinsics
        self.config = config or {}
        
        # Processing parameters
        proc_config = self.config.get('processing', {})
        self.max_depth = proc_config.get('max_depth', 8.0)
        self.min_depth = proc_config.get('min_depth', 0.1)
        self.voxel_size = proc_config.get('voxel_size', 0.02)
        
        # Precompute pixel coordinate grids for efficiency
        self._precompute_grids()
        
        print(f"[PointCloudGenerator] Initialized")
        print(f"  Depth range: {self.min_depth}m - {self.max_depth}m")
        print(f"  Voxel size: {self.voxel_size}m")
    
    def _precompute_grids(self):
        """Precompute pixel coordinate grids for vectorized back-projection."""
        u = np.arange(self.intrinsics.width)
        v = np.arange(self.intrinsics.height)
        self.u_grid, self.v_grid = np.meshgrid(u, v)
        
        # Precompute normalized coordinates
        self.u_norm = (self.u_grid - self.intrinsics.cx) / self.intrinsics.fx
        self.v_norm = (self.v_grid - self.intrinsics.cy) / self.intrinsics.fy
    
    def depth_to_pointcloud(self, frame: Frame, subsample: int = 1) -> PointCloudFrame:
        """
        Convert a single RGB-D frame to a point cloud in world coordinates.
        
        Args:
            frame: Frame with loaded RGB and depth images
            subsample: Subsample factor (1 = all pixels, 2 = every 2nd pixel, etc.)
            
        Returns:
            PointCloudFrame with 3D points and colors
        """
        if frame.depth_image is None or frame.rgb_image is None:
            raise ValueError("Frame must have loaded RGB and depth images")
        
        # Convert depth to meters
        depth = frame.depth_image.astype(np.float32) / self.intrinsics.depth_scale
        
        # Create valid depth mask
        valid_mask = (depth > self.min_depth) & (depth < self.max_depth)
        
        # Apply subsampling
        if subsample > 1:
            subsample_mask = np.zeros_like(valid_mask)
            subsample_mask[::subsample, ::subsample] = True
            valid_mask = valid_mask & subsample_mask
        
        # Get valid pixel coordinates
        valid_v, valid_u = np.where(valid_mask)
        valid_depth = depth[valid_mask]

        # Back-project to camera coordinates
        x_cam = (valid_u - self.intrinsics.cx) * valid_depth / self.intrinsics.fx
        y_cam = (valid_v - self.intrinsics.cy) * valid_depth / self.intrinsics.fy
        z_cam = valid_depth
        
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1).astype(np.float64)
        
        # Transform to world coordinates using pose matrix
        R = frame.pose_matrix[:3, :3].astype(np.float64)
        t = frame.pose_matrix[:3, 3].astype(np.float64)
        points_world = (R @ points_cam.T).T + t
        
        # Filter out any invalid points (NaN/Inf)
        valid_mask = np.isfinite(points_world).all(axis=1)
        points_world = points_world[valid_mask]
        valid_depth = valid_depth[valid_mask]
        valid_u = valid_u[valid_mask]
        valid_v = valid_v[valid_mask]

        # Reconstruct valid mask for color indexing
        color_mask = np.zeros((frame.rgb_image.shape[0], frame.rgb_image.shape[1]), dtype=bool)
        color_mask[valid_v.astype(int), valid_u.astype(int)] = True
        
        # Get colors (normalized to 0-1)
        colors = frame.rgb_image[valid_v.astype(int), valid_u.astype(int)].astype(np.float32) / 255.0
        
        # Store pixel indices for later feature mapping
        pixel_indices = np.stack([valid_u, valid_v], axis=-1)
        
        return PointCloudFrame(
            timestamp=frame.timestamp,
            points=points_world,
            colors=colors,
            pixel_indices=pixel_indices,
            depths=valid_depth
        )
    
    def process_frames(self, frames: List[Frame], subsample: int = 2,
                       voxel_downsample: bool = True) -> SemanticPointCloud:
        """
        Process multiple frames into a unified point cloud.
        
        Args:
            frames: List of frames with loaded images
            subsample: Pixel subsampling factor
            voxel_downsample: Whether to apply voxel downsampling
            
        Returns:
            SemanticPointCloud with all points
        """
        
        semantic_pc = SemanticPointCloud()
        
        print(f"\nProcessing {len(frames)} frames into point cloud...")
        
        for idx, frame in enumerate(tqdm(frames, desc="Generating point cloud")):
            # Convert frame to point cloud frame
            pc_frame = self.depth_to_pointcloud(frame, subsample=subsample)
            pc_frame.frame_idx = idx
            
            # Apend it to the semantic point cloud
            semantic_pc.add_frame(pc_frame)
        
        print(f"  Total points before downsampling: {len(semantic_pc):,}")
        
        # Apply voxel downsampling if required
        if voxel_downsample and self.voxel_size > 0:
            semantic_pc = self._voxel_downsample(semantic_pc)
            print(f"  Total points after downsampling: {len(semantic_pc):,}")
        
        semantic_pc.save('./point_cloud.npz')
        return semantic_pc
    
    def _voxel_downsample(self, semantic_pc: SemanticPointCloud) -> SemanticPointCloud:
        """
        Apply voxel downsampling to reduce point density.
        
        For each voxel, we keep the average position, color, and embedding.
        """
        # Compute voxel indices
        voxel_indices = np.floor(semantic_pc.points / self.voxel_size).astype(np.int32)
        
        # Create unique voxel keys
        # Use a large prime to create unique keys from 3D indices
        keys = (voxel_indices[:, 0].astype(np.int64) * 73856093 + 
                voxel_indices[:, 1].astype(np.int64) * 19349663 + 
                voxel_indices[:, 2].astype(np.int64) * 83492791)
        
        # Find unique voxels and aggregate
        unique_keys, inverse_indices = np.unique(keys, return_inverse=True)
        n_voxels = len(unique_keys)
        
        # Aggregate by averaging
        new_points = np.zeros((n_voxels, 3))
        new_colors = np.zeros((n_voxels, 3))
        new_embeddings = np.zeros((n_voxels, semantic_pc.embeddings.shape[1]))
        counts = np.zeros(n_voxels)
        
        np.add.at(new_points, inverse_indices, semantic_pc.points)
        np.add.at(new_colors, inverse_indices, semantic_pc.colors)
        np.add.at(new_embeddings, inverse_indices, semantic_pc.embeddings)
        np.add.at(counts, inverse_indices, 1)
        
        # Normalize by counts
        counts = counts.reshape(-1, 1)
        new_points /= counts
        new_colors /= counts
        new_embeddings /= counts
        
        # Create new semantic point cloud
        downsampled = SemanticPointCloud()
        downsampled.points = new_points
        downsampled.colors = new_colors
        downsampled.embeddings = new_embeddings
        downsampled.frame_indices = np.zeros(n_voxels, dtype=np.int32)  # Lost original frame info
        
        return downsampled


def test_point_cloud_generator():
    """Test the point cloud generator."""
    from pathlib import Path
    import yaml

    from data_loader import TUMDatasetLoader

    # Load config
    config_path = Path(__file__).parent.parent / "config" / "tum_freiburg3.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize loader
    dataset_path = Path(__file__).parent.parent / "data" / "rgbd_dataset_freiburg3_long_office_household"
    loader = TUMDatasetLoader(str(dataset_path), config)
    
    # Get a few frames
    frames = loader.get_frames(start=0, end=10, skip=1, load_images=True)
    print(f"\nLoaded {len(frames)} frames")
    
    # Initialize generator
    generator = PointCloudGenerator(loader.intrinsics, config)
    
    # Generate point cloud
    semantic_pc = generator.process_frames(frames, subsample=4, voxel_downsample=True)
    
    print(f"\n✅ Point Cloud Generator test passed!")
    print(f"  Final point cloud: {len(semantic_pc):,} points")


if __name__ == "__main__":
    test_point_cloud_generator()
