"""
Visualization Module using Rerun.io

Provides interactive 3D visualization of the semantic point cloud
with query highlighting capabilities.

Rerun.io is a multimodal data visualization tool designed for
robotics and computer vision applications.
"""

import numpy as np
from typing import Optional, List, Tuple
from pathlib import Path
import colorsys

from point_cloud import SemanticPointCloud


def similarity_to_color(similarities: np.ndarray, 
                        colormap: str = "turbo") -> np.ndarray:
    """
    Convert similarity scores to RGB colors using a colormap.
    
    Args:
        similarities: Nx1 similarity scores (0-1)
        colormap: Colormap name ("turbo", "viridis", "plasma", "hot")
        
    Returns:
        Nx3 RGB colors (0-1)
    """
    # Normalize to 0-1
    min_sim = similarities.min()
    max_sim = similarities.max()
    if max_sim - min_sim > 1e-6:
        normalized = (similarities - min_sim) / (max_sim - min_sim)
    else:
        normalized = np.zeros_like(similarities)
    
    # Apply colormap
    if colormap == "turbo":
        # Turbo colormap approximation
        colors = _turbo_colormap(normalized)
    elif colormap == "hot":
        # Hot colormap (black -> red -> yellow -> white)
        colors = _hot_colormap(normalized)
    elif colormap == "viridis":
        colors = _viridis_colormap(normalized)
    else:
        # Default: blue (low) to red (high)
        colors = _blue_to_red_colormap(normalized)
    
    return colors


def _turbo_colormap(values: np.ndarray) -> np.ndarray:
    """Approximation of Google's Turbo colormap."""
    # Simplified turbo colormap
    colors = np.zeros((len(values), 3))
    
    for i, v in enumerate(values):
        if v < 0.25:
            # Dark blue to cyan
            t = v / 0.25
            colors[i] = [0.19, 0.07 + t * 0.43, 0.35 + t * 0.45]
        elif v < 0.5:
            # Cyan to green/yellow
            t = (v - 0.25) / 0.25
            colors[i] = [0.19 + t * 0.31, 0.5 + t * 0.4, 0.8 - t * 0.5]
        elif v < 0.75:
            # Yellow to orange
            t = (v - 0.5) / 0.25
            colors[i] = [0.5 + t * 0.45, 0.9 - t * 0.3, 0.3 - t * 0.2]
        else:
            # Orange to red
            t = (v - 0.75) / 0.25
            colors[i] = [0.95 - t * 0.1, 0.6 - t * 0.5, 0.1 + t * 0.05]
    
    return colors


def _hot_colormap(values: np.ndarray) -> np.ndarray:
    """Hot colormap."""
    colors = np.zeros((len(values), 3))
    
    for i, v in enumerate(values):
        if v < 0.33:
            colors[i] = [v * 3, 0, 0]  # Black to red
        elif v < 0.67:
            colors[i] = [1, (v - 0.33) * 3, 0]  # Red to yellow
        else:
            colors[i] = [1, 1, (v - 0.67) * 3]  # Yellow to white
    
    return colors


def _viridis_colormap(values: np.ndarray) -> np.ndarray:
    """Simplified viridis colormap."""
    colors = np.zeros((len(values), 3))
    
    for i, v in enumerate(values):
        # Viridis goes from dark purple to yellow
        colors[i] = [
            0.267 + v * 0.726,  # R: purple to yellow
            0.004 + v * 0.871,  # G: low to high
            0.329 + v * (0.208 - 0.329)  # B: purple to low
        ]
    
    return np.clip(colors, 0, 1)


def _blue_to_red_colormap(values: np.ndarray) -> np.ndarray:
    """Simple blue to red colormap via white."""
    colors = np.zeros((len(values), 3))
    
    for i, v in enumerate(values):
        if v < 0.5:
            # Blue to white
            t = v * 2
            colors[i] = [t, t, 1]
        else:
            # White to red
            t = (v - 0.5) * 2
            colors[i] = [1, 1 - t, 1 - t]
    
    return colors


class RerunVisualizer:
    """
    Rerun.io-based visualization for semantic 3D maps.
    
    Features:
    - Interactive 3D point cloud visualization
    - Query result highlighting
    - Camera trajectory display
    - Multi-view support
    """
    
    def __init__(self, app_name: str = "Queryable 3D Map"):
        """
        Initialize the Rerun visualizer.
        
        Args:
            app_name: Name shown in Rerun viewer
        """
        self.app_name = app_name
        self._initialized = False
        
        print(f"[RerunVisualizer] Initialized: {app_name}")
    
    def _ensure_initialized(self):
        """Initialize Rerun recording."""
        if self._initialized:
            return
        
        try:
            import rerun as rr
            
            rr.init(self.app_name, spawn=True)
            self._initialized = True
            print("  Rerun viewer spawned")
            
        except ImportError:
            raise ImportError(
                "Rerun not installed. Install with: pip install rerun-sdk"
            )
    
    def log_point_cloud(self, semantic_pc: SemanticPointCloud,
                        entity_path: str = "world/point_cloud",
                        point_size: float = 0.01,
                        use_colors: bool = True):
        """
        Log the full point cloud to Rerun.
        
        Args:
            semantic_pc: Semantic point cloud to visualize
            entity_path: Rerun entity path
            point_size: Size of points in viewer
            use_colors: Whether to use RGB colors (vs white)
        """
        self._ensure_initialized()
        import rerun as rr
        
        if len(semantic_pc.points) == 0:
            print("  Warning: Empty point cloud")
            return
        
        # Get colors
        if use_colors and len(semantic_pc.colors) > 0:
            colors = (semantic_pc.colors * 255).astype(np.uint8)
        else:
            colors = np.full((len(semantic_pc.points), 3), 200, dtype=np.uint8)
        
        # Log to Rerun
        rr.log(
            entity_path,
            rr.Points3D(
                semantic_pc.points,
                colors=colors,
                radii=point_size
            )
        )
        
        print(f"  Logged {len(semantic_pc.points):,} points to '{entity_path}'")


    def log_similarity_heatmap(self, semantic_pc: SemanticPointCloud,
                               similarities: np.ndarray,
                               entity_path: str = "world/heatmap",
                               point_size: float = 0.01,
                               colormap: str = "turbo"):
        """
        Visualize similarity scores as a heatmap on the point cloud.
        
        Args:
            semantic_pc: Point cloud
            similarities: Nx1 similarity scores
            entity_path: Rerun entity path
            point_size: Point size
            colormap: Colormap name
        """
        self._ensure_initialized()
        import rerun as rr
        
        # Convert similarities to colors
        colors = similarity_to_color(similarities, colormap=colormap)
        colors = (colors * 255).astype(np.uint8)
        
        rr.log(
            entity_path,
            rr.Points3D(
                semantic_pc.points,
                colors=colors,
                radii=point_size
            )
        )
        
        print(f"  Logged similarity heatmap to '{entity_path}'")
    
    def log_camera_trajectory(self, poses: List[np.ndarray],
                              entity_path: str = "world/camera_trajectory",
                              color: Tuple[int, int, int] = (0, 255, 0)):
        """
        Log camera trajectory as a line strip.
        
        Args:
            poses: List of 4x4 pose matrices
            entity_path: Rerun entity path
            color: RGB color for trajectory
        """
        self._ensure_initialized()
        import rerun as rr
        
        # Extract positions from poses
        positions = np.array([pose[:3, 3] for pose in poses])
        
        # Log as line strip
        rr.log(
            entity_path,
            rr.LineStrips3D(
                [positions],
                colors=[color]
            )
        )
        
        print(f"  Logged camera trajectory with {len(poses)} poses")
    
    def log_rgb_image(self, image: np.ndarray,
                      entity_path: str = "camera/rgb",
                      timestamp: Optional[float] = None):
        """
        Log an RGB image.
        
        Args:
            image: HxWx3 RGB image
            entity_path: Rerun entity path
            timestamp: Optional timestamp for timeline
        """
        self._ensure_initialized()
        import rerun as rr
        
        if timestamp is not None:
            rr.set_time_seconds("timestamp", timestamp)
        
        rr.log(entity_path, rr.Image(image))
    
    def log_depth_image(self, depth: np.ndarray,
                        entity_path: str = "camera/depth",
                        max_depth: float = 8.0):
        """
        Log a depth image.
        
        Args:
            depth: HxW depth in meters
            entity_path: Rerun entity path
            max_depth: Maximum depth for normalization
        """
        self._ensure_initialized()
        import rerun as rr
        
        # Normalize for visualization
        depth_viz = np.clip(depth / max_depth, 0, 1)
        
        rr.log(entity_path, rr.DepthImage(depth_viz))


def create_visualizer() -> RerunVisualizer:
    return RerunVisualizer()


def test_visualizer():
    """Test visualization module."""
    print("Visualizer Test")
    print("="*60)

    from pathlib import Path
    import yaml

    # from .data_loader import TUMDatasetLoader
    from data_loader import TUMDatasetLoader
    from point_cloud import PointCloudGenerator, SemanticPointCloud

    pc_path = Path(__file__).parent.parent / "point_cloud.npz"

    if pc_path.exists():
        print("Using existing point cloud from ", pc_path)
        semantic_pc = SemanticPointCloud.load(pc_path)
    else:
        # Load config
        config_path = Path(__file__).parent.parent / "config" / "tum_freiburg3.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize loader
        dataset_path = Path(__file__).parent.parent / "data" / "rgbd_dataset_freiburg3_long_office_household"
        loader = TUMDatasetLoader(str(dataset_path), config)
        
        # Get a few frames
        frames = loader.get_frames(skip=24, load_images=True) # start=0, end=10,  
        print(f"\nLoaded {len(frames)} frames")
        
        # Initialize generator
        generator = PointCloudGenerator(loader.intrinsics, config)
        
        # Generate point cloud
        semantic_pc = generator.process_frames(frames, subsample=4, voxel_downsample=True)
    
    n_points = len(semantic_pc)

    # # Create dummy data
    # n_points = 5000
    # semantic_pc = SemanticPointCloud()
    # semantic_pc.points = np.random.randn(n_points, 3)
    # semantic_pc.colors = np.random.rand(n_points, 3)
    # semantic_pc.embeddings = np.random.randn(n_points, 512)
    
    # Test colormap functions
    print("\nTesting colormaps...")
    test_values = np.linspace(0, 1, 100)
    
    for cmap in ["turbo", "hot", "viridis"]:
        colors = similarity_to_color(test_values, colormap=cmap)
        print(f"  {cmap}: shape={colors.shape}, range=[{colors.min():.2f}, {colors.max():.2f}]")
    
    # Try Rerun visualization
    print("\nTesting Rerun visualization...")
    try:
        visualizer = RerunVisualizer(app_name="Test Visualization")
        visualizer.log_point_cloud(semantic_pc, point_size=0.02)
        
        # Test heatmap
        similarities = np.random.rand(n_points)
        visualizer.log_similarity_heatmap(semantic_pc, similarities)
        
        print("\n✅ Rerun visualization test completed!")
        print("  Check the Rerun viewer for results.")
        
    except ImportError as e:
        print(f"  Rerun not available: {e}")


if __name__ == "__main__":
    test_visualizer()
