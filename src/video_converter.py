"""
Video to TUM RGB-D Format Converter

Converts input video files to TUM RGB-D dataset format for use with TUMDatasetLoader.
Generates:
- rgb/ folder with RGB frames
- depth/ folder with depth images (estimated if not provided)
- rgb.txt, depth.txt, groundtruth.txt files

Uses:
- Depth Anything V2 / MiDaS for monocular depth estimation
- OpenCV-based Visual Odometry for camera pose estimation
"""

import os
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List
from scipy.spatial.transform import Rotation
from tqdm import tqdm


@dataclass
class ConverterConfig:
    """Configuration for video to TUM conversion."""
    frame_skip: int = 1  # Process every nth frame
    max_frames: Optional[int] = None  # Max frames to process (None = all)
    depth_scale: float = 5000.0  # TUM depth scale factor
    max_depth: float = 10.0  # Max depth in meters for normalization
    
    # Camera intrinsics (default - should be calibrated for best results)
    fx: float = 525.0
    fy: float = 525.0
    cx: float = 319.5
    cy: float = 239.5
    
    # Visual Odometry settings
    feature_detector: str = "ORB"  # ORB or SIFT
    min_features: int = 500
    

class DepthEstimator:
    """
    Monocular depth estimation using Depth Anything V2 or MiDaS.
    
    Falls back to simpler methods if transformers models unavailable.
    """
    
    def __init__(self, device: str = "auto"):
        """Initialize depth estimator with best available model."""
        self.model = None
        self.processor = None
        self.model_name = None
        self.device = self._get_device(device)
        
        self._load_model()
    
    def _get_device(self, device: str) -> str:
        """Determine best available device."""
        if device != "auto":
            return device
        
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def _load_model(self):
        """Load the best available depth estimation model."""
        import torch
        
        try:
            from transformers import pipeline
            
            # Try Depth Anything V2 first (best quality)
            try:
                print("[DepthEstimator] Loading Depth Anything V2...")
                self.pipe = pipeline(
                    task="depth-estimation",
                    model="depth-anything/Depth-Anything-V2-Small-hf",
                    device=0 if self.device == "cuda" else -1 if self.device == "cpu" else self.device
                )
                self.model_name = "Depth-Anything-V2"
                print(f"[DepthEstimator] Loaded {self.model_name} on {self.device}")
                return
            except Exception as e:
                print(f"[DepthEstimator] Depth Anything V2 unavailable: {e}")
            
            # Fallback to MiDaS
            try:
                print("[DepthEstimator] Trying MiDaS...")
                self.pipe = pipeline(
                    task="depth-estimation",
                    model="Intel/dpt-hybrid-midas",
                    device=0 if self.device == "cuda" else -1 if self.device == "cpu" else self.device
                )
                self.model_name = "MiDaS"
                print(f"[DepthEstimator] Loaded {self.model_name} on {self.device}")
                return
            except Exception as e:
                print(f"[DepthEstimator] MiDaS unavailable: {e}")
            
        except ImportError:
            print("[DepthEstimator] transformers not available")
        
        # Final fallback: simple edge-based depth (not recommended)
        print("[DepthEstimator] WARNING: Using simple edge-based depth estimation (low quality)")
        self.model_name = "edge-fallback"
        self.pipe = None
    
    def estimate(self, rgb_image: np.ndarray, max_depth: float = 10.0) -> np.ndarray:
        """
        Estimate depth from RGB image.
        
        Args:
            rgb_image: RGB image (H, W, 3) numpy array
            max_depth: Maximum depth value in meters
            
        Returns:
            Depth image as uint16 numpy array (scaled by depth_scale)
        """
        if self.pipe is not None:
            # Use transformers pipeline
            from PIL import Image
            pil_image = Image.fromarray(rgb_image)
            result = self.pipe(pil_image)
            depth = np.array(result["depth"])
            
            # Normalize depth to [0, max_depth] range
            depth = depth.astype(np.float32)
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            depth = depth * max_depth
            
        else:
            # Edge-based fallback (very rough approximation)
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            depth = cv2.GaussianBlur(edges.astype(np.float32), (31, 31), 0)
            depth = max_depth * (1.0 - depth / (depth.max() + 1e-8))
        
        # Resize to match input if needed
        if depth.shape[:2] != rgb_image.shape[:2]:
            depth = cv2.resize(depth, (rgb_image.shape[1], rgb_image.shape[0]))
        
        return depth
    
    def depth_to_tum_format(self, depth_meters: np.ndarray, depth_scale: float = 5000.0) -> np.ndarray:
        """Convert depth in meters to TUM 16-bit format."""
        depth_scaled = (depth_meters * depth_scale).astype(np.uint16)
        return depth_scaled


class VisualOdometry:
    """
    Simple feature-based Visual Odometry using OpenCV.
    
    Estimates camera poses between consecutive frames using feature matching
    and essential matrix decomposition.
    """
    
    def __init__(self, config: ConverterConfig):
        """Initialize VO with camera intrinsics."""
        self.config = config
        self.K = np.array([
            [config.fx, 0, config.cx],
            [0, config.fy, config.cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Feature detector
        if config.feature_detector == "SIFT":
            self.detector = cv2.SIFT_create(nfeatures=config.min_features)
        else:
            self.detector = cv2.ORB_create(nfeatures=config.min_features)
        
        # Feature matcher
        if config.feature_detector == "SIFT":
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # State
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        
        # Accumulated pose (camera-to-world)
        self.current_pose = np.eye(4)
        self.poses = []
    
    def reset(self):
        """Reset VO state."""
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.current_pose = np.eye(4)
        self.poses = []
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a new frame and estimate pose.
        
        Args:
            frame: RGB image
            
        Returns:
            4x4 camera-to-world transformation matrix
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Detect features
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        
        if self.prev_frame is None:
            # First frame - just store
            self.prev_frame = gray
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            self.poses.append(self.current_pose.copy())
            return self.current_pose.copy()
        
        if descriptors is None or self.prev_descriptors is None or \
           len(keypoints) < 8 or len(self.prev_keypoints) < 8:
            # Not enough features, keep previous pose
            self.poses.append(self.current_pose.copy())
            self.prev_frame = gray
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            return self.current_pose.copy()
        
        # Match features
        try:
            matches = self.matcher.knnMatch(self.prev_descriptors, descriptors, k=2)
        except cv2.error:
            self.poses.append(self.current_pose.copy())
            return self.current_pose.copy()
        
        # Lowe's ratio test
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 8:
            # Not enough good matches
            self.poses.append(self.current_pose.copy())
            self.prev_frame = gray
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            return self.current_pose.copy()
        
        # Extract matched point coordinates
        pts1 = np.float32([self.prev_keypoints[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([keypoints[m.trainIdx].pt for m in good_matches])
        
        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        if E is None:
            self.poses.append(self.current_pose.copy())
            self.prev_frame = gray
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            return self.current_pose.copy()
        
        # Recover pose
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K)
        
        # Build transformation matrix (this frame relative to previous)
        T_rel = np.eye(4)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = t.flatten()
        
        # Accumulate pose
        self.current_pose = self.current_pose @ T_rel
        self.poses.append(self.current_pose.copy())
        
        # Update previous frame
        self.prev_frame = gray
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        
        return self.current_pose.copy()
    
    def get_all_poses(self) -> List[np.ndarray]:
        """Get all accumulated poses."""
        return self.poses


def pose_matrix_to_tum(pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert 4x4 pose matrix to TUM format (position, quaternion).
    
    Args:
        pose: 4x4 camera-to-world transformation matrix
        
    Returns:
        (position [tx, ty, tz], quaternion [qx, qy, qz, qw])
    """
    position = pose[:3, 3]
    rotation = Rotation.from_matrix(pose[:3, :3])
    quaternion = rotation.as_quat()  # [qx, qy, qz, qw]
    return position, quaternion


class VideoToTUMConverter:
    """
    Main converter class for video to TUM RGB-D format.
    
    Usage:
        converter = VideoToTUMConverter(
            video_path="input.mp4",
            output_dir="output/my_dataset",
            config=ConverterConfig(frame_skip=2)
        )
        converter.convert()
        
        # Then use with TUMDatasetLoader:
        from src2.data_loader import TUMDatasetLoader
        loader = TUMDatasetLoader("output/my_dataset")
    """
    
    def __init__(
        self,
        video_path: str,
        output_dir: str,
        config: Optional[ConverterConfig] = None,
        depth_dir: Optional[str] = None,
        poses_file: Optional[str] = None,
    ):
        """
        Initialize the converter.
        
        Args:
            video_path: Path to input video file
            output_dir: Output directory for TUM format data
            config: Converter configuration
            depth_dir: Optional pre-computed depth images directory
            poses_file: Optional pre-computed poses file (TUM format)
        """
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.config = config or ConverterConfig()
        
        # Optional pre-computed data
        self.depth_dir = Path(depth_dir) if depth_dir else None
        self.poses_file = Path(poses_file) if poses_file else None
        
        # Create output directories
        self.rgb_dir = self.output_dir / "rgb"
        self.depth_output_dir = self.output_dir / "depth"
        
        # Lazy-loaded components
        self._depth_estimator = None
        self._visual_odometry = None
        
        # Pre-loaded external data
        self.external_poses = None
        self.external_depths = None
    
    @property
    def depth_estimator(self) -> DepthEstimator:
        """Lazy-load depth estimator."""
        if self._depth_estimator is None:
            self._depth_estimator = DepthEstimator()
        return self._depth_estimator
    
    @property
    def visual_odometry(self) -> VisualOdometry:
        """Lazy-load visual odometry."""
        if self._visual_odometry is None:
            self._visual_odometry = VisualOdometry(self.config)
        return self._visual_odometry
    
    def _load_external_poses(self):
        """Load pre-computed poses from TUM format file."""
        if self.poses_file is None or not self.poses_file.exists():
            return None
        
        poses = {}
        with open(self.poses_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                parts = line.split()
                if len(parts) >= 8:
                    timestamp = float(parts[0])
                    position = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                    quaternion = np.array([float(parts[4]), float(parts[5]), 
                                          float(parts[6]), float(parts[7])])
                    poses[timestamp] = (position, quaternion)
        
        print(f"[VideoToTUMConverter] Loaded {len(poses)} external poses")
        return poses
    
    def _load_external_depths(self) -> Optional[dict]:
        """Check for pre-computed depth images."""
        if self.depth_dir is None or not self.depth_dir.exists():
            return None
        
        depth_files = {}
        for f in self.depth_dir.glob("*.png"):
            # Try to parse timestamp from filename
            try:
                timestamp = float(f.stem)
                depth_files[timestamp] = f
            except ValueError:
                # If not timestamp-named, use index
                pass
        
        if depth_files:
            print(f"[VideoToTUMConverter] Found {len(depth_files)} external depth files")
        return depth_files if depth_files else None
    
    def convert(self):
        """
        Run the full conversion pipeline.
        
        Creates:
        - rgb/ directory with frame images
        - depth/ directory with depth images
        - rgb.txt, depth.txt, groundtruth.txt files
        """
        print(f"\n{'='*60}")
        print(f"Video to TUM RGB-D Converter")
        print(f"{'='*60}")
        print(f"Input video: {self.video_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"Frame skip: {self.config.frame_skip}")
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rgb_dir.mkdir(exist_ok=True)
        self.depth_output_dir.mkdir(exist_ok=True)
        
        # Load external data if provided
        self.external_poses = self._load_external_poses()
        self.external_depths = self._load_external_depths()
        
        # Open video
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video FPS: {fps}")
        print(f"Total frames: {total_frames}")
        
        # Compute camera intrinsics from video dimensions if not set
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if self.config.cx == 319.5 and width != 640:
            # Adjust default intrinsics for actual video size
            self.config.cx = width / 2.0
            self.config.cy = height / 2.0
            self.config.fx = max(width, height)
            self.config.fy = max(width, height)
            print(f"[VideoToTUMConverter] Auto-adjusted intrinsics for {width}x{height}")
        
        # Reset VO
        if self.external_poses is None:
            self.visual_odometry.reset()
        
        # Process frames
        rgb_entries = []
        depth_entries = []
        pose_entries = []
        
        frame_idx = 0
        processed_count = 0
        
        max_frames = self.config.max_frames or total_frames
        
        with tqdm(total=min(total_frames, max_frames), desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames
                if frame_idx % self.config.frame_skip != 0:
                    frame_idx += 1
                    continue
                
                # Check max frames
                if processed_count >= max_frames:
                    break
                
                # Compute timestamp
                timestamp = frame_idx / fps
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Save RGB frame
                rgb_filename = f"{timestamp:.6f}.png"
                rgb_path = self.rgb_dir / rgb_filename
                cv2.imwrite(str(rgb_path), frame)  # Save as BGR for OpenCV compatibility
                rgb_entries.append((timestamp, f"rgb/{rgb_filename}"))
                
                # Get/estimate depth
                if self.external_depths and timestamp in self.external_depths:
                    # Use external depth
                    depth_src = self.external_depths[timestamp]
                    depth_img = cv2.imread(str(depth_src), cv2.IMREAD_UNCHANGED)
                else:
                    # Estimate depth
                    depth_meters = self.depth_estimator.estimate(rgb_frame, self.config.max_depth)
                    depth_img = self.depth_estimator.depth_to_tum_format(
                        depth_meters, self.config.depth_scale
                    )
                
                # Save depth
                depth_filename = f"{timestamp:.6f}.png"
                depth_path = self.depth_output_dir / depth_filename
                cv2.imwrite(str(depth_path), depth_img)
                depth_entries.append((timestamp, f"depth/{depth_filename}"))
                
                # Get/estimate pose
                if self.external_poses:
                    # Find nearest pose
                    nearest_ts = min(self.external_poses.keys(), 
                                   key=lambda x: abs(x - timestamp))
                    if abs(nearest_ts - timestamp) < 0.1:
                        position, quaternion = self.external_poses[nearest_ts]
                    else:
                        # No close pose, use identity
                        position = np.zeros(3)
                        quaternion = np.array([0, 0, 0, 1])
                else:
                    # Estimate pose using VO
                    pose_matrix = self.visual_odometry.process_frame(rgb_frame)
                    position, quaternion = pose_matrix_to_tum(pose_matrix)
                
                pose_entries.append((timestamp, position, quaternion))
                
                frame_idx += 1
                processed_count += 1
                pbar.update(1)
        
        cap.release()
        
        # Write output files
        self._write_file_list(self.output_dir / "rgb.txt", rgb_entries)
        self._write_file_list(self.output_dir / "depth.txt", depth_entries)
        self._write_groundtruth(self.output_dir / "groundtruth.txt", pose_entries)
        
        print(f"\n{'='*60}")
        print(f"Conversion complete!")
        print(f"{'='*60}")
        print(f"Processed frames: {processed_count}")
        print(f"Output files:")
        print(f"  - {self.output_dir / 'rgb.txt'}")
        print(f"  - {self.output_dir / 'depth.txt'}")
        print(f"  - {self.output_dir / 'groundtruth.txt'}")
        print(f"  - {self.rgb_dir}/ ({len(rgb_entries)} files)")
        print(f"  - {self.depth_output_dir}/ ({len(depth_entries)} files)")
        
        return processed_count
    
    def _write_file_list(self, filepath: Path, entries: List[Tuple[float, str]]):
        """Write rgb.txt or depth.txt file."""
        with open(filepath, 'w') as f:
            f.write("# timestamp filename\n")
            for timestamp, filename in entries:
                f.write(f"{timestamp:.6f} {filename}\n")
    
    def _write_groundtruth(self, filepath: Path, 
                           entries: List[Tuple[float, np.ndarray, np.ndarray]]):
        """Write groundtruth.txt file."""
        with open(filepath, 'w') as f:
            f.write("# timestamp tx ty tz qx qy qz qw\n")
            for timestamp, position, quaternion in entries:
                f.write(f"{timestamp:.6f} {position[0]:.6f} {position[1]:.6f} {position[2]:.6f} "
                       f"{quaternion[0]:.6f} {quaternion[1]:.6f} {quaternion[2]:.6f} {quaternion[3]:.6f}\n")


def test_converter():
    """Test the video converter with a sample video."""
    import tempfile
    
    print("\n" + "="*60)
    print("Testing VideoToTUMConverter")
    print("="*60)
    
    video_path = Path(__file__).parent.parent / "test_video.mp4"

    output_dir = Path(__file__).parent.parent / "output" / "generated_data"
        
    # Test converter
    print("\n2. Running converter...")
    config = ConverterConfig(frame_skip=25, max_frames=None)
    converter = VideoToTUMConverter(
        video_path=str(video_path),
        output_dir=str(output_dir),
        config=config
    )
    
    num_frames = converter.convert()
    
    # Verify outputs
    print("\n3. Verifying outputs...")
    assert (output_dir / "rgb.txt").exists(), "rgb.txt not created"
    assert (output_dir / "depth.txt").exists(), "depth.txt not created"
    assert (output_dir / "groundtruth.txt").exists(), "groundtruth.txt not created"
    assert (output_dir / "rgb").exists(), "rgb/ directory not created"
    assert (output_dir / "depth").exists(), "depth/ directory not created"
    
    rgb_files = list((output_dir / "rgb").glob("*.png"))
    depth_files = list((output_dir / "depth").glob("*.png"))
    
    print(f"   RGB files: {len(rgb_files)}")
    print(f"   Depth files: {len(depth_files)}")
    assert len(rgb_files) > 0, "No RGB files created"
    assert len(depth_files) > 0, "No depth files created"
    assert len(rgb_files) == len(depth_files), "RGB/depth count mismatch"
    
    # Test loading with TUMDatasetLoader
    print("\n4. Testing with TUMDatasetLoader...")
    from data_loader import TUMDatasetLoader
    loader = TUMDatasetLoader(str(output_dir))
    
    print(f"   Loader found {len(loader)} associated frames")
    assert len(loader) > 0, "No frames loaded"
    
    # Load a frame
    frame = loader[0]
    print(f"   Frame timestamp: {frame.timestamp}")
    print(f"   Position: {frame.position}")
    
    frame = loader.load_frame_images(frame)
    print(f"   RGB shape: {frame.rgb_image.shape}")
    print(f"   Depth shape: {frame.depth_image.shape}")
    
    print("\n" + "="*60)
    print("✅ VideoToTUMConverter test passed!")
    print("="*60)


if __name__ == "__main__":
    test_converter()
