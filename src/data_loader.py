import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class Frame:
    """Single frame with RGB, depth, and pose data"""
    timestamp: float
    rgb_path: str
    depth_path: str
    rgb_image: Optional[np.ndarray] = None
    depth_image: Optional[np.ndarray] = None
    
    # Camera pose (world-to-camera transform)
    position: Optional[np.ndarray] = None  # [tx, ty, tz]
    quaternion: Optional[np.ndarray] = None  # [qx, qy, qz, qw]
    
    # 4x4 transformation matrix (camera-to-world)
    pose_matrix: Optional[np.ndarray] = None


@dataclass 
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float  # focal length x
    fy: float  # focal length y
    cx: float  # principal point x
    cy: float  # principal point y
    width: int
    height: int
    depth_scale: float = 5000.0  # TUM depth scale
    
    def get_matrix(self) -> np.ndarray:
        """Return 3x3 intrinsic matrix."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])


class TUMDatasetLoader:
    """
    TUM RGB-D Dataset Loader

    Parses the TUM RGB-D format which consists of:
    - rgb.txt: timestamp -> RGB image path
    - depth.txt: timestamp -> depth image path  
    - groundtruth.txt: timestamp -> pose (tx, ty, tz, qx, qy, qz, qw)

    The loader associates RGB, depth, and poses by nearest timestamp.
    
    """
    
    def __init__(self, dataset_path: str, config: Optional[dict] = None):
        self.dataset_path = Path(dataset_path)
        self.config = config or {}
        
        # Freiburg3 camera intrinsics as default values
        # https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats#intrinsic_camera_calibration_of_the_kinect
        cam_config = self.config.get('camera', {})
        self.intrinsics = CameraIntrinsics(
            fx=cam_config.get('fx', 535.4),
            fy=cam_config.get('fy', 539.2),
            cx=cam_config.get('cx', 320.1),
            cy=cam_config.get('cy', 247.6),
            width=cam_config.get('width', 640),
            height=cam_config.get('height', 480),
            depth_scale=cam_config.get('depth_scale', 5000.0)
        )
        
        # Load file lists
        self.rgb_list = self._parse_file_list('rgb.txt')
        self.depth_list = self._parse_file_list('depth.txt')
        self.groundtruth = self._parse_groundtruth('groundtruth.txt')
        
        # Associate timestamps
        self.associations = self._associate_data()
        
        print(f"[TUMDatasetLoader] Loaded dataset from {dataset_path}")
        print(f"  RGB frames: {len(self.rgb_list)}")
        print(f"  Depth frames: {len(self.depth_list)}")
        print(f"  Groundtruth poses: {len(self.groundtruth)}")
        print(f"  Associated frames: {len(self.associations)}")
    
    def _parse_file_list(self, filename: str) -> Dict[float, str]:
        """Parse rgb.txt and depth.txt file format"""
        filepath = self.dataset_path / filename
        result = {}
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    timestamp = float(parts[0])
                    path = parts[1]
                    result[timestamp] = path
        
        return result
    
    def _parse_groundtruth(self, filename: str) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
        """Parse camera poses from groundtruth.txt file"""
        filepath = self.dataset_path / filename
        result = {}
        
        with open(filepath, 'r') as f:
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
                    result[timestamp] = (position, quaternion)
        
        return result
    
    def _find_nearest_timestamp(self, target: float, timestamps: List[float], max_diff: float = 0.02) -> Optional[float]:
        """Find nearest timestamp within max_diff seconds."""
        if not timestamps:
            return None
        
        # Binary search
        idx = np.searchsorted(timestamps, target)
        
        candidates = []
        if idx > 0:
            candidates.append(timestamps[idx - 1])
        if idx < len(timestamps):
            candidates.append(timestamps[idx])
        
        if not candidates:
            return None
        
        nearest = min(candidates, key=lambda x: abs(x - target))
        if abs(nearest - target) <= max_diff:
            return nearest
        return None
    
    def _associate_data(self, max_diff: float = 0.02) -> List[Tuple[float, float, float]]:
        """
        Associate RGB, depth, and groundtruth by timestamp.
        
        Returns list of tuples of key timestamps: (rgb_ts, depth_ts, gt_ts)
        """
        rgb_timestamps = sorted(self.rgb_list.keys())
        depth_timestamps = sorted(self.depth_list.keys())
        gt_timestamps = sorted(self.groundtruth.keys())
        
        associations = []
        
        for rgb_ts in rgb_timestamps:
            # Find depth match
            depth_ts = self._find_nearest_timestamp(rgb_ts, depth_timestamps, max_diff)
            if depth_ts is None:
                continue
            
            # Find groundtruth match
            gt_ts = self._find_nearest_timestamp(rgb_ts, gt_timestamps, max_diff)
            if gt_ts is None:
                continue
            
            associations.append((rgb_ts, depth_ts, gt_ts))
        
        return associations
    
    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion [qx, qy, qz, qw] to 3x3 rotation matrix for world coordinate calculation"""
        rot = Rotation.from_quat(q)
        return rot.as_matrix()
    
    def _compute_pose_matrix(self, position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
        """Compute the homogeneous 4x4 pose matrix for camera-to-world transform in this format
            [ R11 R12 R13  tx ]
            [ R21 R22 R23  ty ]
            [ R31 R32 R33  tz ]
            [  0   0   0   1  ]
        """
        R = self._quaternion_to_rotation_matrix(quaternion)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = position
        return T
    
    def __len__(self) -> int:
        return len(self.associations)
    
    def __getitem__(self, idx: int) -> Frame:
        rgb_ts, depth_ts, gt_ts = self.associations[idx]
        
        frame = Frame(
            timestamp=rgb_ts,
            rgb_path=str(self.dataset_path / self.rgb_list[rgb_ts]),
            depth_path=str(self.dataset_path / self.depth_list[depth_ts]),
        )
        
        position, quaternion = self.groundtruth[gt_ts]
        frame.position = position
        frame.quaternion = quaternion
        frame.pose_matrix = self._compute_pose_matrix(position, quaternion)
        
        return frame
    
    def load_frame_images(self, frame: Frame, load_rgb: bool = True, load_depth: bool = True) -> Frame:
        if load_rgb and frame.rgb_image is None:
            frame.rgb_image = cv2.imread(frame.rgb_path)
            if frame.rgb_image is not None:
                frame.rgb_image = cv2.cvtColor(frame.rgb_image, cv2.COLOR_BGR2RGB)
        
        if load_depth and frame.depth_image is None:
            frame.depth_image = cv2.imread(frame.depth_path, cv2.IMREAD_UNCHANGED)
        
        return frame
    
    def get_frames(self, start: int = 0, end: Optional[int] = None, skip: int = 1, load_images: bool = False) -> List[Frame]:
        """
        Get multiple frames with optional sampling.
        
        Args:
            start: Start index
            end: End index (None for all)
            skip: Sample every nth frame
            load_images: Whether to load RGB/depth images
            
        Returns:
            List of Frame objects
        """
        if end is None:
            end = len(self)

        frames = []
        for idx in range(start, end, skip):
            frame = self[idx]
            if load_images:
                frame = self.load_frame_images(frame)
            frames.append(frame)
        
        return frames
    
    def get_depth_in_meters(self, depth_image: np.ndarray) -> np.ndarray:
        """Convert 16-bit depth image to depth in meters."""
        return depth_image.astype(np.float32) / self.intrinsics.depth_scale


def test_loader():
    import yaml

    # Load config
    config_path = Path(__file__).parent.parent / "config" / "tum_freiburg3.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Initialize loader
    dataset_path = Path(__file__).parent.parent / "data" / "rgbd_dataset_freiburg3_long_office_household"
    loader = TUMDatasetLoader(str(dataset_path), config)
    
    # Test loading a frame
    print("\nTesting frame loading...")
    frame = loader[0]
    print(f"  Timestamp: {frame.timestamp}")
    print(f"  RGB path: {frame.rgb_path}")
    print(f"  Depth path: {frame.depth_path}")
    print(f"  Position: {frame.position}")
    print(f"  Quaternion: {frame.quaternion}")
    print(f"  Pose matrix shape: {frame.pose_matrix.shape}")
    
    # Load images
    frame = loader.load_frame_images(frame)
    print(f"  RGB shape: {frame.rgb_image.shape}")
    print(f"  Depth shape: {frame.depth_image.shape}")
    print(f"  Depth dtype: {frame.depth_image.dtype}")
    
    # Convert depth to meters
    depth_meters = loader.get_depth_in_meters(frame.depth_image)
    valid_depth = depth_meters[depth_meters > 0]
    print(f"  Depth range: {valid_depth.min():.3f}m - {valid_depth.max():.3f}m")


    # Get frames
    import sys
    frames = loader.get_frames(load_images=True)
    print(f"  Number of frames: {len(frames)}")

    
    print("\n✅ TUM Dataset Loader test passed!")


if __name__ == "__main__":
    test_loader()
