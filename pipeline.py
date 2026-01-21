"""
Pipeline stages:
1. Load RGB-D data with poses
2. Generate 3D point cloud
3. Segment frames with FastSAM
4. Extract CLIP features
5. Fuse features into 3D points
6. Enable natural language queries
7. Visualize with Rerun.io
"""

import yaml
import numpy as np
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
import time
import argparse

from src.data_loader import TUMDatasetLoader
from src.point_cloud import PointCloudGenerator, SemanticPointCloud
from src.segmentation import FastSAMSegmenter
from src.feature_extractor import CLIPFeatureExtractor, create_feature_extractor
from src.fusion import SemanticFusion, MultiViewFusion
from src.query_engine import QueryEngine
from src.visualizer import RerunVisualizer


@dataclass
class PipelineConfig:
    """Configuration for the pipeline."""
    # Dataset
    dataset_path: str
    config_path: Optional[str] = None
    
    # Processing
    frame_skip: int = 10
    max_frames: Optional[int] = None
    pixel_subsample: int = 4
    voxel_size: float = 0.02
    
    # Models
    segmenter_type: str = "fastsam"
    clip_model: str = "ViT-B/32"
    
    # Output
    output_dir: str = "output"
    save_pointcloud: bool = True
    
    # Visualization
    visualize: bool = True
    visualizer_backend: str = "rerun"


class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        
        # Load dataset config
        self.dataset_config = self._load_config()
        
        # Initialize components
        self._loader = None
        self._segmenter = None
        self._feature_extractor = None
        self._fusion = None
        self._visualizer = None
        
        # Results
        self.semantic_pc = None
        self.query_engine = None
        
        print(f"\n{'='*60}")
        print("Pipeline - Queryable 3D Semantic Map")
        print(f"{'='*60}")
        print(f"Dataset: {config.dataset_path}")
        print(f"Frame skip: {config.frame_skip}")
        print(f"Segmenter: {config.segmenter_type}")
        print(f"CLIP model: {config.clip_model}")
        print()
    
    def _load_config(self) -> dict:
        if self.config.config_path and Path(self.config.config_path).exists():
            with open(self.config.config_path, 'r') as f:
                return yaml.safe_load(f)
        
        return {
            'camera': {
                'fx': 535.4, 'fy': 539.2,
                'cx': 320.1, 'cy': 247.6,
                'width': 640, 'height': 480,
                'depth_scale': 5000.0
            },
            'processing': {
                'max_depth': 8.0,
                'min_depth': 0.1,
                'voxel_size': self.config.voxel_size
            }
        }
    
    @property
    def loader(self) -> TUMDatasetLoader:
        if self._loader is None:
            self._loader = TUMDatasetLoader(
                self.config.dataset_path,
                self.dataset_config
            )
        return self._loader
    
    @property
    def segmenter(self):
        if self._segmenter is None:
            if self.config.segmenter_type == "fastsam":
                try:
                    self._segmenter = FastSAMSegmenter(
                        model_name="FastSAM-s",
                        config=self.dataset_config
                    )
                except Exception as e:
                    print(f"FastSAM not available ({e})r")
        return self._segmenter
    
    @property
    def feature_extractor(self) -> CLIPFeatureExtractor:
        """Get or create feature extractor."""
        if self._feature_extractor is None:
            self._feature_extractor = CLIPFeatureExtractor(
                model_name=self.config.clip_model,
                config=self.dataset_config
            )
        return self._feature_extractor
    
    @property
    def fusion(self) -> SemanticFusion:
        """Get or create fusion module."""
        if self._fusion is None:
            self._fusion = SemanticFusion(
                self.loader.intrinsics,
                self.dataset_config
            )
        return self._fusion
    
    @property
    def visualizer(self):
        if self._visualizer is None:
            if self.config.visualizer_backend == "rerun":
                try:
                    self._visualizer = RerunVisualizer(
                        app_name="- Queryable 3D Map"
                    )
                except ImportError:
                    print("Rerun not available")
        return self._visualizer
    
    def run(self) -> SemanticPointCloud:
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print("Running Pipeline")
        print(f"{'='*60}\n")
        
        # Create semantic map
        self.semantic_pc = self.fusion.create_semantic_map(
            loader=self.loader,
            segmenter=self.segmenter,
            feature_extractor=self.feature_extractor,
            frame_skip=self.config.frame_skip,
            max_frames=self.config.max_frames,
            subsample=self.config.pixel_subsample
        )
        
        # Apply voxel downsampling
        if self.config.voxel_size > 0:
            pc_generator = PointCloudGenerator(self.loader.intrinsics, self.dataset_config)
            self.semantic_pc = pc_generator._voxel_downsample(self.semantic_pc)
            print(f"\nAfter voxel downsampling: {len(self.semantic_pc):,} points")
        
        # Initialize query engine
        self.query_engine = QueryEngine(self.semantic_pc, self.feature_extractor)
        
        # Save if requested
        if self.config.save_pointcloud:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "semantic_pointcloud.npz"
            self.semantic_pc.save(str(output_path))
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*60}")
        print("Pipeline Complete!")
        print(f"{'='*60}")
        print(f"  Total time: {elapsed:.1f}s")
        print(f"  Points: {len(self.semantic_pc):,}")
        print(f"  Embedding dim: {self.semantic_pc.embeddings.shape[1]}")
        
        # Visualize if requested
        if self.config.visualize:
            self.visualize()
        
        return self.semantic_pc
    
    def visualize(self):
        if self.semantic_pc is None:
            print("No point cloud to visualize. Run pipeline first.")
            return
        
        print("\nVisualizing point cloud...")
        
        if isinstance(self.visualizer, RerunVisualizer):
            self.visualizer.log_point_cloud(
                self.semantic_pc,
                entity_path="world/semantic_map",
                point_size=0.01
            )
            
            # Log camera trajectory
            poses = []
            for idx in range(0, len(self.loader), self.config.frame_skip):
                frame = self.loader[idx]
                poses.append(frame.pose_matrix)
            
            if len(poses) > 0:
                self.visualizer.log_camera_trajectory(
                    poses,
                    entity_path="world/camera_path"
                )
        else:
            self.visualizer.visualize_point_cloud(self.semantic_pc)
    
    def query(self, text: str, threshold: float = None, 
              top_k: int = None, percentile: float = 95.0,
              visualize: bool = True):
        """
        Query the semantic map with natural language.
        
        Args:
            text: Natural language query
            threshold: Fixed similarity threshold (None = use percentile)
            top_k: Return top-k points (overrides threshold/percentile)
            percentile: Return top X% of points (default: top 5%)
            visualize: Whether to visualize results
            
        Returns:
            QueryResult
        """
        if self.query_engine is None:
            raise RuntimeError("Pipeline must be run before querying")
        
        result = self.query_engine.query(text, threshold=threshold, 
                                         top_k=top_k, percentile=percentile)
        
        print(f"\nQuery: '{text}'")
        print(f"  Matches: {result.num_matches:,} points")
        print(f"  Max similarity: {result.max_similarity:.4f}")
        print(f"  Mean similarity: {result.mean_similarity:.4f}")
        
        if result.centroid is not None:
            print(f"  Centroid: ({result.centroid[0]:.2f}, {result.centroid[1]:.2f}, {result.centroid[2]:.2f})")
        
        if visualize and result.num_matches > 0:
            if isinstance(self.visualizer, RerunVisualizer):
                self.visualizer.log_query_result(
                    self.semantic_pc,
                    result,
                    entity_path=f"world/query/{text.replace(' ', '_')[:20]}",
                    show_heatmap=True
                )
            else:
                self.visualizer.visualize_query_result(self.semantic_pc, result)
        
        return result
    
    def interactive_session(self):
        if self.query_engine is None:
            raise RuntimeError("Pipeline must be run before querying")
        
        self.query_engine.interactive_query(self.visualizer, self.semantic_pc)
    
    def load_cached(self, path: str):
        """
        Load a previously saved semantic point cloud.
        
        Args:
            path: Path to saved .npz file
        """
        self.semantic_pc = SemanticPointCloud.load(path)
        self.query_engine = QueryEngine(self.semantic_pc, self.feature_extractor)
        
        print(f"Loaded cached point cloud from {path}")
        print(f"  Points: {len(self.semantic_pc):,}")


def create_pipeline(dataset_path: str,
                    config_path: Optional[str] = None,
                    **kwargs) -> Pipeline:
    config = PipelineConfig(
        dataset_path=dataset_path,
        config_path=config_path,
        **kwargs
    )
    return Pipeline(config)


class Options:
    def __init__(self):    
        parser = argparse.ArgumentParser(
            description="Pipeline - Create Queryable 3D Semantic Maps"
        )
        parser.add_argument(
            "--dataset", "-d",
            default="data/rgbd_dataset_freiburg3_long_office_household",
            help="Path to TUM RGB-D dataset"
        )
        parser.add_argument(
            "--config", "-c",
            default="config/tum_freiburg3.yaml",
            help="Path to YAML configuration"
        )
        parser.add_argument(
            "--frame-skip", "-s",
            type=int, default=10,
            help="Process every nth frame"
        )
        parser.add_argument(
            "--max-frames", "-m",
            type=int, default=None,
            help="Maximum frames to process"
        )
        parser.add_argument(
            "--output", "-o",
            default="output",
            help="Output directory"
        )
        parser.add_argument(
            "--no-visualize",
            action="store_true",
            help="Disable visualization"
        )
        parser.add_argument(
            "--query", "-q",
            nargs="+",
            help="Queries to run after building map"
        )
        parser.add_argument(
            "--interactive", "-i",
            action="store_true",
            help="Start interactive query session"
        )
        parser.add_argument(
            "--load",
            help="Load cached point cloud instead of processing"
        )

        self.parser = parser
    
    def parse(self):
        args = self.parser.parse_args()
        return args


def main():
    args = Options().parse()
    
    # Create pipeline
    pipeline = create_pipeline(
        dataset_path=args.dataset,
        config_path=args.config,
        frame_skip=args.frame_skip,
        max_frames=args.max_frames,
        output_dir=args.output,
        visualize=not args.no_visualize
    )
    
    if args.load:
        pipeline.load_cached(args.load)
        if not args.no_visualize:
            pipeline.visualize()
    else:
        pipeline.run()
    
    if args.query:
        for query in args.query:
            pipeline.query(query)
    
    if args.interactive:
        pipeline.interactive_session()


if __name__ == "__main__":
    main()
