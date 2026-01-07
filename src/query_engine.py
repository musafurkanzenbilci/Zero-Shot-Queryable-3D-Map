import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from point_cloud import SemanticPointCloud
from feature_extractor import CLIPFeatureExtractor
from visualizer import RerunVisualizer


@dataclass
class QueryResult:
    query_text: str
    query_embedding: np.ndarray
    
    # Matching points
    point_indices: np.ndarray  # Indices of matching points
    similarities: np.ndarray  # Cosine similarities
    
    # Statistics
    max_similarity: float
    mean_similarity: float
    num_matches: int
    
    # Matched 3D region
    centroid: Optional[np.ndarray] = None  # Center of matched region
    bbox_min: Optional[np.ndarray] = None  # Bounding box minimum
    bbox_max: Optional[np.ndarray] = None  # Bounding box maximum


class QueryEngine:
    """
    Natural language query interface for semantic 3D maps.
    
    Supports:
    - Single text queries
    - Multi-query search (find regions matching all queries)
    - Negative queries (exclude certain concepts)
    - Spatial filtering
    """
    
    def __init__(self, semantic_pc: SemanticPointCloud,
                 feature_extractor: CLIPFeatureExtractor):
        self.semantic_pc = semantic_pc
        self.feature_extractor = feature_extractor
        
        # Precompute normalized embeddings for efficient search
        self._normalized_embeddings = None
        self._prepare_embeddings()
        
        print(f"[QueryEngine] Initialized")
        print(f"  Points: {len(semantic_pc):,}")
        print(f"  Embedding dim: {semantic_pc.embeddings.shape[1]}")
    
    def _prepare_embeddings(self):
        if len(self.semantic_pc.embeddings) == 0:
            self._normalized_embeddings = np.empty((0, 512))
            return
        
        # Handle NaN/Inf values in embeddings
        embeddings = self.semantic_pc.embeddings.copy()
        nan_mask = ~np.isfinite(embeddings).all(axis=1)
        if nan_mask.any():
            print(f"  Warning: {nan_mask.sum()} embeddings contain NaN/Inf, replacing with zeros")
            embeddings[nan_mask] = 0
        
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self._normalized_embeddings = embeddings / np.maximum(norms, 1e-8)
    
    def query(self, text: str, 
              top_k: Optional[int] = None,
              threshold: Optional[float] = None,
              percentile: float = 95.0,
              return_all_similarities: bool = False) -> QueryResult:
        """
        Query the semantic map with natural language.
        
        Args:
            text: Natural language query (e.g., "red chair", "coffee mug")
            top_k: Return top-k most similar points (None = use threshold/percentile)
            threshold: Fixed similarity threshold (0-1). If None, uses percentile.
            percentile: Return top X% of points by similarity (default: top 5%)
            return_all_similarities: Whether to return similarities for all points
            
        Returns:
            QueryResult with matching points and statistics
        """
        # Encode query text
        query_embedding = self.feature_extractor.encode_text(text)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Compute similarities
        similarities = self._normalized_embeddings @ query_embedding
        
        # Handle NaN values in similarities
        valid_mask = np.isfinite(similarities)
        if not valid_mask.all():
            similarities[~valid_mask] = -1  # Set invalid to low similarity
        
        # Select matching points
        if top_k is not None:
            # Get top-k indices
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            point_indices = top_indices
            match_similarities = similarities[top_indices]
        elif threshold is not None:
            # Use fixed threshold
            mask = similarities >= threshold
            point_indices = np.where(mask)[0]
            match_similarities = similarities[point_indices]
            
            # Sort by similarity
            sort_idx = np.argsort(match_similarities)[::-1]
            point_indices = point_indices[sort_idx]
            match_similarities = match_similarities[sort_idx]
        else:
            # Use percentile-based threshold (more adaptive)
            auto_threshold = np.percentile(similarities[valid_mask], percentile)
            mask = similarities >= auto_threshold
            point_indices = np.where(mask)[0]
            match_similarities = similarities[point_indices]
            
            # Sort by similarity
            sort_idx = np.argsort(match_similarities)[::-1]
            point_indices = point_indices[sort_idx]
            match_similarities = match_similarities[sort_idx]
        
        # Compute statistics
        result = QueryResult(
            query_text=text,
            query_embedding=query_embedding,
            point_indices=point_indices,
            similarities=match_similarities if not return_all_similarities else similarities,
            max_similarity=float(similarities.max()) if len(similarities) > 0 else 0.0,
            mean_similarity=float(match_similarities.mean()) if len(match_similarities) > 0 else 0.0,
            num_matches=len(point_indices)
        )
        
        # Compute spatial extent of matches
        if len(point_indices) > 0:
            matched_points = self.semantic_pc.points[point_indices]
            result.centroid = matched_points.mean(axis=0)
            result.bbox_min = matched_points.min(axis=0)
            result.bbox_max = matched_points.max(axis=0)
        
        return result
    
    def query_multiple(self, queries: List[str],
                       combine_mode: str = "intersection",
                       threshold: float = 0.2) -> QueryResult:
        """
        Query with multiple text descriptions.
        
        Args:
            queries: List of query strings
            combine_mode: "intersection" (AND) or "union" (OR)
            threshold: Similarity threshold per query
            
        Returns:
            Combined QueryResult
        """
        if len(queries) == 0:
            raise ValueError("Need at least one query")
        
        # Get individual results
        results = [self.query(q, threshold=threshold) for q in queries]
        
        # Combine indices
        if combine_mode == "intersection":
            # Points must match ALL queries
            common_indices = set(results[0].point_indices)
            for r in results[1:]:
                common_indices &= set(r.point_indices)
            point_indices = np.array(list(common_indices))
        else:  # union
            # Points match ANY query
            all_indices = set()
            for r in results:
                all_indices |= set(r.point_indices)
            point_indices = np.array(list(all_indices))
        
        if len(point_indices) == 0:
            return QueryResult(
                query_text=" AND ".join(queries) if combine_mode == "intersection" else " OR ".join(queries),
                query_embedding=np.mean([r.query_embedding for r in results], axis=0),
                point_indices=np.array([]),
                similarities=np.array([]),
                max_similarity=0.0,
                mean_similarity=0.0,
                num_matches=0
            )
        
        # Compute combined similarity (average across queries)
        combined_sim = np.zeros(len(point_indices))
        for i, idx in enumerate(point_indices):
            sims = []
            for r in results:
                # Get similarity from each query result
                all_sims = self._normalized_embeddings @ r.query_embedding
                sims.append(all_sims[idx])
            combined_sim[i] = np.mean(sims)
        
        # Sort by combined similarity
        sort_idx = np.argsort(combined_sim)[::-1]
        point_indices = point_indices[sort_idx]
        combined_sim = combined_sim[sort_idx]
        
        # Build result
        result = QueryResult(
            query_text=" AND ".join(queries) if combine_mode == "intersection" else " OR ".join(queries),
            query_embedding=np.mean([r.query_embedding for r in results], axis=0),
            point_indices=point_indices,
            similarities=combined_sim,
            max_similarity=float(combined_sim.max()) if len(combined_sim) > 0 else 0.0,
            mean_similarity=float(combined_sim.mean()) if len(combined_sim) > 0 else 0.0,
            num_matches=len(point_indices)
        )
        
        if len(point_indices) > 0:
            matched_points = self.semantic_pc.points[point_indices]
            result.centroid = matched_points.mean(axis=0)
            result.bbox_min = matched_points.min(axis=0)
            result.bbox_max = matched_points.max(axis=0)
        
        return result
    
    def query_with_negation(self, positive_query: str,
                            negative_queries: List[str],
                            threshold: float = 0.2,
                            negative_threshold: float = 0.3) -> QueryResult:
        """
        Query with positive and negative constraints.
        
        Finds points matching positive_query but NOT matching negative_queries.
        
        Args:
            positive_query: What to find
            negative_queries: What to exclude
            threshold: Similarity threshold for positive match
            negative_threshold: Similarity threshold for negative exclusion
        """
        # Get positive matches
        positive_result = self.query(positive_query, threshold=threshold)
        
        if len(positive_result.point_indices) == 0:
            return positive_result
        
        # Get negative matches to exclude
        exclude_indices = set()
        for neg_query in negative_queries:
            neg_result = self.query(neg_query, threshold=negative_threshold)
            exclude_indices |= set(neg_result.point_indices)
        
        # Filter out negative matches
        filtered_indices = [idx for idx in positive_result.point_indices 
                           if idx not in exclude_indices]
        filtered_indices = np.array(filtered_indices)
        
        if len(filtered_indices) == 0:
            return QueryResult(
                query_text=f"{positive_query} NOT ({', '.join(negative_queries)})",
                query_embedding=positive_result.query_embedding,
                point_indices=np.array([]),
                similarities=np.array([]),
                max_similarity=0.0,
                mean_similarity=0.0,
                num_matches=0
            )
        
        # Get similarities for filtered indices
        filtered_sims = self._normalized_embeddings[filtered_indices] @ positive_result.query_embedding
        
        result = QueryResult(
            query_text=f"{positive_query} NOT ({', '.join(negative_queries)})",
            query_embedding=positive_result.query_embedding,
            point_indices=filtered_indices,
            similarities=filtered_sims,
            max_similarity=float(filtered_sims.max()),
            mean_similarity=float(filtered_sims.mean()),
            num_matches=len(filtered_indices)
        )
        
        matched_points = self.semantic_pc.points[filtered_indices]
        result.centroid = matched_points.mean(axis=0)
        result.bbox_min = matched_points.min(axis=0)
        result.bbox_max = matched_points.max(axis=0)
        
        return result
    
    def spatial_query(self, text: str,
                      center: np.ndarray,
                      radius: float,
                      threshold: float = 0.2) -> QueryResult:
        """
        Query within a spatial region.
        
        Args:
            text: Query text
            center: 3D center point
            radius: Search radius in meters
            threshold: Similarity threshold
        """
        # Get text matches
        result = self.query(text, threshold=threshold)
        
        if len(result.point_indices) == 0:
            return result
        
        # Filter by spatial distance
        matched_points = self.semantic_pc.points[result.point_indices]
        distances = np.linalg.norm(matched_points - center, axis=1)
        spatial_mask = distances <= radius
        
        filtered_indices = result.point_indices[spatial_mask]
        filtered_sims = result.similarities[spatial_mask]
        
        result.point_indices = filtered_indices
        result.similarities = filtered_sims
        result.num_matches = len(filtered_indices)
        
        if len(filtered_indices) > 0:
            matched_points = self.semantic_pc.points[filtered_indices]
            result.centroid = matched_points.mean(axis=0)
            result.bbox_min = matched_points.min(axis=0)
            result.bbox_max = matched_points.max(axis=0)
            result.max_similarity = float(filtered_sims.max())
            result.mean_similarity = float(filtered_sims.mean())
        
        return result
    
    def get_similarity_heatmap(self, text: str) -> np.ndarray:
        query_embedding = self.feature_extractor.encode_text(text)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        return self._normalized_embeddings @ query_embedding
    
    def interactive_query(self, visualizer:RerunVisualizer = None, semantic_pc:SemanticPointCloud = None):
        print("\n" + "="*60)
        print("Interactive Query Session")
        print("="*60)
        print("Enter natural language queries to search the 3D map.")
        print("Commands:")
        print("  'exit' - End session")
        print("  'top N' - Show top N results (e.g., 'top 100')")
        print("  'threshold X' - Set similarity threshold (e.g., 'threshold 0.3')")
        print()

        
        threshold = 0.2
        top_k = None
        
        while True:
            try:
                query = input("Query> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nSession ended.")
                break
            
            if not query:
                continue
            
            if query.lower() == 'exit':
                print("Session ended.")
                break
            
            if query.lower().startswith('top '):
                try:
                    top_k = int(query.split()[1])
                    print(f"  Set top_k = {top_k}")
                except:
                    print("  Usage: top N")
                continue
            
            if query.lower().startswith('threshold '):
                try:
                    threshold = float(query.split()[1])
                    print(f"  Set threshold = {threshold}")
                except:
                    print("  Usage: threshold X")
                continue
            
            # Run query
            result = self.query(query, top_k=top_k, threshold=threshold)
            if visualizer and semantic_pc:
                heatmap = self.get_similarity_heatmap(query)
                visualizer.log_similarity_heatmap(semantic_pc, heatmap)
                visualizer.log_capsule(semantic_pc, [result.centroid])
            
            print(f"\n  Results for '{query}':")
            print(f"  Matches: {result.num_matches:,} points")
            print(f"  Max similarity: {result.max_similarity:.4f}")
            print(f"  Mean similarity: {result.mean_similarity:.4f}")
            
            if result.centroid is not None:
                print(f"  Centroid: ({result.centroid[0]:.2f}, {result.centroid[1]:.2f}, {result.centroid[2]:.2f})")
            print()


def test_query_engine():
    """Test the query engine."""
    print("Query Engine Test")
    print("="*60)

    from pathlib import Path
    import yaml
    from data_loader import TUMDatasetLoader
    from fusion import SemanticFusion
    from segmentation import FastSAMSegmenter


    # Initialize feature extractor
    feature_extractor = CLIPFeatureExtractor(model_name="ViT-B/32")
    # Load config
    config_path = Path(__file__).parent.parent / "config" / "tum_freiburg3.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize loader
    dataset_path = Path(__file__).parent.parent / "data" / "rgbd_dataset_freiburg3_long_office_household"
    loader = TUMDatasetLoader(str(dataset_path), config)
    
    segmenter = FastSAMSegmenter(config=config)
    
    fusion = SemanticFusion(loader.intrinsics, config)

    semantic_pc = fusion.create_semantic_map(loader, segmenter, feature_extractor, frame_skip=10)

    # Create dummy semantic point cloud for testing
    # n_points = 1000
    # embedding_dim = 512
    
    # semantic_pc = SemanticPointCloud()
    # semantic_pc.points = np.random.randn(n_points, 3)
    # semantic_pc.colors = np.random.rand(n_points, 3)
    # semantic_pc.embeddings = np.random.randn(n_points, embedding_dim)
    
    # Normalize embeddings
    norms = np.linalg.norm(semantic_pc.embeddings, axis=1, keepdims=True)
    semantic_pc.embeddings = semantic_pc.embeddings / norms
    
    
    # Initialize query engine
    engine = QueryEngine(semantic_pc, feature_extractor)

    
    # Test queries
    test_queries = [
        "a red chair",
        "computer monitor",
        "coffee mug on desk",
        "yellow dice",
        "water bottle"
    ]
    
    print("\nTest Queries:")
    for query in test_queries:
        result = engine.query(query, threshold=0.0, top_k=50)
        print(f"  '{query}': {result.num_matches} matches, max_sim={result.max_similarity:.4f}")
    
    print("\n✅ Query Engine test completed!")


if __name__ == "__main__":
    test_query_engine()
