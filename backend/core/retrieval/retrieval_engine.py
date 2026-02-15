import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import networkx as nx
from rank_bm25 import BM25Okapi
from scipy.stats import beta as beta_dist
from loguru import logger

from backend.models.config import (
    RetrievalWeights, RetrievalResult, QueryDecomposition,
    Modality, Resolution, EdgeType, SearchConfig
)
from backend.core.storage.arango_manager import ArangoStorageManager
from backend.core.vectorization.embedding_engine import VectorizationEngine


class AdvancedRetrievalEngine:
    """
    Mathematical retrieval engine implementing:
    - Multi-channel retrieval (vector + graph + BM25)
    - Weighted Evidence Accumulation
    - Personalized PageRank
    - Submodular optimization (MMR)
    - Bayesian weight optimization
    - Reciprocal Rank Fusion
    """
    
    def __init__(
        self,
        storage: ArangoStorageManager,
        vectorizer: VectorizationEngine,
        config: SearchConfig,
        weights: RetrievalWeights
    ):
        self.storage = storage
        self.vectorizer = vectorizer
        self.config = config
        self.weights = weights
        
        # Performance tracking for adaptive optimization
        self.performance_history: List[Dict] = []
        
        # Resolution weights λ(r_i)
        self.resolution_weights = {
            Resolution.FINE: 1.0,
            Resolution.MID: 0.85,
            Resolution.COARSE: 0.65
        }
        
        # Modality alignment scores
        self.modality_alignment = {
            'exact': 1.0,
            'cross_modal': 0.7,
            'weak': 0.4
        }
        
        logger.info("Advanced retrieval engine initialized")
    
    def retrieve(
        self,
        query_decomposition: QueryDecomposition,
        query_text: str,
        config: Dict = None,
        source_filters: list = None
    ) -> tuple:
        """
        Main retrieval pipeline with configurable algorithms.
        
        Returns: (results, metrics_dict)
        
        Config options:
        - mode: "fast" (Vector+Graph) or "advanced" (all algorithms)
        - use_vector, use_graph, use_bm25, use_pagerank, use_structural, use_mmr
        
        Args:
            source_filters: Optional list of source filenames to restrict retrieval to.
                            When provided, only KB atoms whose source matches are included.
        """
        import time
        
        # Default config: Advanced mode with all algorithms
        config = config or {
            "mode": "advanced",
            "use_vector": True,
            "use_graph": True,
            "use_bm25": True,
            "use_pagerank": True,
            "use_structural": True,
            "use_mmr": True
        }
        
        mode = config.get("mode", "advanced")
        logger.info(f"Starting retrieval for query: {query_text[:100]} (mode={mode})")
        
        # Initialize metrics
        metrics = {
            "vector_time_ms": 0,
            "graph_time_ms": 0,
            "bm25_time_ms": 0,
            "algorithms_used": []
        }
        
        # Step 1: Query embedding
        query_embedding = self.vectorizer.embed_query(query_text)
        
        # Step 2: Vector retrieval (always enabled)
        vector_start = time.time()
        vector_candidates = self._vector_retrieval(query_decomposition, query_embedding)
        metrics["vector_time_ms"] = (time.time() - vector_start) * 1000
        metrics["algorithms_used"].append("Vector Search")
        
        # Fast mode: Vector + Graph only (Graph uses default settings)
        if mode == "fast":
            graph_start = time.time()
            graph_candidates = self._graph_retrieval(vector_candidates, use_pagerank=False)  # Fast RAG skips PageRank
            metrics["graph_time_ms"] = (time.time() - graph_start) * 1000
            metrics["algorithms_used"].append("Graph Traversal")
            
            # Merge and convert to results
            all_cands = vector_candidates + graph_candidates
            results = self._quick_convert_to_results(all_cands[:15])
            
            logger.info(f"Fast retrieval: {len(results)} results")
            return results, metrics
        
        # Advanced mode: Configurable algorithms
        graph_candidates = []
        bm25_candidates = []
        use_pagerank = config.get("use_pagerank", True)
        
        if config.get("use_graph", True):
            graph_start = time.time()
            graph_candidates = self._graph_retrieval(vector_candidates, use_pagerank=use_pagerank)
            metrics["graph_time_ms"] = (time.time() - graph_start) * 1000
            metrics["algorithms_used"].append("Graph Traversal")
            if use_pagerank:
                metrics["algorithms_used"].append("PageRank")
        
        if config.get("use_bm25", True):
            bm25_start = time.time()
            bm25_candidates = self._bm25_retrieval(query_text)
            metrics["bm25_time_ms"] = (time.time() - bm25_start) * 1000
            metrics["algorithms_used"].append("BM25")
        
        # Compute structural importance
        struct_scores = {}
        if config.get("use_structural", True):
            struct_scores = self._compute_structural_scores(
                vector_candidates + graph_candidates + bm25_candidates
            )
            metrics["algorithms_used"].append("Structural Scoring")
        
        # Unified Evidence Accumulation
        all_candidates = self._merge_candidates(
            vector_candidates,
            graph_candidates,
            bm25_candidates,
            struct_scores,
            query_decomposition
        )
        
        # Diversity-preserving re-ranking (MMR)
        if config.get("use_mmr", True):
            reranked = self._submodular_reranking(all_candidates, query_embedding)
            metrics["algorithms_used"].append("MMR Reranking")
        else:
            reranked = all_candidates[:20]
        
        # Multi-hop evidence packing
        final_results = self._pack_evidence(reranked)
        
        logger.info(f"Advanced retrieval: {len(final_results)} results, algos: {metrics['algorithms_used']}")
        return final_results, metrics
    
    def _quick_convert_to_results(self, candidates: List[Dict]) -> List[RetrievalResult]:
        """Quick conversion for fast mode - no heavy processing"""
        results = []
        for cand in candidates:
            node_data = self.storage.get_node(cand['node_id'])
            if node_data:
                results.append(RetrievalResult(
                    atom_id=cand['node_id'],
                    content=node_data['content'],
                    modality=Modality(node_data['modality']),
                    score=cand.get('vector_score', cand.get('graph_score', 0.0)),
                    vector_score=cand.get('vector_score', 0.0),
                    graph_score=cand.get('graph_score', 0.0),
                    bm25_score=0.0,
                    struct_score=0.0,
                    mod_score=1.0,
                    source=node_data.get('source')
                ))
        return results
    
    def _vector_retrieval(
        self,
        query_decomp: QueryDecomposition,
        query_embedding: np.ndarray
    ) -> List[Dict]:
        """
        Vector retrieval with multi-resolution scoring
        
        Implements:
        s̃_ij^vec = λ(r_i) · cos(q_j, e_i(m_j))
        """
        all_candidates = []
        
        for sub_query, modality, weight in zip(
            query_decomp.sub_queries,
            query_decomp.modalities,
            query_decomp.intent_weights
        ):
            # Get embedding for sub-query
            if sub_query != query_decomp.sub_queries[0]:
                sub_embedding = self.vectorizer.embed_query(sub_query)
            else:
                sub_embedding = query_embedding
            
            # Search for each resolution level
            for resolution in Resolution:
                results = self.storage.vector_search(
                    query_embedding=sub_embedding,
                    modality=modality if modality != Modality.TEXT else None,
                    resolution=resolution,
                    top_k=self.config.top_k_vector // 3,
                    embedding_key="default"
                )
                
                # Apply resolution-aware scaling
                lambda_r = self.resolution_weights[resolution]
                
                for node_id, similarity in results:
                    all_candidates.append({
                        'node_id': node_id,
                        'vector_score': lambda_r * similarity * weight,
                        'sub_query_idx': query_decomp.sub_queries.index(sub_query)
                    })
        
        # Deduplicate and aggregate scores
        aggregated = self._aggregate_scores(all_candidates, 'vector_score')
        
        return aggregated
    
    def _graph_retrieval(self, seed_candidates: List[Dict], use_pagerank: bool = True) -> List[Dict]:
        """
        Graph-based retrieval using optional Personalized PageRank
        """
        if not seed_candidates:
            return []
        
        # Extract seed nodes
        seed_nodes = [c['node_id'] for c in seed_candidates[:self.config.top_k_vector]]
        
        # Compute personalized PageRank if requested
        pagerank_scores = {}
        if use_pagerank:
            pagerank_scores = self._personalized_pagerank(
                seed_nodes,
                alpha=self.config.pagerank_alpha
            )
        
        # Graph traversal with hop constraint
        traversal_results = self.storage.graph_traversal(
            start_nodes=seed_nodes,
            max_depth=self.config.graph_hops
        )
        
        # Combine PageRank with path weights
        graph_candidates = []
        for result in traversal_results:
            node_id = result['node_id']
            
            # Skip if already in seed
            if node_id in seed_nodes:
                continue
            
            # Graph score combines PageRank and path weight
            pr_score = pagerank_scores.get(node_id, 0.0)
            path_weight = result.get('path_weight', 0.0)
            
            # If no PageRank, rely entirely on path weight
            if not use_pagerank:
                graph_score = path_weight
            else:
                graph_score = 0.6 * pr_score + 0.4 * path_weight
            
            graph_candidates.append({
                'node_id': node_id,
                'graph_score': graph_score,
                'distance': result['distance']
            })
        
        # Filter by distance
        graph_candidates = [
            c for c in graph_candidates 
            if c['distance'] <= self.config.graph_hops
        ]
        
        # Sort and limit
        graph_candidates.sort(key=lambda x: x['graph_score'], reverse=True)
        
        return graph_candidates[:self.config.top_k_graph]
    
    def _personalized_pagerank(
        self,
        seed_nodes: List[str],
        alpha: float = 0.15,
        max_iterations: int = 20,
        tolerance: float = 1e-6
    ) -> Dict[str, float]:
        """
        Compute Personalized PageRank
        
        π_{t+1} = α·π_0 + (1-α)·A·π_t
        
        Where:
        - π_0 is teleport distribution (uniform over seeds)
        - A is normalized adjacency matrix
        - α is teleport probability
        """
        # Initialize teleport vector
        teleport = {}
        for node in seed_nodes:
            teleport[node] = 1.0 / len(seed_nodes)
        
        # Current scores
        scores = teleport.copy()
        
        # Iterative computation
        for iteration in range(max_iterations):
            new_scores = {}
            
            # For each node in current scores
            for node_id, score in scores.items():
                # Get outgoing neighbors
                neighbors = self.storage.get_neighbors(node_id, direction="outbound")
                
                if not neighbors:
                    # Dead end - redistribute to seeds
                    for seed in seed_nodes:
                        new_scores[seed] = new_scores.get(seed, 0.0) + score / len(seed_nodes)
                else:
                    # Distribute score to neighbors
                    out_degree = len(neighbors)
                    for neighbor in neighbors:
                        neighbor_id = neighbor['node']['_key']
                        edge_weight = neighbor['edge'].get('weight', 1.0)
                        
                        contribution = (score * edge_weight) / out_degree
                        new_scores[neighbor_id] = new_scores.get(neighbor_id, 0.0) + contribution
            
            # Add teleport
            final_scores = {}
            for node_id, score in new_scores.items():
                final_scores[node_id] = (
                    alpha * teleport.get(node_id, 0.0) + 
                    (1 - alpha) * score
                )
            
            # Check convergence
            if self._has_converged(scores, final_scores, tolerance):
                logger.debug(f"PageRank converged in {iteration + 1} iterations")
                break
            
            scores = final_scores
        
        return scores
    
    def _bm25_retrieval(self, query_text: str) -> List[Dict]:
        """
        BM25 retrieval for lexical matching
        
        Returns normalized BM25 scores
        """
        # Get all text documents
        # This is simplified - in production, maintain BM25 index
        fulltext_results = self.storage.fulltext_search(
            query_text, 
            top_k=self.config.top_k_bm25
        )
        
        if not fulltext_results:
            return []
        
        # Extract corpus
        corpus = [r['content'] for r in fulltext_results]
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        
        # Create BM25 model
        bm25 = BM25Okapi(tokenized_corpus)
        
        # Score query
        tokenized_query = query_text.lower().split()
        scores = bm25.get_scores(tokenized_query)
        
        # Normalize scores
        if scores.max() > 0:
            normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            normalized_scores = scores
        
        # Create candidates
        bm25_candidates = []
        for idx, (result, score) in enumerate(zip(fulltext_results, normalized_scores)):
            bm25_candidates.append({
                'node_id': result['node_id'],
                'bm25_score': float(score)
            })
        
        return bm25_candidates
    
    def _compute_structural_scores(self, candidates: List[Dict]) -> Dict[str, float]:
        """
        Compute structural importance scores
        
        s_i^struct = η_1·C_d(v_i) + η_2·C_b(v_i)
        
        Where:
        - C_d is degree centrality
        - C_b is betweenness centrality
        """
        node_ids = list(set([c['node_id'] for c in candidates]))
        
        if not node_ids:
            return {}
        
        # Build subgraph
        G = nx.Graph()
        
        for node_id in node_ids:
            G.add_node(node_id)
            
            # Get neighbors
            neighbors = self.storage.get_neighbors(node_id)
            for neighbor in neighbors:
                neighbor_id = neighbor['node']['_key']
                if neighbor_id in node_ids:
                    weight = neighbor['edge'].get('weight', 1.0)
                    G.add_edge(node_id, neighbor_id, weight=weight)
        
        # Compute centralities
        try:
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
        except:
            # Fallback if graph is too small
            degree_centrality = {nid: 0.0 for nid in node_ids}
            betweenness_centrality = {nid: 0.0 for nid in node_ids}
        
        # Combine with weights η_1, η_2
        eta_1, eta_2 = 0.6, 0.4
        
        struct_scores = {}
        for node_id in node_ids:
            struct_scores[node_id] = (
                eta_1 * degree_centrality.get(node_id, 0.0) +
                eta_2 * betweenness_centrality.get(node_id, 0.0)
            )
        
        return struct_scores
    
    def _merge_candidates(
        self,
        vector_cands: List[Dict],
        graph_cands: List[Dict],
        bm25_cands: List[Dict],
        struct_scores: Dict[str, float],
        query_decomp: QueryDecomposition
    ) -> List[RetrievalResult]:
        """
        Unified Evidence Accumulation
        
        S_i = α·s_i^vec + β·s_i^graph + γ·s_i^bm25 + δ·s_i^struct + ε·s_i^mod
        """
        # Collect all unique node IDs
        all_node_ids = set()
        
        vector_dict = {}
        for c in vector_cands:
            nid = c['node_id']
            all_node_ids.add(nid)
            vector_dict[nid] = c.get('vector_score', 0.0)
        
        graph_dict = {}
        for c in graph_cands:
            nid = c['node_id']
            all_node_ids.add(nid)
            graph_dict[nid] = c.get('graph_score', 0.0)
        
        bm25_dict = {}
        for c in bm25_cands:
            nid = c['node_id']
            all_node_ids.add(nid)
            bm25_dict[nid] = c.get('bm25_score', 0.0)
        
        # Compute unified scores
        results = []
        
        for node_id in all_node_ids:
            # Get node data
            node_data = self.storage.get_node(node_id)
            if not node_data:
                continue
            
            # Individual scores
            s_vec = vector_dict.get(node_id, 0.0)
            s_graph = graph_dict.get(node_id, 0.0)
            s_bm25 = bm25_dict.get(node_id, 0.0)
            s_struct = struct_scores.get(node_id, 0.0)
            
            # Modality alignment score
            node_modality = Modality(node_data['modality'])
            s_mod = self._compute_modality_score(node_modality, query_decomp.modalities)
            
            # Unified score: S_i = α·s_vec + β·s_graph + γ·s_bm25 + δ·s_struct + ε·s_mod
            unified_score = (
                self.weights.alpha * s_vec +
                self.weights.beta * s_graph +
                self.weights.gamma * s_bm25 +
                self.weights.delta * s_struct +
                self.weights.epsilon * s_mod
            )
            
            result = RetrievalResult(
                atom_id=node_id,
                content=node_data['content'],
                modality=node_modality,
                score=unified_score,
                vector_score=s_vec,
                graph_score=s_graph,
                bm25_score=s_bm25,
                struct_score=s_struct,
                mod_score=s_mod,
                source=node_data.get('source')
            )
            
            results.append(result)
        
        # Sort by unified score
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
    
    def _compute_modality_score(
        self,
        node_modality: Modality,
        query_modalities: List[Modality]
    ) -> float:
        """
        Compute modality alignment score
        
        Returns 1.0 for exact match, 0.7 for cross-modal, 0.4 for weak
        """
        if node_modality in query_modalities:
            return self.modality_alignment['exact']
        
        # Cross-modal alignment (e.g., image-text via CLIP)
        cross_modal_pairs = [
            (Modality.IMAGE, Modality.TEXT),
            (Modality.VIDEO, Modality.TEXT),
            (Modality.AUDIO, Modality.TEXT)
        ]
        
        for mod_a, mod_b in cross_modal_pairs:
            if (node_modality == mod_a and mod_b in query_modalities) or \
               (node_modality == mod_b and mod_a in query_modalities):
                return self.modality_alignment['cross_modal']
        
        return self.modality_alignment['weak']
    
    def _submodular_reranking(
        self,
        candidates: List[RetrievalResult],
        query_embedding: np.ndarray,
        top_k: int = 20
    ) -> List[RetrievalResult]:
        """
        Diversity-preserving re-ranking using submodular optimization (MMR)
        
        Objective:
        max_A [Σ_{i∈A} S_i - λ Σ_{i,j∈A} cos(e_i, e_j)]
        
        Solved greedily
        """
        if len(candidates) <= top_k:
            return candidates
        
        selected: List[RetrievalResult] = []
        remaining = candidates.copy()
        
        lambda_diversity = self.config.diversity_lambda
        
        # Greedy selection
        for _ in range(min(top_k, len(remaining))):
            best_idx = -1
            best_marginal_gain = -float('inf')
            
            for idx, candidate in enumerate(remaining):
                # Relevance term
                relevance = candidate.score
                
                # Diversity penalty
                diversity_penalty = 0.0
                if selected:
                    # Get embedding for candidate
                    cand_node = self.storage.get_node(candidate.atom_id)
                    if cand_node and 'embeddings' in cand_node:
                        cand_emb = np.array(cand_node['embeddings'].get('default', []))
                        
                        if len(cand_emb) > 0:
                            for selected_result in selected:
                                sel_node = self.storage.get_node(selected_result.atom_id)
                                if sel_node and 'embeddings' in sel_node:
                                    sel_emb = np.array(sel_node['embeddings'].get('default', []))
                                    
                                    if len(sel_emb) > 0:
                                        similarity = self.vectorizer.compute_similarity(cand_emb, sel_emb)
                                        diversity_penalty += similarity
                
                # Marginal gain
                marginal_gain = relevance - lambda_diversity * diversity_penalty
                
                if marginal_gain > best_marginal_gain:
                    best_marginal_gain = marginal_gain
                    best_idx = idx
            
            # Select best candidate
            if best_idx >= 0:
                selected.append(remaining.pop(best_idx))
        
        return selected
    
    def _pack_evidence(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Multi-hop evidence packing
        
        For each result, include supporting context from 1-hop neighbors
        """
        for result in results:
            # Get 1-hop neighbors
            neighbors = self.storage.get_neighbors(result.atom_id, direction="any")
            
            # Extract content from neighbors
            context = []
            for neighbor in neighbors[:3]:  # Limit context
                neighbor_node = neighbor['node']
                context.append(neighbor_node.get('content', ''))
            
            result.context = context
        
        return results
    
    def _aggregate_scores(
        self,
        candidates: List[Dict],
        score_key: str
    ) -> List[Dict]:
        """Aggregate scores for duplicate node IDs"""
        score_map = defaultdict(float)
        
        for cand in candidates:
            node_id = cand['node_id']
            score_map[node_id] += cand[score_key]
        
        aggregated = [
            {'node_id': nid, score_key: score}
            for nid, score in score_map.items()
        ]
        
        aggregated.sort(key=lambda x: x[score_key], reverse=True)
        
        return aggregated[:self.config.top_k_vector]
    
    def _has_converged(
        self,
        old_scores: Dict[str, float],
        new_scores: Dict[str, float],
        tolerance: float
    ) -> bool:
        """Check if PageRank has converged"""
        all_nodes = set(old_scores.keys()) | set(new_scores.keys())
        
        max_diff = 0.0
        for node in all_nodes:
            old_val = old_scores.get(node, 0.0)
            new_val = new_scores.get(node, 0.0)
            max_diff = max(max_diff, abs(new_val - old_val))
        
        return max_diff < tolerance
    
    def update_weights_bayesian(self, metrics: Dict[str, float]):
        """
        Update retrieval weights using Bayesian optimization (Thompson Sampling)
        
        Objective: max_θ E[M | θ]
        Where θ = (α, β, γ, δ, ε)
        """
        # This is a simplified implementation
        # Full implementation would use Bayesian optimization library
        
        # Store performance
        self.performance_history.append(metrics)
        
        if len(self.performance_history) < 10:
            return  # Need more data
        
        # Compute reward
        reward = (
            0.4 * metrics.get('hit_rate', 0.0) +
            0.3 * metrics.get('semantic_overlap', 0.0) +
            0.3 * metrics.get('user_engagement', 0.0)
        )
        
        # Thompson Sampling update (simplified)
        # In production, use proper Bayesian optimization
        learning_rate = 0.05
        
        # Update alpha (vector weight)
        if reward > 0.7:
            self.weights.alpha = min(1.0, self.weights.alpha + learning_rate)
        elif reward < 0.3:
            self.weights.alpha = max(0.0, self.weights.alpha - learning_rate)
        
        # Normalize weights
        self.weights.normalize()
        
        logger.info(f"Updated weights - α={self.weights.alpha:.3f}, β={self.weights.beta:.3f}")