import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import spacy
from loguru import logger

from backend.models.config import (
    KnowledgeAtom, GraphEdge, EdgeType, 
    Modality, Resolution
)
from backend.core.vectorization.embedding_engine import VectorizationEngine


class GraphConstructionEngine:
    """
    Heterogeneous, weighted, directed multigraph construction
    
    Implements:
    G = (V, E, T_V, T_E, W)
    
    Where:
    - V: nodes (knowledge atoms)
    - E: edges
    - T_V: node types
    - T_E: edge types
    - W: edge weights
    """
    
    def __init__(self, vectorizer: VectorizationEngine):
        self.vectorizer = vectorizer
        
        # Load spaCy for entity extraction
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("spaCy model not loaded, entity extraction disabled")
            self.nlp = None
        
        # Similarity thresholds for different modalities
        self.similarity_thresholds = {
            Modality.TEXT: 0.7,
            Modality.IMAGE: 0.75,
            Modality.AUDIO: 0.7,
            Modality.VIDEO: 0.75,
            Modality.TABLE: 0.7
        }
        
        # k for k-NN graph
        self.k_neighbors = 5
        
        logger.info("Graph construction engine initialized")
    
    def construct_edges_for_atoms(
        self,
        atoms: List[KnowledgeAtom],
        atom_ids: List[str]
    ) -> List[GraphEdge]:
        """
        Construct all edges for a batch of knowledge atoms
        
        Returns list of GraphEdge objects
        """
        edges = []
        
        # 1. Structural edges
        structural_edges = self._create_structural_edges(atoms, atom_ids)
        edges.extend(structural_edges)
        
        # 2. Semantic similarity edges
        semantic_edges = self._create_semantic_edges(atoms, atom_ids)
        edges.extend(semantic_edges)
        
        # 3. Entity-based edges
        entity_edges = self._create_entity_edges(atoms, atom_ids)
        edges.extend(entity_edges)
        
        # 4. Cross-modal edges
        cross_modal_edges = self._create_cross_modal_edges(atoms, atom_ids)
        edges.extend(cross_modal_edges)
        
        # 5. Normalize edge weights
        edges = self._normalize_edge_weights(edges)
        
        logger.info(f"Constructed {len(edges)} edges for {len(atoms)} atoms")
        
        return edges
    
    def _create_structural_edges(
        self,
        atoms: List[KnowledgeAtom],
        atom_ids: List[str]
    ) -> List[GraphEdge]:
        """
        Create structural edges: PART_OF, NEXT, DERIVED_FROM
        
        These form the hierarchical and temporal structure
        """
        edges = []
        
        # Group atoms by source and resolution
        source_groups = defaultdict(lambda: defaultdict(list))
        for atom, atom_id in zip(atoms, atom_ids):
            source_groups[atom.source][atom.resolution].append((atom, atom_id))
        
        # For each source (document/video/etc)
        for source, resolution_dict in source_groups.items():
            
            # PART_OF edges: fine -> mid -> coarse
            if Resolution.FINE in resolution_dict and Resolution.MID in resolution_dict:
                fine_atoms = resolution_dict[Resolution.FINE]
                mid_atoms = resolution_dict[Resolution.MID]
                
                # Create PART_OF edges from fine to mid
                # Simple heuristic: sequential assignment
                for fine_atom, fine_id in fine_atoms:
                    # Find closest mid-level atom (by position/timestamp)
                    closest_mid_id = self._find_closest_parent(
                        fine_atom, mid_atoms
                    )
                    if closest_mid_id:
                        edges.append(GraphEdge(
                            source_id=fine_id,
                            target_id=closest_mid_id,
                            edge_type=EdgeType.PART_OF,
                            weight=1.0
                        ))
            
            if Resolution.MID in resolution_dict and Resolution.COARSE in resolution_dict:
                mid_atoms = resolution_dict[Resolution.MID]
                coarse_atoms = resolution_dict[Resolution.COARSE]
                
                for mid_atom, mid_id in mid_atoms:
                    closest_coarse_id = self._find_closest_parent(
                        mid_atom, coarse_atoms
                    )
                    if closest_coarse_id:
                        edges.append(GraphEdge(
                            source_id=mid_id,
                            target_id=closest_coarse_id,
                            edge_type=EdgeType.PART_OF,
                            weight=1.0
                        ))
            
            # NEXT edges: temporal adjacency
            for resolution, atoms_list in resolution_dict.items():
                sorted_atoms = sorted(atoms_list, key=lambda x: x[0].timestamp)
                
                for i in range(len(sorted_atoms) - 1):
                    curr_atom, curr_id = sorted_atoms[i]
                    next_atom, next_id = sorted_atoms[i + 1]
                    
                    # Compute time difference
                    delta_t = next_atom.timestamp - curr_atom.timestamp
                    
                    # Weight decays with time: w = e^(-Δt)
                    weight = np.exp(-delta_t / 3600.0)  # Decay over hours
                    
                    edges.append(GraphEdge(
                        source_id=curr_id,
                        target_id=next_id,
                        edge_type=EdgeType.NEXT,
                        weight=weight
                    ))
        
        return edges
    
    def _create_semantic_edges(
        self,
        atoms: List[KnowledgeAtom],
        atom_ids: List[str]
    ) -> List[GraphEdge]:
        """
        Create semantic similarity edges
        
        Uses k-NN graph with cosine similarity
        """
        edges = []
        
        # Group by modality
        modality_groups = defaultdict(list)
        for atom, atom_id in zip(atoms, atom_ids):
            modality_groups[atom.modality].append((atom, atom_id))
        
        # For each modality, create k-NN graph
        for modality, atoms_list in modality_groups.items():
            if len(atoms_list) < 2:
                continue
            
            # Extract embeddings
            embeddings = []
            valid_atoms = []
            valid_ids = []
            
            for atom, atom_id in atoms_list:
                if 'default' in atom.embeddings:
                    embeddings.append(atom.embeddings['default'])
                    valid_atoms.append(atom)
                    valid_ids.append(atom_id)
            
            if len(embeddings) < 2:
                continue
            
            embeddings = np.array(embeddings)
            
            # Compute pairwise similarities
            # Normalize embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized = embeddings / (norms + 1e-8)
            
            # Compute similarity matrix
            similarity_matrix = np.dot(normalized, normalized.T)
            
            # Create k-NN edges
            threshold = self.similarity_thresholds.get(modality, 0.7)
            
            for i in range(len(valid_ids)):
                # Get k nearest neighbors
                sims = similarity_matrix[i]
                
                # Exclude self
                sims[i] = -1.0
                
                # Get top-k
                top_k_indices = np.argsort(sims)[-self.k_neighbors:]
                
                for j in top_k_indices:
                    sim_score = sims[j]
                    
                    if sim_score >= threshold:
                        edges.append(GraphEdge(
                            source_id=valid_ids[i],
                            target_id=valid_ids[j],
                            edge_type=EdgeType.SIMILAR_TO,
                            weight=float(sim_score)
                        ))
        
        # Make undirected by adding reverse edges
        reverse_edges = []
        for edge in edges:
            if edge.edge_type == EdgeType.SIMILAR_TO:
                reverse_edges.append(GraphEdge(
                    source_id=edge.target_id,
                    target_id=edge.source_id,
                    edge_type=EdgeType.SIMILAR_TO,
                    weight=edge.weight
                ))
        
        edges.extend(reverse_edges)
        
        return edges
    
    def _create_entity_edges(
        self,
        atoms: List[KnowledgeAtom],
        atom_ids: List[str]
    ) -> List[GraphEdge]:
        """
        Create entity-based edges: ENTITY_SHARED
        
        Connects atoms that share named entities
        """
        if not self.nlp:
            return []
        
        edges = []
        
        # Extract entities for text atoms
        text_atoms = [
            (atom, atom_id) 
            for atom, atom_id in zip(atoms, atom_ids)
            if atom.modality == Modality.TEXT
        ]
        
        if not text_atoms:
            return []
        
        # Extract entities
        atom_entities = {}
        for atom, atom_id in text_atoms:
            # Use cached entities if available
            if atom.entities:
                atom_entities[atom_id] = set(atom.entities)
            else:
                doc = self.nlp(atom.content[:1000])  # Limit for performance
                entities = set([ent.text.lower() for ent in doc.ents])
                atom_entities[atom_id] = entities
        
        # Create edges based on shared entities
        atom_ids_list = list(atom_entities.keys())
        
        for i in range(len(atom_ids_list)):
            for j in range(i + 1, len(atom_ids_list)):
                id_i = atom_ids_list[i]
                id_j = atom_ids_list[j]
                
                entities_i = atom_entities[id_i]
                entities_j = atom_entities[id_j]
                
                # Compute Jaccard similarity
                intersection = entities_i & entities_j
                union = entities_i | entities_j
                
                if len(union) > 0 and len(intersection) > 0:
                    jaccard = len(intersection) / len(union)
                    
                    if jaccard > 0.1:  # Threshold
                        edges.append(GraphEdge(
                            source_id=id_i,
                            target_id=id_j,
                            edge_type=EdgeType.ENTITY_SHARED,
                            weight=jaccard
                        ))
                        
                        # Undirected
                        edges.append(GraphEdge(
                            source_id=id_j,
                            target_id=id_i,
                            edge_type=EdgeType.ENTITY_SHARED,
                            weight=jaccard
                        ))
        
        return edges
    
    def _create_cross_modal_edges(
        self,
        atoms: List[KnowledgeAtom],
        atom_ids: List[str]
    ) -> List[GraphEdge]:
        """
        Create cross-modal alignment edges
        
        Types:
        - ALIGNED_WITH: audio ↔ text (via timestamp)
        - VISUAL_GROUNDED_IN: image ↔ text (via CLIP)
        - SCENE_DESCRIBES: video ↔ text
        """
        edges = []
        
        # Group atoms by source and modality
        source_groups = defaultdict(lambda: defaultdict(list))
        for atom, atom_id in zip(atoms, atom_ids):
            source_groups[atom.source][atom.modality].append((atom, atom_id))
        
        # For each source
        for source, modality_dict in source_groups.items():
            
            # Audio ↔ Text alignment (via timestamp)
            if Modality.AUDIO in modality_dict and Modality.TEXT in modality_dict:
                audio_atoms = modality_dict[Modality.AUDIO]
                text_atoms = modality_dict[Modality.TEXT]
                
                edges.extend(self._align_audio_text(audio_atoms, text_atoms))
            
            # Image ↔ Text alignment (via CLIP)
            if Modality.IMAGE in modality_dict and Modality.TEXT in modality_dict:
                image_atoms = modality_dict[Modality.IMAGE]
                text_atoms = modality_dict[Modality.TEXT]
                
                edges.extend(self._align_image_text(image_atoms, text_atoms))
            
            # Video ↔ Text alignment
            if Modality.VIDEO in modality_dict and Modality.TEXT in modality_dict:
                video_atoms = modality_dict[Modality.VIDEO]
                text_atoms = modality_dict[Modality.TEXT]
                
                edges.extend(self._align_video_text(video_atoms, text_atoms))
        
        return edges
    
    def _align_audio_text(
        self,
        audio_atoms: List[Tuple[KnowledgeAtom, str]],
        text_atoms: List[Tuple[KnowledgeAtom, str]]
    ) -> List[GraphEdge]:
        """
        Align audio and text via timestamp overlap
        
        Weight: w = overlap(a_k, t_k) / duration(a_k)
        """
        edges = []
        
        for audio_atom, audio_id in audio_atoms:
            # Audio timestamp range
            audio_start = audio_atom.timestamp
            audio_duration = audio_atom.metadata.get('duration', 0.0)
            audio_end = audio_start + audio_duration
            
            for text_atom, text_id in text_atoms:
                text_start = text_atom.timestamp
                text_duration = text_atom.metadata.get('duration', 0.0)
                text_end = text_start + text_duration
                
                # Compute overlap
                overlap_start = max(audio_start, text_start)
                overlap_end = min(audio_end, text_end)
                overlap = max(0.0, overlap_end - overlap_start)
                
                if overlap > 0 and audio_duration > 0:
                    weight = overlap / audio_duration
                    
                    edges.append(GraphEdge(
                        source_id=audio_id,
                        target_id=text_id,
                        edge_type=EdgeType.ALIGNED_WITH,
                        weight=weight
                    ))
                    
                    # Bidirectional
                    edges.append(GraphEdge(
                        source_id=text_id,
                        target_id=audio_id,
                        edge_type=EdgeType.ALIGNED_WITH,
                        weight=weight
                    ))
        
        return edges
    
    def _align_image_text(
        self,
        image_atoms: List[Tuple[KnowledgeAtom, str]],
        text_atoms: List[Tuple[KnowledgeAtom, str]]
    ) -> List[GraphEdge]:
        """
        Align image and text via CLIP embeddings
        
        Weight: w = cos(e_i^img, e_j^txt) in CLIP space
        """
        edges = []
        
        clip_threshold = 0.25  # CLIP alignment threshold
        
        for image_atom, image_id in image_atoms:
            # Get CLIP image embedding
            if 'clip' not in image_atom.embeddings:
                continue
            
            image_clip_emb = np.array(image_atom.embeddings['clip'])
            
            for text_atom, text_id in text_atoms:
                # Get CLIP text embedding
                if 'clip' not in text_atom.embeddings:
                    continue
                
                text_clip_emb = np.array(text_atom.embeddings['clip'])
                
                # Compute similarity
                similarity = self.vectorizer.compute_similarity(
                    image_clip_emb,
                    text_clip_emb
                )
                
                if similarity >= clip_threshold:
                    edges.append(GraphEdge(
                        source_id=image_id,
                        target_id=text_id,
                        edge_type=EdgeType.VISUAL_GROUNDED_IN,
                        weight=similarity
                    ))
                    
                    # Bidirectional
                    edges.append(GraphEdge(
                        source_id=text_id,
                        target_id=image_id,
                        edge_type=EdgeType.VISUAL_GROUNDED_IN,
                        weight=similarity
                    ))
        
        return edges
    
    def _align_video_text(
        self,
        video_atoms: List[Tuple[KnowledgeAtom, str]],
        text_atoms: List[Tuple[KnowledgeAtom, str]]
    ) -> List[GraphEdge]:
        """
        Align video scenes with text descriptions
        """
        # Similar to image-text alignment
        # Video atoms should have CLIP embeddings from keyframes
        return self._align_image_text(video_atoms, text_atoms)
    
    def _find_closest_parent(
        self,
        child_atom: KnowledgeAtom,
        parent_atoms: List[Tuple[KnowledgeAtom, str]]
    ) -> Optional[str]:
        """
        Find the closest parent atom for hierarchical PART_OF edges
        
        Uses timestamp proximity
        """
        if not parent_atoms:
            return None
        
        # Find parent with closest timestamp
        min_distance = float('inf')
        closest_id = None
        
        for parent_atom, parent_id in parent_atoms:
            distance = abs(parent_atom.timestamp - child_atom.timestamp)
            
            if distance < min_distance:
                min_distance = distance
                closest_id = parent_id
        
        return closest_id
    
    def _normalize_edge_weights(self, edges: List[GraphEdge]) -> List[GraphEdge]:
        """
        Normalize edge weights per edge type
        
        For each edge type t:
        ŵ_ij(t) = (w_ij - μ_t) / σ_t
        
        Then squashed: w̃_ij = σ(ŵ_ij)
        """
        # Group by edge type
        type_groups = defaultdict(list)
        for edge in edges:
            type_groups[edge.edge_type].append(edge)
        
        # Normalize per type
        for edge_type, edges_of_type in type_groups.items():
            weights = [e.weight for e in edges_of_type]
            
            if not weights:
                continue
            
            mean_weight = np.mean(weights)
            std_weight = np.std(weights) + 1e-8
            
            for edge in edges_of_type:
                # Z-score normalization
                normalized = (edge.weight - mean_weight) / std_weight
                
                # Sigmoid squashing to [0, 1]
                squashed = 1.0 / (1.0 + np.exp(-normalized))
                
                edge.weight = float(squashed)
        
        return edges
    
    def dynamic_edge_update(
        self,
        source_id: str,
        target_id: str,
        usage_feedback: float,
        rho: float = 0.05
    ) -> float:
        """
        Dynamic edge weight update based on usage
        
        w_ij(t+1) = (1-ρ)·w_ij(t) + ρ·usage_ij
        
        Returns new weight
        """
        # This would be called from the retrieval engine
        # when an edge participates in successful retrieval
        
        # Placeholder - actual implementation needs current weight
        new_weight = 0.0  # Would fetch current weight from storage
        new_weight = (1 - rho) * new_weight + rho * usage_feedback
        
        return new_weight