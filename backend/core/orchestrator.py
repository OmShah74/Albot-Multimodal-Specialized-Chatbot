"""
Main RAG System Orchestrator
Coordinates all components: ingestion, vectorization, storage, graph, retrieval, LLM
"""
import os
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger

from backend.models.config import (
    DatabaseConfig, LLMConfig, RetrievalWeights,
    SearchConfig, KnowledgeAtom, QueryDecomposition,
    Modality
)
from backend.core.storage.arango_manager import ArangoStorageManager
from backend.core.vectorization.embedding_engine import VectorizationEngine
from backend.core.graph.graph_builder import GraphConstructionEngine
from backend.core.retrieval.retrieval_engine import AdvancedRetrievalEngine
from backend.core.llm.llm_router import LLMRouter
from backend.core.ingestion.multimodal_processor import MultimodalProcessor


class RAGOrchestrator:
    """
    Central orchestrator for the entire RAG pipeline
    """
    
    def __init__(
        self,
        db_config: Optional[DatabaseConfig] = None,
        llm_config: Optional[LLMConfig] = None
    ):
        # Load configurations
        self.db_config = db_config or DatabaseConfig(
            host=os.getenv("ARANGO_HOST", "localhost"),
            port=int(os.getenv("ARANGO_PORT", "8529")),
            username=os.getenv("ARANGO_USERNAME", "root"),
            password=os.getenv("ARANGO_PASSWORD", "rootpassword"),
            database=os.getenv("ARANGO_DATABASE", "multimodal_rag")
        )
        
        self.llm_config = llm_config or LLMConfig()
        
        # Initialize components
        logger.info("Initializing RAG system components...")
        
        # Storage
        self.storage = ArangoStorageManager(self.db_config)
        self.storage.connect()
        
        # Vectorization
        self.vectorizer = VectorizationEngine()
        
        # Graph construction
        self.graph_builder = GraphConstructionEngine(self.vectorizer)
        
        # Retrieval
        self.retrieval_weights = RetrievalWeights()
        self.search_config = SearchConfig()
        self.retriever = AdvancedRetrievalEngine(
            storage=self.storage,
            vectorizer=self.vectorizer,
            config=self.search_config,
            weights=self.retrieval_weights
        )
        
        # LLM
        self.llm_router = LLMRouter(self.llm_config)
        
        # Ingestion
        self.processor = MultimodalProcessor()
        
        logger.info("RAG system initialized successfully")
    
    def ingest_file(self, file_path: str) -> Dict:
        """
        Ingest a file into the knowledge base
        
        Pipeline:
        1. Multimodal processing -> knowledge atoms
        2. Vectorization -> embeddings
        3. Storage -> database
        4. Graph construction -> edges
        """
        logger.info(f"Ingesting file: {file_path}")
        
        # Step 1: Process file into knowledge atoms
        atoms = self.processor.process_file(file_path)
        
        if not atoms:
            return {'status': 'error', 'message': 'No content extracted'}
        
        # Step 2: Vectorize atoms
        for atom in atoms:
            # Generate embedding based on modality
            if atom.modality == Modality.TEXT:
                embedding = self.vectorizer.embed_text([atom.content])[0]
                atom.embeddings['default'] = embedding.tolist()
            
            elif atom.modality == Modality.IMAGE:
                # Would need image data from metadata
                pass
            
            elif atom.modality == Modality.AUDIO:
                # Audio uses transcript embedding
                embedding = self.vectorizer.embed_text([atom.content])[0]
                atom.embeddings['default'] = embedding.tolist()
        
        # Step 3: Store atoms
        atom_ids = self.storage.batch_insert_atoms(atoms)
        
        # Update atoms with IDs
        for atom, atom_id in zip(atoms, atom_ids):
            atom.node_id = atom_id
        
        # Step 4: Construct graph edges
        edges = self.graph_builder.construct_edges_for_atoms(atoms, atom_ids)
        
        # Step 5: Store edges
        if edges:
            self.storage.batch_insert_edges(edges)
        
        logger.info(f"Ingested {len(atoms)} atoms, {len(edges)} edges")
        
        return {
            'status': 'success',
            'atoms_count': len(atoms),
            'edges_count': len(edges)
        }
    
    def query(
        self,
        query_text: str,
        query_modalities: Optional[List[Modality]] = None
    ) -> str:
        """
        Main query pipeline
        
        Pipeline:
        1. Query decomposition (LLM)
        2. Multi-channel retrieval
        3. Evidence accumulation
        4. Re-ranking
        5. Synthesis (LLM)
        """
        logger.info(f"Processing query: {query_text}")
        
        # Step 1: Query decomposition
        query_decomp = self._decompose_query(query_text, query_modalities)
        
        # Step 2: Retrieval
        results = self.retriever.retrieve(query_decomp, query_text)
        
        # Step 3: Format evidence for LLM
        evidence = self._format_evidence(results)
        
        # Step 4: LLM synthesis
        response = self._synthesize_answer(query_text, evidence)
        
        return response
    
    def _decompose_query(
        self,
        query: str,
        modalities: Optional[List[Modality]]
    ) -> QueryDecomposition:
        """
        Decompose query into sub-queries and detect intent
        Uses LLM for complex decomposition
        """
        # Simplified - full version would use LLM
        
        mods = modalities or [Modality.TEXT]
        
        return QueryDecomposition(
            sub_queries=[query],
            modalities=mods,
            intent_weights=[1.0],
            depth=2,
            freshness_required=False,
            reasoning_required=True
        )
    
    def _format_evidence(self, results) -> str:
        """Format retrieval results for LLM context"""
        evidence_parts = []
        
        for i, result in enumerate(results[:10], 1):
            evidence_parts.append(
                f"[{i}] {result.content}\n"
                f"   (Score: {result.score:.3f}, Modality: {result.modality.value})"
            )
            
            # Add context if available
            if result.context:
                for ctx in result.context[:2]:
                    evidence_parts.append(f"   Context: {ctx[:200]}...")
        
        return "\n\n".join(evidence_parts)
    
    def _synthesize_answer(self, query: str, evidence: str) -> str:
        """Use LLM to synthesize final answer"""
        
        system_prompt = """You are a helpful AI assistant. Use the provided evidence to answer the user's question accurately and concisely. Cite specific evidence when relevant."""
        
        messages = [
            {
                "role": "user",
                "content": f"Evidence:\n\n{evidence}\n\nQuestion: {query}\n\nProvide a comprehensive answer based on the evidence."
            }
        ]
        
        try:
            response = self.llm_router.complete(
                messages=messages,
                system_prompt=system_prompt,
                max_tokens=1000,
                temperature=0.7
            )
            return response
        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}")
            return f"Error generating response: {str(e)}"
    
    def add_api_key(self, provider: str, name: str, key: str):
        """Add API key to LLM router"""
        from backend.models.config import LLMProvider
        
        provider_enum = LLMProvider(provider.lower())
        self.llm_router.add_api_key(provider_enum, name, key)
    
    def get_statistics(self) -> Dict:
        """Get system statistics"""
        return {
            'database': self.storage.get_statistics(),
            'llm': self.llm_router.get_statistics(),
            'retrieval_weights': {
                'alpha': self.retrieval_weights.alpha,
                'beta': self.retrieval_weights.beta,
                'gamma': self.retrieval_weights.gamma,
                'delta': self.retrieval_weights.delta,
                'epsilon': self.retrieval_weights.epsilon
            }
        }
    
    def shutdown(self):
        """Cleanup resources"""
        self.storage.close()
        logger.info("RAG system shutdown complete")