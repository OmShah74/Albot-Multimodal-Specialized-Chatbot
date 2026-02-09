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
        
        # Load persistent API keys
        self._load_persistent_keys()
        
        logger.info("RAG system initialized successfully")

    def _load_persistent_keys(self):
        """Load API keys from persistent storage"""
        try:
            from backend.models.config import LLMProvider
            # Ensure storage is connected
            if not hasattr(self.storage, 'db'):
                logger.warning("Storage not connected, skipping key load")
                return

            keys = self.storage.list_api_keys()
            for k in keys:
                try:
                    provider = LLMProvider(k['provider'].lower())
                    self.llm_router.add_api_key(
                        provider, 
                        k['name'], 
                        k['key'], 
                        k.get('model_name')
                    )
                except Exception as e:
                    logger.error(f"Failed to load key {k.get('name')}: {e}")
            logger.info(f"Loaded {len(keys)} persistent API keys")
        except Exception as e:
            logger.error(f"Failed to load persistent keys: {e}")
    
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
        
        # Step 2: Vectorize atoms (BATCHED for performance)
        text_atoms = [a for a in atoms if a.modality in [Modality.TEXT, Modality.AUDIO]]
        
        if text_atoms:
            # Batch embed all text content at once
            contents = [a.content[:2000] for a in text_atoms]  # Truncate for performance
            embeddings = self.vectorizer.embed_text(contents)
            
            for atom, embedding in zip(text_atoms, embeddings):
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
    
    def ingest_url(self, url: str) -> Dict:
        """
        Ingest a URL (web or YouTube) into the knowledge base
        """
        logger.info(f"Ingesting URL: {url}")
        
        # Step 1: Process URL into knowledge atoms
        if "youtube.com" in url or "youtu.be" in url:
            atoms = self.processor.process_youtube(url)
        else:
            atoms = self.processor.process_url(url)
        
        if not atoms:
            return {'status': 'error', 'message': 'No content extracted from URL'}
        
        # Step 2: Vectorize atoms
        for atom in atoms:
            if atom.modality in [Modality.TEXT, Modality.AUDIO]:
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
        
        logger.info(f"Ingested {len(atoms)} atoms from URL, {len(edges)} edges")
        
        return {
            'status': 'success',
            'atoms_count': len(atoms),
            'edges_count': len(edges)
        }
    
    def query(
        self,
        query_text: str,
        query_modalities: Optional[List[Modality]] = None,
        retrieval_config: Optional[Dict] = None
    ) -> Dict:
        """
        Main query pipeline with configurable retrieval
        
        Returns dict with answer, sources, and performance metrics
        """
        import time
        start_time = time.time()
        
        if not query_text or not query_text.strip():
            logger.warning("Empty query received")
            return {
                "answer": "Please provide a query.",
                "sources": [],
                "metrics": {"total_time_ms": 0, "mode": "none"}
            }
        
        # Default config: Advanced mode with all algorithms
        config = retrieval_config or {
            "mode": "advanced",
            "use_vector": True,
            "use_graph": True,
            "use_bm25": True,
            "use_pagerank": True,
            "use_structural": True,
            "use_mmr": True
        }
        
        logger.info(f"Processing query: {query_text} (mode={config.get('mode', 'advanced')})")
        
        # Step 1: Query decomposition
        query_decomp = self._decompose_query(query_text, query_modalities)
        
        # Step 2: Retrieval with config and timing
        retrieval_start = time.time()
        results, retrieval_metrics = self.retriever.retrieve(query_decomp, query_text, config)
        retrieval_time = (time.time() - retrieval_start) * 1000
        
        # Step 3: Format evidence for LLM and extract unique sources
        evidence, sources = self._format_evidence(results)
        
        # Step 4: LLM synthesis with timing
        synthesis_start = time.time()
        answer = self._synthesize_answer(query_text, evidence)
        synthesis_time = (time.time() - synthesis_start) * 1000
        
        total_time = (time.time() - start_time) * 1000
        
        # Build metrics
        metrics = {
            "total_time_ms": round(total_time, 1),
            "vector_time_ms": round(retrieval_metrics.get("vector_time_ms", 0), 1),
            "graph_time_ms": round(retrieval_metrics.get("graph_time_ms", 0), 1),
            "bm25_time_ms": round(retrieval_metrics.get("bm25_time_ms", 0), 1),
            "synthesis_time_ms": round(synthesis_time, 1),
            "results_count": len(results),
            "mode": config.get("mode", "advanced"),
            "algorithms_used": retrieval_metrics.get("algorithms_used", [])
        }
        
        logger.info(f"Query completed in {total_time:.0f}ms (retrieval: {retrieval_time:.0f}ms, synthesis: {synthesis_time:.0f}ms)")
        
        return {
            "answer": answer,
            "sources": sources,
            "metrics": metrics
        }

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

    def _format_evidence(self, results) -> tuple:
        """Format retrieval results for LLM context and extract unique sources"""
        evidence_parts = []
        unique_sources = set()
        
        for i, result in enumerate(results[:5], 1):
            # Truncate content to avoid hitting token limits
            content_preview = result.content[:1500] + "..." if len(result.content) > 1500 else result.content
            
            evidence_parts.append(
                f"[{i}] {content_preview}\n"
                f"   (Source: {result.source}, Modality: {result.modality.value})"
            )
            
            if result.source:
                unique_sources.add(result.source)
            
            # Add context if available
            if result.context:
                for ctx in result.context[:1]:
                    evidence_parts.append(f"   Context: {ctx[:150]}...")
        
        return "\n\n".join(evidence_parts), list(unique_sources)

    def _synthesize_answer(self, query: str, evidence: str) -> str:
        """Use LLM to synthesize final answer"""
        
        system_prompt = """You are the Albot Intelligence Specialist, an elite multi-modal reasoning engine.
Your objective is to synthesize complex data into clear, expert-level insights while maintaining a professional and helpful persona.

CRITICAL ARCHITECTURAL CONSTRAINTS:
1. Grounding: You must only use the provided context. If information is missing, state this precisely and suggest what type of data would be needed to answer.
2. Natural Synthesis: Do NOT use robotic citation markers like [1] or [2]. Instead, attribute information naturally if needed (e.g., "The documentation states...", "According to the PDF...").
3. Structure: Use high-quality Markdown. Use bold headers for sections, bullet points for lists, and code blocks for technical parameters.
4. Professionalism: Maintain a sophisticated, expert-level tone. Avoid fluff or filler.
5. Multi-Modality: If the context includes different modalities (Audio, Image, PDF), synthesize them into a unified explanation showing how they relate (e.g., "The visual data confirms what is discussed in the audio segment...").

INTELLECTUAL FRAMEWORK:
- Analyze: Don't just repeat; connect the dots between different pieces of evidence.
- Contextualize: Explain WHY the information matters in the context of the user's query.
- Precision: Use exact numbers, dates, and technical terms found in the context.
- Depth: Provide to the point, deep when needed explanations. Understand the user's query properly.
- Do not give code to the user when not necessary, or when the user does not ask for it."""
        
        user_prompt = f"""Based on the following context, please answer the user's question.

Context:
{evidence}

Question: {query}

Provide a clear, deep and insightful technically grounded response according to the query given by the user and the context:"""
        
        messages = [
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        
        try:
            response = self.llm_router.complete(
                messages=messages,
                system_prompt=system_prompt,
                max_tokens=1300,
                temperature=0.5
            )
            return response
        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}")
            return f"Error generating response: {str(e)}"
    
    def add_api_key(self, provider: str, name: str, key: str, model_name: Optional[str] = None):
        """Add API key to LLM router and save to storage"""
        from backend.models.config import LLMProvider
        
        provider_enum = LLMProvider(provider.lower().strip())
        self.llm_router.add_api_key(provider_enum, name, key, model_name)
        
        # Save to persistent storage
        self.storage.save_api_key(provider.lower().strip(), name, key, model_name)

    def delete_api_key(self, provider: str, name: str):
        """Delete API key from router and storage"""
        from backend.models.config import LLMProvider
        provider_enum = LLMProvider(provider.lower().strip())
        
        # Remove from persistent storage
        self.storage.delete_api_key(provider.lower().strip(), name)
        
        # Remove from memory (LLMRouter)
        self.llm_router.remove_api_key(provider_enum, name)

    def get_api_keys(self) -> Dict:
        """Get all API keys"""
        return self.llm_router.get_api_keys()
    
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

    def list_sources(self) -> List[str]:
        """List all unique sources in the knowledge base"""
        return self.storage.list_sources()

    def delete_document(self, source_name: str):
        """Delete a document from the knowledge base"""
        return self.storage.delete_document(source_name)

    def reset_system(self):
        """Clear the entire knowledge base"""
        self.storage.clear_database()
        logger.info("System reset triggered")

    def shutdown(self):
        """Cleanup resources"""
        self.storage.close()
        logger.info("RAG system shutdown complete")