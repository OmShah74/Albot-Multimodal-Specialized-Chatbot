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
from backend.core.web_search.web_search_engine import WebSearchEngine


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
        try:
            self.storage.connect()
            logger.info("Storage manager connected.")
        except Exception as e:
            logger.critical(f"Failed to connect storage manager: {e}")
            # Raise or handle gracefully? For now log Critical.
        
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
        
        # Web Search
        self.web_search = WebSearchEngine()
        
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
    
    async def query(
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
        
        # Save User Message to History
        self.storage.save_chat_message(role="user", content=query_text)

        logger.info(f"Processing query: {query_text} (mode={config.get('mode', 'advanced')})")
        
        # Step 1: Query decomposition
        query_decomp = self._decompose_query(query_text, query_modalities)
        
        # Step 2: Retrieval with config and timing
        retrieval_start = time.time()
        results, retrieval_metrics = self.retriever.retrieve(query_decomp, query_text, config)
        retrieval_time = (time.time() - retrieval_start) * 1000
        
        # Step 3: Format evidence for LLM and extract unique sources
        evidence, sources = self._format_evidence(results)
        
        # Step 3.5: Context sufficiency check → Web search fallback
        web_search_used = False
        web_search_metrics = {}
        
        if not self._is_context_sufficient(results, evidence, query_text):
            logger.info("Local context insufficient - triggering web search")
            try:
                web_results, web_metrics = await self.web_search.search(query_text, top_k=10)
                web_search_metrics = web_metrics
                
                if web_results:
                    web_search_used = True
                    web_evidence = self.web_search.format_for_llm(web_results)
                    web_sources = [r.url for r in web_results if r.url]
                    
                    # Merge: local evidence + web evidence
                    if evidence and evidence.strip():
                        combined_evidence = (
                            f"=== LOCAL KNOWLEDGE BASE ===\n{evidence}\n\n"
                            f"=== WEB SEARCH RESULTS ===\n{web_evidence}"
                        )
                    else:
                        combined_evidence = web_evidence
                    
                    evidence = combined_evidence
                    sources = list(set(sources + web_sources))
                    
                    logger.info(f"Web search added {len(web_results)} results")
                else:
                    logger.warning("Web search returned 0 results.")
                    
                    # GUARDRAIL: If web search failed and local context is invalid, abort.
                    if not evidence or len(evidence.strip()) < 200:
                        logger.warning("Aborting synthesis due to total retrieval failure.")
                        
                        failure_metrics = {
                            "total_time_ms": round((time.time() - start_time) * 1000, 1),
                            "mode": config.get("mode", "advanced"),
                            "web_search_used": True,
                            "web_search_time_ms": round(web_metrics.get("web_search_time_ms", 0), 1),
                            "status": "failed_retrieval"
                        }
                        
                        return {
                            "answer": "I apologize, but I couldn't retrieve reliable information from the web at this time due to a search provider failure. Please try again later.",
                            "sources": [],
                            "metrics": failure_metrics
                        }

            except Exception as e:
                logger.error(f"Web search failed: {e}")
                # Fallback guardrail for crashes
                if not evidence or len(evidence.strip()) < 200:
                     return {
                        "answer": "I encountered an error while searching the web and have no local information on this topic.",
                        "sources": [],
                        "metrics": {"total_time_ms": round((time.time() - start_time) * 1000, 1), "error": str(e)}
                    }
        
        # Step 4: LLM synthesis with timing
        synthesis_start = time.time()
        
        # GUARDRAIL: Final check before synthesis
        if not evidence or len(evidence.strip()) < 100:
             return {
                "answer": "I have no information available to answer this query.",
                "sources": [],
                "metrics": {"total_time_ms": round((time.time() - start_time) * 1000, 1), "status": "no_context"}
            }

        if web_search_used:
            answer = self._synthesize_with_web_context(query_text, evidence)
        else:
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
            "algorithms_used": retrieval_metrics.get("algorithms_used", []),
            "web_search_used": web_search_used,
            "web_search_time_ms": round(web_search_metrics.get("web_search_time_ms", 0), 1),
            "web_providers_used": web_search_metrics.get("provider_breakdown", {})
        }
        
        logger.info(f"Query completed in {total_time:.0f}ms (retrieval: {retrieval_time:.0f}ms, synthesis: {synthesis_time:.0f}ms, web_search: {web_search_used})")
        
        # Save Assistant Message to History
        self.storage.save_chat_message(
            role="assistant", 
            content=answer, 
            sources=sources,
            metrics=metrics
        )
        
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
        
        system_prompt = """You are Albot, an advanced AI research assistant designed to provide clear, comprehensive answers with natural flow.

RESPONSE PHILOSOPHY:
Your goal is to educate and inform, not to showcase sources. Information should flow naturally like an expert explaining a topic to a colleague. Sources exist in the context, but your job is to synthesize knowledge into a coherent narrative.

FORMATTING RULES:
1. **Natural Paragraphs**: Write in substantial, well-developed paragraphs (4-8 sentences). Do NOT break into new paragraphs every 2-3 sentences. Let ideas breathe and develop fully.

2. **NO Citation Markers**: NEVER use citation formats like [1], [2], (Source: X), "According to...", "As mentioned by...", etc. The user can see sources separately. Your job is to present the information as unified knowledge.

3. **Markdown Structure**:
   - Use **bold** for key terms and important concepts (sparingly - 2-3 times per response maximum)
   - Use headers (##) only for major section breaks in complex multi-faceted topics
   - Use bullet points or numbered lists ONLY when listing distinct items/steps that genuinely benefit from enumeration
   - Use code blocks (```) only for actual code, commands, or technical syntax
   - Avoid excessive formatting - let the content speak for itself

4. **Paragraph Flow**: Each paragraph should:
   - Start with a clear topic sentence
   - Develop the idea with supporting details and explanations
   - Connect smoothly to the next paragraph using transitional phrases
   - Be substantial enough to convey complete thoughts (4-7 sentences minimum)
   - Build on previous paragraphs to create a logical progression

5. **NO Redundant Sections**: Do NOT add "References", "Sources", "Bibliography", "Further Reading", or "Conclusion" sections at the end. These are handled automatically by the system.

CONTENT GUIDELINES:
- **Synthesis Over Summary**: Don't just list facts from different sources. Weave them into a coherent explanation that builds understanding progressively. Connect related concepts and show how they fit together.

- **Technical Precision**: Use exact terminology, numbers, and technical details from the context when relevant. Be specific rather than vague.

- **Contextual Depth**: Explain WHY things matter, HOW they work, WHEN they're applicable, and WHAT makes them different - not just WHAT they are. Provide insight, not just information.

- **Conversational Expertise**: Write like a knowledgeable expert having a conversation, not like a formal academic paper or Wikipedia article. Be engaging and clear.

- **Code Inclusion**: Only provide code examples if the query explicitly requests implementation details, asks "how to" do something programmatically, or clearly needs a code sample to be fully answered.

- **Avoid Formulaic Structure**: Don't follow rigid templates like "Introduction → Body → Conclusion". Let the content determine the structure organically.

GROUNDING RULES:
- Only use information present in the provided context
- If information is insufficient, acknowledge gaps clearly: "Based on the available information..." or "The provided sources don't specify..."
- Never invent facts or extrapolate beyond what's given
- If the context is contradictory, present both perspectives clearly without taking sides

EXAMPLE OF EXCELLENT FLOW:
"Graph RAG and Vector RAG represent two fundamentally different approaches to retrieval-augmented generation, each optimized for distinct use cases and data structures. Understanding their differences is crucial for selecting the right architecture for your specific application needs.

Vector RAG operates by converting documents into dense numerical embeddings that capture semantic meaning in high-dimensional space. This approach excels at finding conceptually similar content across large, unstructured text corpora through similarity search. Vector databases like Milvus and Pinecone power these systems, enabling fast nearest-neighbor searches across millions of embeddings. The strength of Vector RAG lies in its ability to retrieve relevant information even when exact keywords don't match, making it ideal for general-purpose question answering and broad knowledge retrieval tasks.

Graph RAG takes a more structured approach by representing knowledge as an interconnected network of entities and relationships. Instead of relying solely on semantic similarity, it leverages explicit connections between concepts stored in graph databases. This enables sophisticated multi-hop reasoning where the system can traverse relationships to answer complex queries that require understanding how different pieces of information connect. For instance, answering "What companies did the former CEO of Microsoft invest in after retiring?" requires following multiple relationship chains that Vector RAG would struggle with.

The practical implications of choosing between these approaches are significant. Vector RAG offers simpler implementation and lower maintenance overhead, making it the default choice for most retrieval tasks. Graph RAG requires substantial upfront investment in knowledge graph construction and maintenance but delivers superior performance for domain-specific applications where entity relationships are central. Many production systems now adopt hybrid architectures that combine both approaches, using Graph RAG for structured queries over known entities while falling back to Vector RAG for broader semantic search."

EXAMPLE OF POOR FLOW (AVOID THIS):
"According to Designveloper, Graph RAG and Vector RAG are different approaches.

Vector RAG uses embeddings. As Instaclustr explains, this method works well.

Graph RAG uses knowledge graphs. Ragaboutit notes that it handles relationships.

The choice depends on requirements. Meilisearch provides comparison details.

In conclusion, both methods have uses.

References:
1. Designveloper
2. Instaclustr 
3. Ragaboutit"

CRITICAL REMINDERS:
- Write in flowing, connected paragraphs - not choppy 2-3 sentence blocks
- NO citation markers anywhere in the response
- NO reference/source sections at the end
- Let ideas develop fully before moving to the next concept
- Sound like an expert teacher, not a robot aggregating sources"""
        
        user_prompt = f"""Based on the following context, answer the user's question with natural, flowing prose.

Context:
{evidence}

Question: {query}

Provide a comprehensive response that synthesizes the information naturally:"""
        
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

    def _is_context_sufficient(self, results, evidence: str, query: str = "") -> bool:
        """
        Check if local retrieval results are sufficient to answer the query.
        """
        # Criteria 0: Forced Web Search for specific real-time intents
        force_keywords = [
            "score", "vs", "match", "playing", "live", "news", 
            "latest", "currently", "price", "stock", "weather",
            "anthropic", "openai", "google", "meta", "nvidia", "cricket"
        ]
        if query and any(kw in query.lower() for kw in force_keywords):
            logger.info(f"Force web search triggered by keyword in: {query}")
            return False

        # Criteria 1: No results
        if not results:
            return False
            
        # Criteria 2: Very short evidence (Critical check)
        if len(evidence.strip()) < 100:
             logger.info(f"Context insufficient: Evidence length {len(evidence.strip())} < 100")
             return False
            
        # Criteria 3: Entity Mismatch Check
        # If query has proper nouns (Entities) that are NOT in evidence, fail.
        if query:
            import re
            query_words = set(re.findall(r'\b[A-Z][a-z]{2,}\b', query))
            if query_words:
                evidence_lower = evidence.lower()
                missing_entities = [w for w in query_words if w.lower() not in evidence_lower]
                if len(missing_entities) > 0:
                      logger.info(f"Context insufficient: Missing key entities {missing_entities}")
                      return False

        # Criteria 4: Low relevance score
        max_score = max((r.score for r in results), default=0.0)
        if max_score < 0.70:
            logger.info(f"Context insufficient: Max score {max_score:.2f} < 0.70")
            return False
            
        return True

    def _synthesize_with_web_context(self, query: str, evidence: str) -> str:
        """Synthesize answer using web search context."""
        
        system_prompt = """You are Albot, an advanced AI assistant with real-time web access and research capabilities.

RESPONSE PHILOSOPHY:
Your goal is to provide comprehensive, accurate answers by synthesizing information from multiple web sources into a coherent narrative. Present knowledge naturally without excessive attribution that disrupts reading flow.

FORMATTING RULES:
1. **Natural Paragraphs**: Write in substantial, well-developed paragraphs (5-8 sentences). Do NOT break into new paragraphs every 2-3 sentences. Let ideas develop fully within each paragraph before transitioning.

2. **Minimal Attribution**: While you can occasionally attribute major claims to sources, do NOT constantly interrupt flow with citations. Instead:
   - GOOD: "Recent developments in the field show that..." (no attribution)
   - GOOD: "Industry analysis reveals that..." (no attribution)
   - ACCEPTABLE (use sparingly): "Research from leading institutions indicates that..." (vague attribution)
   - BAD: "According to TechCrunch..." (specific attribution - avoid unless critical)
   - NEVER: [1], [2], (Source: X) notation

3. **Markdown Structure**:
   - Use **bold** sparingly for key terms (2-4 times maximum)
   - Use headers (##) only for major section breaks in complex topics
   - Use bullet points or numbered lists ONLY when listing distinct items
   - Avoid excessive formatting

4. **Paragraph Development**: Each paragraph should:
   - Start with a clear topic sentence
   - Develop 3-5 supporting points
   - Use transitions between paragraphs ("Building on this...", "This approach...", "However...")
   - Maintain conceptual coherence within the paragraph

5. **NO Redundant Sections**: Do NOT add "References", "Sources", "Conclusion", or "Summary" sections. These are handled by the system.

CONTENT GUIDELINES:
- **Authority Awareness**: Give more weight to official documentation, academic sources, and reputable news outlets over forums or blogs. Synthesize authoritative sources first.

- **Conflict Resolution**: If sources disagree, present both perspectives naturally: "While some implementations favor X, others demonstrate Y provides better results in specific scenarios." Don't flag conflicts explicitly unless critical to the answer.

- **Technical Precision**: Include specific numbers, dates, and technical details from sources. Be precise.

- **Synthesis Over Listing**: Don't present information source-by-source. Instead, identify themes across sources and present unified insights.

- **Code Examples**: Only include code if the query explicitly requests implementation or asks "how to" do something programmatically.

- **Recency**: For time-sensitive queries, naturally incorporate temporal context without over-emphasizing it: "The latest data shows..." rather than "As of [specific date]..."

GROUNDING RULES:
- Only use information from the provided web search results
- If information is contradictory, present multiple viewpoints without declaring one correct
- If sources lack depth on a specific aspect, acknowledge this briefly: "While comprehensive data on X isn't readily available..."
- Never invent information not present in the sources

EXAMPLE OF EXCELLENT WEB SYNTHESIS:
"Graph RAG and Vector RAG represent distinct paradigms in retrieval-augmented generation, each offering unique advantages for different use cases. The fundamental difference lies in how they structure and query information, which has significant implications for system design and performance.

Vector RAG systems encode documents as dense embeddings in high-dimensional vector space, enabling semantic similarity search across large corpora. This approach excels at retrieving conceptually related content even when exact terminology differs. Modern implementations leverage specialized vector databases optimized for billion-scale approximate nearest neighbor search, achieving sub-second query latency. The simplicity of this architecture makes it the default choice for general-purpose question answering, content recommendation, and broad knowledge retrieval tasks.

Graph RAG extends beyond pure semantic similarity by explicitly modeling relationships between entities through knowledge graphs. This structural representation enables multi-hop reasoning and complex queries that traverse entity connections. The approach proves particularly valuable in domains with rich relational data, such as scientific literature, financial networks, or organizational hierarchies. By maintaining explicit edges between concepts, Graph RAG can answer queries like "Which companies founded by former Google engineers received Series A funding in 2023?" that require chaining multiple relationship types.

Performance characteristics differ substantially between these approaches. Vector RAG offers superior scaling for unstructured text and simpler maintenance, while Graph RAG provides more precise answers for relationship-oriented queries but requires significant investment in graph construction and curation. Recent hybrid architectures attempt to capture both benefits by using Graph RAG for entity-centric queries while falling back to Vector RAG for broader semantic search, though this adds architectural complexity. The choice ultimately depends on whether your use case primarily involves semantic content retrieval or requires explicit reasoning over entity relationships."

EXAMPLE OF POOR WEB SYNTHESIS (AVOID THIS):
"According to TechCrunch, Graph RAG is different from Vector RAG.

Designveloper explains that Vector RAG uses embeddings. The article from Instaclustr notes this is effective.

Wikipedia states that Graph databases store relationships. Ragaboutit mentions this enables complex queries.

Meilisearch provides details on when to use each approach. According to their blog post, it depends on requirements.

Sources indicate hybrid approaches exist. Multiple articles discuss this option.

References:
- TechCrunch
- Designveloper
- Instaclustr
- Wikipedia
- Ragaboutit
- Meilisearch"

CRITICAL REMINDERS:
- Write in flowing 5-8 sentence paragraphs, not choppy fragments
- Synthesize across sources - don't present source-by-source
- Minimize explicit attribution - let information flow naturally
- NO citation markers like [1], [2], or (Source: X)
- NO reference/source sections at the end
- Build conceptual understanding progressively across paragraphs"""

        user_prompt = f"""Based on the real-time web search results below, answer the user's question with natural, flowing synthesis.

Context:
{evidence}

Question: {query}

Provide a comprehensive response that weaves information from multiple sources into a coherent explanation:"""

        messages = [
            {"role": "user", "content": user_prompt}
        ]

        try:
            response = self.llm_router.complete(
                messages=messages,
                system_prompt=system_prompt,
                max_tokens=1500,
                temperature=0.6  # Slightly higher for creative synthesis of web results
            )
            return response
        except Exception as e:
            logger.error(f"Web synthesis failed: {e}")
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

    def get_chat_history(self, limit: int = 100) -> List[Dict]:
        """Get chat history"""
        return self.storage.get_chat_history(limit)
    
    def clear_chat_history(self):
        """Clear chat history"""
        self.storage.clear_chat_history()
    
    def shutdown(self):
        """Cleanup resources"""
        self.storage.close()
        logger.info("RAG system shutdown complete")