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
from backend.core.storage.sqlite_manager import SQLiteStorageManager # Added SQLite
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
        
        # Chat History Storage (SQLite)
        self.chat_storage = SQLiteStorageManager()
        
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
        chat_id: str,
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
        
        # Save User Message to History (SQLite)
        self.chat_storage.save_chat_message(chat_id=chat_id, role="user", content=query_text)

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
        
        # Save Assistant Message to History (SQLite)
        self.chat_storage.save_chat_message(
            chat_id=chat_id,
            role="assistant", 
            content=answer, 
            sources=sources,
            metrics=metrics
        )
        
        # Auto-title if it's a new chat (heuristic: query length < 5 messages or title is "New Chat")
        # For now, just fire and forget on every turn (or check if title is "New Chat" - needs DB check)
        # Efficient way: Assume we want to re-title if it's the first turn.
        # But we don't have message count here easily without querying DB.
        # Let's just run it. The LLM call is cheap-ish for titles.
        # Better: Only if current title is default.
        # We can't easily check current title without querying. 
        # Let's just call it. The _generate_chat_title method can do a check or just overwrite.
        new_title = self._generate_chat_title(chat_id, query_text, answer)
        
        return {
            "answer": answer,
            "sources": sources,
            "metrics": metrics,
            "chat_title": new_title
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
1. **Semantic Hierarchy**: Use **Markdown headers (## and ###)** to organize information into logical, scannable sections.
2. **Structural Spacing**: ENSURE at least two newlines between different Markdown blocks (paragraphs, headers, lists) to prevent congestion.
3. **Strategic Emphasis**: Use **bold text** for primary terms, important concepts, and key definitions. Use it moderately but effectively for scanning.
4. **Lists & Bullets**: Use bulleted or numbered lists for all enumerations, features, steps, or multi-point explanations.
5. **Rich Synthesis**: Write like an expert. Within each section, maintain a professional and insightful tone while utilizing structural formatting.
6. **No Meta-Talk**: Do NOT add "Sources", "References", or "According to..." markers. Focus purely on the content structure.
7. **Spacing Excellence**: Maintain generous vertical spacing between sections to ensure a premium, modern chat experience.
8.  **Markdown Structure**:
   - Use **bold** for key terms and important concepts (sparingly - 2-3 times per response maximum)
   - Use headers (##) only for major section breaks in complex multi-faceted topics
   - Use bullet points or numbered lists ONLY when listing distinct items/steps that genuinely benefit from enumeration
   - Use code blocks (```) only for actual code, commands, or technical syntax
   - Avoid excessive formatting - let the content speak for itself

CONTENT GUIDELINES:
- **Synthesis Over Summary**: Don't just list facts from different sources. Weave them into a coherent explanation that builds understanding progressively. Connect related concepts and show how they fit together.
- **Contextual Depth**: Explain WHY things matter, HOW they work, and WHAT makes them different.
- **Conversational Expertise**: Write like a knowledgeable expert having a conversation. Be engaging and clear.
- **Code Inclusion**: Only provide code examples if explicitly requested or essential.
- **Technical Precision**: Use exact terminology, numbers, and technical details from the context when relevant. Be specific rather than vague.

GROUNDING RULES:
- Only use information present in the provided context
- If information is insufficient, acknowledge gaps clearly: "Based on the available information..." or "The provided sources don't specify..."
- Never invent facts or extrapolate beyond what's given
- If the context is contradictory, present both perspectives clearly without taking sides.

EXAMPLE OF EXCELLENT FLOW:
"Graph RAG and Vector RAG represent two fundamentally different approaches to retrieval-augmented generation, each optimized for distinct use cases and data structures. Understanding their differences is crucial for selecting the right architecture for your specific application needs.

Vector RAG operates by converting documents into dense numerical embeddings that capture semantic meaning in high-dimensional space. This approach excels at finding conceptually similar content across large, unstructured text corpora through similarity search. Vector databases like Milvus and Pinecone power these systems, enabling fast nearest-neighbor searches across millions of embeddings. The strength of Vector RAG lies in its ability to retrieve relevant information even when exact keywords don't match, making it ideal for general-purpose question answering and broad knowledge retrieval tasks.

Graph RAG takes a more structured approach by representing knowledge as an interconnected network of entities and relationships. Instead of relying solely on semantic similarity, it leverages explicit connections between concepts stored in graph databases. This enables sophisticated multi-hop reasoning where the system can traverse relationships to answer complex queries that require understanding how different pieces of information connect. For instance, answering 'What companies did the former CEO of Microsoft invest in after retiring?' requires following multiple relationship chains that Vector RAG would struggle with.

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
FORMATTING RULES:
1. **Rich Markdown**: Use a professional Markdown structure (headers, bolding, lists) to make the response highly readable.
2. **Structural Bolding**: Use **bold** for key terms, definitions, and important concepts to help users scan the information.
3. **Lists & Bullets**: Use bullet points or numbered lists for features, steps, or distinct categories of information.
4. **Logical Headers**: Use `##` and `###` headers to organize the response into clear, distinct sections (e.g., Overview, How it Works, Applications).
5. **Flowing Tone**: Maintain the tone of an expert teacher. While using Markdown structure, ensure the prose within sections remains insightful and professional.
6. **No Citations**: Do NOT use citation markers (e.g., [1], [Source: X]) in the text.
7. **No Ending Sections**: Do NOT add a "References" or "Sources" section at the end of the text response. The system handles sources separately.
The practical implications of choosing between these approaches are significant. Vector RAG offers simpler implementation and lower maintenance overhead, making it the default choice for most retrieval tasks. Graph RAG requires substantial upfront investment in knowledge graph construction and maintenance but delivers superior performance for domain-specific applications where entity relationships are central."
"""
        
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
1. **Professional Structure**: Use **Markdown headers (## and ###)** to organize the answer into logical sections. Avoid long, unbroken walls of text.
2. **Strategic Bolding**: Use **bold text** to highlight key terms, critical facts, and core concepts.
3. **Lists for Clusters**: When presenting multiple features, advantages, or categories, use bulleted or numbered lists.
4. **Rich Synthesis**: Synthesize information across sources to provide a unified, deep explanation rather than a list of "according to X..." summaries.
5. **No Citations**: Do NOT use citation markers (e.g., [1], [2], (Source: X)) in the response text.
6. **No Ending Sections**: Do NOT add "Sources", "References", or "Further Reading" at the end of the response. The system displays them separately."""

        user_prompt = f"""Synthesize a comprehensive answer for the following query based on the provided web evidence.
        
Query: {query}

Evidence:
{evidence}

Answer in a well-structured, professional Markdown format:"""

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
    
    def _generate_chat_title(self, chat_id: str, query: str, answer: str):
        """Generate a short title for the chat using LLM"""
        try:
            logger.info(f"Generating title for chat {chat_id}...")
            system_prompt = "You are a helpful assistant. Generate a 3 to 4 word title for a conversation based on the user prompt and your response. Do not use quotes. Just the word."
            
            messages = [
                {
                    "role": "user", 
                    "content": f"User: {query}\nAssistant: {answer}\nTitle:"
                }
            ]
            
            # Use a fast model if possible, or default
            response = self.llm_router.complete(
                messages=messages,
                system_prompt=system_prompt,
                max_tokens=20,
                temperature=0.7
            )
            
            clean_title = response.strip().strip('"')
            self.chat_storage.rename_chat(chat_id, clean_title)
            logger.info(f"Auto-titled chat {chat_id} -> '{clean_title}'")
            return clean_title
            
        except Exception as e:
            logger.error(f"Failed to auto-title chat: {e}")
            return None
    
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

    # --- Chat Management ---
    
    def create_chat(self, title: str = "New Chat") -> Dict:
        """Create a new chat session"""
        return self.chat_storage.create_chat(title)

    def get_chats(self) -> List[Dict]:
        """Get all chat sessions"""
        return self.chat_storage.get_chats()
        
    def get_chat(self, chat_id: str) -> Optional[Dict]:
        """Get specific chat session"""
        return self.chat_storage.get_chat(chat_id)

    def rename_chat(self, chat_id: str, new_title: str):
        """Rename a chat session"""
        self.chat_storage.rename_chat(chat_id, new_title)

    def delete_chat(self, chat_id: str):
        """Delete a chat session"""
        self.chat_storage.delete_chat(chat_id)

    def get_chat_history(self, chat_id: str, limit: int = 100) -> List[Dict]:
        """Get chat history for a specific session"""
        return self.chat_storage.get_chat_history(chat_id, limit)
    
    def clear_chat_history(self, chat_id: str):
        """Clear chat history for a specific session"""
        self.chat_storage.clear_chat_history(chat_id)
    
    def shutdown(self):
        """Cleanup resources"""
        self.storage.close()
        logger.info("RAG system shutdown complete")