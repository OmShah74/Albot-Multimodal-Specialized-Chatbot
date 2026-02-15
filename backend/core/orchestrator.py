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
from backend.models.memory import ReasoningTrace, MemoryFragment
from backend.core.storage.arango_manager import ArangoStorageManager
from backend.core.storage.sqlite_manager import SQLiteStorageManager # Added SQLite
from backend.core.vectorization.embedding_engine import VectorizationEngine
from backend.core.graph.graph_builder import GraphConstructionEngine
from backend.core.retrieval.retrieval_engine import AdvancedRetrievalEngine
from backend.core.llm.llm_router import LLMRouter
from backend.core.ingestion.multimodal_processor import MultimodalProcessor
from backend.core.web_search.web_search_engine import WebSearchEngine
from backend.core.memory.memory_manager import MemoryManager
from backend.core.memory.fragment_extractor import FragmentExtractor
from backend.core.memory.namespace_resolver import NamespaceResolver


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
        
        # Memory System
        self.memory = MemoryManager(self.chat_storage, self.storage, self.vectorizer)
        self.fragment_extractor = FragmentExtractor(self.llm_router)
        self.namespace_resolver = NamespaceResolver(self.memory)
        
        # Cancellation tracking
        self._cancelled_chats: set = set()
        
        # Load persistent API keys
        self._load_persistent_keys()
        
        logger.info("RAG system initialized successfully (with memory system)")
    
    def cancel_query(self, chat_id: str):
        """Mark a chat's query as cancelled"""
        self._cancelled_chats.add(chat_id)
        logger.info(f"Query cancelled for chat {chat_id}")
    
    def is_cancelled(self, chat_id: str) -> bool:
        """Check if a chat's query has been cancelled"""
        return chat_id in self._cancelled_chats
    
    def _clear_cancellation(self, chat_id: str):
        """Clear cancellation flag for a chat"""
        self._cancelled_chats.discard(chat_id)

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
        retrieval_config: Optional[Dict] = None,
        search_mode: str = "web_search"
    ) -> Dict:
        """
        Main query pipeline with configurable retrieval
        
        Args:
            search_mode: "web_search" (local + web fallback) or "knowledge_base" (local only)
        
        Returns dict with answer, sources, and performance metrics
        """
        import time
        start_time = time.time()
        
        # Clear any previous cancellation for this chat
        self._clear_cancellation(chat_id)
        
        if not query_text or not query_text.strip():
            logger.warning("Empty query received")
            return {
                "answer": "Please provide a query.",
                "sources": [],
                "metrics": {"total_time_ms": 0, "mode": "none", "search_mode": search_mode}
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

        logger.info(f"Processing query: {query_text} (mode={config.get('mode', 'advanced')}, search_mode={search_mode})")
        
        # Cancellation checkpoint 1: Before retrieval
        if self.is_cancelled(chat_id):
            return self._cancelled_response(chat_id, start_time, config, search_mode)
        
        # Step 1: Query decomposition
        query_decomp = self._decompose_query(query_text, query_modalities)
        
        # Step 1.5: Resolve namespace scope for retrieval
        retrieval_scope = self.namespace_resolver.resolve(chat_id)
        
        # Step 2: Retrieval with config and timing (SAME algorithms for both modes)
        retrieval_start = time.time()
        results, retrieval_metrics = self.retriever.retrieve(
            query_decomp, query_text, config,
            source_filters=retrieval_scope.source_filters if retrieval_scope.source_filters else None
        )
        retrieval_time = (time.time() - retrieval_start) * 1000
        
        # Step 2.5: Augment with memory fragments if enabled
        memory_fragments_used = []
        if retrieval_scope.include_fragments:
            try:
                memory_results = self.memory.search_fragments(
                    query_text, retrieval_scope.active_namespaces, top_k=3
                )
                memory_fragments_used = memory_results
            except Exception as mem_err:
                logger.warning(f"Memory fragment search failed (non-fatal): {mem_err}")
        
        # Cancellation checkpoint 2: After retrieval
        if self.is_cancelled(chat_id):
            return self._cancelled_response(chat_id, start_time, config, search_mode)
        
        # Step 3: Format evidence for LLM and extract unique sources
        evidence, sources = self._format_evidence(results)
        
        # Step 3.5: Context sufficiency check → Web search fallback (ONLY in web_search mode)
        web_search_used = False
        web_search_metrics = {}
        
        if search_mode == "web_search" and not self._is_context_sufficient(results, evidence, query_text):
            logger.info("Local context insufficient - triggering web search (web_search mode)")
            
            # Cancellation checkpoint 3: Before web search
            if self.is_cancelled(chat_id):
                return self._cancelled_response(chat_id, start_time, config, search_mode)
            
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
                            "search_mode": search_mode,
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
                        "metrics": {"total_time_ms": round((time.time() - start_time) * 1000, 1), "search_mode": search_mode, "error": str(e)}
                    }
        elif search_mode == "knowledge_base":
            logger.info("Knowledge Base mode - skipping web search fallback")
        
        # Step 4: LLM synthesis with timing
        synthesis_start = time.time()
        
        # GUARDRAIL: Final check before synthesis
        if not evidence or len(evidence.strip()) < 100:
            if search_mode == "knowledge_base":
                # In KB mode, give a helpful message suggesting web search
                return {
                    "answer": "I don't have enough information in the knowledge base to fully answer this query. Try switching to **Web Search** mode for more comprehensive results.",
                    "sources": [],
                    "metrics": {"total_time_ms": round((time.time() - start_time) * 1000, 1), "search_mode": search_mode, "status": "insufficient_kb"}
                }
            else:
                return {
                    "answer": "I have no information available to answer this query.",
                    "sources": [],
                    "metrics": {"total_time_ms": round((time.time() - start_time) * 1000, 1), "search_mode": search_mode, "status": "no_context"}
                }

        # Cancellation checkpoint 4: Before LLM synthesis
        if self.is_cancelled(chat_id):
            return self._cancelled_response(chat_id, start_time, config, search_mode)

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
            "web_providers_used": web_search_metrics.get("provider_breakdown", {}),
            "search_mode": search_mode
        }
        
        logger.info(f"Query completed in {total_time:.0f}ms (retrieval: {retrieval_time:.0f}ms, synthesis: {synthesis_time:.0f}ms, web_search: {web_search_used}, search_mode: {search_mode})")
        
        # Save Assistant Message to History (SQLite)
        self.chat_storage.save_chat_message(
            chat_id=chat_id,
            role="assistant", 
            content=answer, 
            sources=sources,
            metrics=metrics
        )
        
        new_title = self._generate_chat_title(chat_id, query_text, answer)
        
        # ── Memory System: Post-synthesis processing ──────────
        try:
            import uuid
            turn_index = self.memory.get_turn_count(chat_id)
            
            # Log reasoning trace
            trace = ReasoningTrace(
                trace_id=str(uuid.uuid4()),
                session_id=chat_id,
                turn_index=turn_index,
                user_query=query_text,
                reformulated_query=query_decomp.reformulated_query if hasattr(query_decomp, 'reformulated_query') else None,
                retrieved_doc_ids=[getattr(r, 'atom_id', '') for r in results] if results else [],
                retrieved_doc_sources=list(set(s for s in sources if s)),
                retrieval_scores={getattr(r, 'atom_id', ''): getattr(r, 'score', 0.0) for r in results} if results else {},
                algorithms_used=retrieval_metrics.get("algorithms_used", []),
                web_search_triggered=web_search_used,
                web_urls_searched=[r.url for r in (web_results if web_search_used and 'web_results' in dir() else []) if hasattr(r, 'url')],
                web_snippets=[],
                search_mode=search_mode,
                synthesis_model="",
                answer_summary=answer[:300] if answer else "",
                total_time_ms=round(total_time, 1),
                created_at=__import__('datetime').datetime.utcnow().isoformat()
            )
            self.memory.log_reasoning_trace(trace)
            
            # Log web interactions if any occurred
            if web_search_used and 'web_results' in dir() and web_results:
                self.memory.log_web_interactions(
                    session_id=chat_id,
                    turn_index=turn_index,
                    query=query_text,
                    results=web_results
                )
            
            # Extract and store memory fragments (non-blocking)
            try:
                fragments = self.fragment_extractor.extract(
                    query=query_text,
                    answer=answer,
                    sources=sources,
                    session_id=chat_id
                )
                for frag in fragments:
                    related_ids = [getattr(r, 'atom_id', '') for r in results[:5]] if results else []
                    self.memory.store_fragment(frag, related_doc_ids=related_ids)
            except Exception as frag_err:
                logger.warning(f"Fragment extraction failed (non-fatal): {frag_err}")
                
        except Exception as mem_err:
            logger.warning(f"Memory post-processing failed (non-fatal): {mem_err}")
        
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

    def _cancelled_response(self, chat_id: str, start_time, config: Dict, search_mode: str) -> Dict:
        """Generate a standardized response when a query is cancelled by the user."""
        import time
        self._clear_cancellation(chat_id)
        
        cancelled_msg = "⏹ Generation stopped by user."
        total_time = round((time.time() - start_time) * 1000, 1)
        
        # Save the cancellation message to chat history
        self.chat_storage.save_chat_message(
            chat_id=chat_id,
            role="assistant",
            content=cancelled_msg
        )
        
        return {
            "answer": cancelled_msg,
            "sources": [],
            "metrics": {
                "total_time_ms": total_time,
                "mode": config.get("mode", "advanced"),
                "search_mode": search_mode,
                "status": "cancelled"
            }
        }

    def _detect_query_intent(self, query: str) -> str:
        """
        Detect the intent/type of query to guide response style.
        Returns one of: 'factual', 'explanatory', 'comparative', 'procedural',
                        'analytical', 'conversational', 'creative', 'debugging'
        """
        q = query.lower().strip()

        # Conversational / greeting
        conversational_patterns = [
            "hi", "hello", "hey", "how are you", "what's up", "who are you",
            "what can you do", "thanks", "thank you", "bye", "goodbye"
        ]
        if any(q.startswith(p) or q == p for p in conversational_patterns):
            return "conversational"

        # Debugging / error fixing
        if any(kw in q for kw in ["error", "bug", "fix", "debug", "exception", "traceback", "why is this failing", "not working", "crash"]):
            return "debugging"

        # Procedural / how-to
        if any(kw in q for kw in ["how to", "how do i", "steps to", "guide", "tutorial", "walk me through", "set up", "install", "configure", "implement"]):
            return "procedural"

        # Comparative
        if any(kw in q for kw in ["vs", "versus", "compare", "difference between", "better than", "pros and cons", "which is", "similarities"]):
            return "comparative"

        # Analytical / opinion
        if any(kw in q for kw in ["why", "analyze", "explain why", "reason", "impact", "effect", "opinion", "thoughts on", "evaluate", "assess", "implications"]):
            return "analytical"

        # Creative
        if any(kw in q for kw in ["write", "generate", "create", "draft", "poem", "story", "essay", "summarize", "rewrite", "paraphrase"]):
            return "creative"

        # Explanatory
        if any(kw in q for kw in ["what is", "what are", "explain", "describe", "tell me about", "define", "meaning of"]):
            return "explanatory"

        # Factual / data lookup
        if any(kw in q for kw in ["when", "where", "who", "which", "how many", "how much", "list", "name", "give me"]):
            return "factual"

        return "explanatory"  # Safe default

    def _build_system_prompt(self, query: str, intent: str) -> str:
        """
        Build a dynamic, intent-aware system prompt that produces
        natural, context-appropriate responses — like ChatGPT or Perplexity.
        """

        # ── Core identity and philosophy (always present) ──────────────────────
        base = """You are Albot, a highly capable AI assistant — precise, direct, and genuinely helpful. You adapt your communication style to the nature of each question: concise and sharp for simple queries, structured and thorough for complex ones.

## Core Principles

- **Be direct.** Get to the answer immediately. Don't open with filler phrases like "Certainly!", "Great question!", "Of course!", "Sure!", or "Absolutely!".
- **Match depth to complexity.** A yes/no question gets a short answer. A multi-faceted technical question gets a well-structured, in-depth response.
- **Ground everything in the provided context.** Never fabricate facts, statistics, names, or dates. If the context doesn't contain the answer, say so clearly and concisely.
- **Think, then write.** Your responses should feel like they come from someone who understood the question deeply, not from someone template-filling.
- **No sycophancy.** Never tell the user their question is good, interesting, or great. Just answer it."""

        # ── Intent-specific formatting and tone guidelines ──────────────────────
        intent_guides = {

            "conversational": """
## Response Style: Conversational
- Reply naturally, like a smart colleague in a chat. Short, warm, and direct.
- No headers, no bullet lists, no bold unless it genuinely helps.
- If the user greets you, greet back and offer to help. One or two sentences max.
- Match the user's energy and brevity.""",

            "factual": """
## Response Style: Factual / Direct Answer
- Lead immediately with the answer — the most important fact first.
- Keep it concise. A sentence or short paragraph is usually enough.
- Add brief supporting context only if it meaningfully helps understanding.
- Use a list only if multiple distinct items are being enumerated.
- No unnecessary preamble, no restating the question.""",

            "explanatory": """
## Response Style: Explanation
- Open with a single clear, jargon-free sentence that directly answers the question.
- Then build depth progressively: what it is → how it works → why it matters.
- Use **bold** for key terms when you first introduce them.
- Use short paragraphs (3–5 sentences). Avoid walls of text.
- Use a bullet list or numbered list only if breaking things into distinct points genuinely aids clarity (e.g., listing components, types, or properties).
- No artificial headers like "Introduction" or "Conclusion." Use headers only if the explanation covers multiple distinct major sections.""",

            "procedural": """
## Response Style: Step-by-Step Guide
- Start with one sentence stating what the steps will accomplish.
- Use a **numbered list** for the steps. Each step should be an action.
- Keep step descriptions tight: verb-first, specific, actionable (e.g., "Run `pip install X`", not "You should probably install X").
- Use inline code formatting (backticks) for all commands, file names, and code values.
- Use code blocks for multi-line code or config snippets.
- Add a short "Prerequisites" note at the top only if truly necessary.
- Omit filler steps like "Open your terminal" unless the audience is explicitly a beginner.""",

            "comparative": """
## Response Style: Comparison
- Don't open with a lengthy disclaimer. State the core difference in the first sentence.
- Structure the comparison logically: shared context → key differences → when to use which.
- A comparison table is excellent for side-by-side attribute comparisons — use it when there are 3+ attributes being compared.
- After the table or list, add a brief "When to choose X vs Y" paragraph to give actionable guidance.
- Avoid false balance. If one option is clearly better for a specific use case, say so directly.""",

            "analytical": """
## Response Style: Analysis
- Open with your direct assessment or answer — don't bury the lede.
- Develop your reasoning in clear paragraphs. Show cause-and-effect relationships.
- Use **bold** to highlight key claims or important distinctions.
- If there are multiple perspectives or competing factors, address them honestly without false equivalence.
- Conclude with a synthesized takeaway that follows logically from the analysis.""",

            "debugging": """
## Response Style: Debugging / Problem Solving
- Identify the likely root cause first, in plain language.
- Provide the corrected code or fix immediately after. Use a code block.
- Briefly explain *why* this fixes the issue (one or two sentences is enough).
- If there are multiple possible causes, list them in order of likelihood.
- Don't explain concepts the user clearly already knows. Stay focused on the problem.""",

            "creative": """
## Response Style: Creative / Generative
- Deliver the requested content directly. Don't preface with "Here is the text you asked for."
- Match the tone and style implied by the request (formal, casual, technical, poetic, etc.).
- For summaries: be concise. Capture the essential meaning, not every detail.
- For written content: be specific and vivid. Avoid generic, padded language.
- Only use structure (headers, bullets) if the content type calls for it.""",
        }

        # ── Shared output quality rules (always appended) ──────────────────────
        output_rules = """
## Output Quality Rules

**Formatting:**
- Use `##` headers only for responses covering multiple major sections (4+ distinct topics). Not for short answers.
- Use **bold** for genuinely important terms and key takeaways — sparingly (not every other sentence).
- Use bullet or numbered lists when presenting 3+ discrete items, steps, or options. Not for flowing prose.
- Use code blocks (``` ```) for all code, commands, terminal output, config, and file paths.
- Leave a blank line between paragraphs and between Markdown blocks for visual breathing room.

**Content:**
- Never pad the response with restated context ("Based on the provided context, it appears that...") or meta-commentary ("This is a complex topic...").
- Never end with "I hope this helps!" or similar.
- If the context is insufficient or missing a key fact, state this plainly: "The available information doesn't cover X specifically."
- If the question is ambiguous, answer the most likely interpretation and briefly note the ambiguity.
- Cite nothing inline. The system displays sources separately."""

        intent_section = intent_guides.get(intent, intent_guides["explanatory"])
        return base + "\n\n" + intent_section + "\n\n" + output_rules

    def _build_user_prompt(self, query: str, evidence: str, intent: str) -> str:
        """
        Build a clean, intent-aware user prompt.
        """
        intent_instructions = {
            "conversational": "Respond naturally and briefly.",
            "factual": "Answer directly and concisely. Lead with the fact.",
            "explanatory": "Explain clearly. Build from fundamentals to depth.",
            "procedural": "Provide clear, numbered steps. Use code blocks for commands.",
            "comparative": "Highlight key differences. Be direct about trade-offs.",
            "analytical": "Analyze the topic. Support claims with reasoning from the context.",
            "debugging": "Identify the root cause. Provide a working fix with a brief explanation.",
            "creative": "Generate the requested content directly, matching the implied tone and style.",
        }

        instruction = intent_instructions.get(intent, "Answer the question clearly and accurately.")

        return f"""Context:
{evidence}

User question: {query}

Instruction: {instruction}

Answer:"""

    def _synthesize_answer(self, query: str, evidence: str) -> str:
        """Use LLM to synthesize final answer from local knowledge base context."""

        intent = self._detect_query_intent(query)
        system_prompt = self._build_system_prompt(query, intent)
        user_prompt = self._build_user_prompt(query, evidence, intent)

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
                max_tokens=1500,
                temperature=0.4
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

        intent = self._detect_query_intent(query)

        system_prompt = self._build_system_prompt(query, intent) + """

## Web Search Context — Additional Rules

You have access to real-time web search results combined with local knowledge. Apply these rules on top of your standard guidelines:

- **Recency first.** When sources conflict, prefer the most recent information.
- **Synthesize, don't aggregate.** Don't write "According to site A... and according to site B...". Weave information into a unified answer.
- **No inline citations.** The system renders source links separately. Do not add [1], [Source: X], or similar markers.
- **Flag genuine uncertainty.** If sources conflict on a key fact and you can't resolve it, acknowledge the disagreement briefly.
- **Prioritize authoritative sources.** Official documentation, primary sources, and reputable publications take precedence over aggregator content."""

        user_prompt = self._build_user_prompt(query, evidence, intent)

        messages = [
            {"role": "user", "content": user_prompt}
        ]

        try:
            response = self.llm_router.complete(
                messages=messages,
                system_prompt=system_prompt,
                max_tokens=1500,
                temperature=0.4
            )
            return response
        except Exception as e:
            logger.error(f"Web synthesis failed: {e}")
            return f"Error generating response: {str(e)}"
    
    def _generate_chat_title(self, chat_id: str, query: str, answer: str):
        """Generate a short title for the chat using LLM"""
        try:
            logger.info(f"Generating title for chat {chat_id}...")

            system_prompt = (
                "Generate a concise 3–5 word title that captures the topic of the conversation. "
                "Use title case. No punctuation. No quotes. Return only the title, nothing else."
            )
            
            messages = [
                {
                    "role": "user", 
                    "content": (
                        f"User message: {query[:200]}\n"
                        f"Assistant response (first 200 chars): {answer[:200]}\n\n"
                        "Title:"
                    )
                }
            ]
            
            response = self.llm_router.complete(
                messages=messages,
                system_prompt=system_prompt,
                max_tokens=20,
                temperature=0.5
            )
            
            clean_title = response.strip().strip('"').strip("'")
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