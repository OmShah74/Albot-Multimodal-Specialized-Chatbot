"""
Memory Manager — Central coordinator for all memory operations.
Bridges SQLite (structured metadata/logs) and ArangoDB (embeddings/graph).
"""

import uuid
from datetime import datetime
from typing import List, Dict, Optional
from loguru import logger

from backend.models.memory import (
    MemoryFragment,
    ReasoningTrace,
    WebInteractionLog,
    SessionMemoryConfig,
)


class MemoryManager:
    """
    Central manager for all memory operations.
    
    Coordinates between:
    - SQLite  → structured logs (traces, fragments metadata, web logs, config)
    - ArangoDB → embeddings + graph edges (for semantic search over fragments)
    """

    def __init__(self, sqlite_storage, arango_storage, vectorizer):
        """
        Args:
            sqlite_storage: SQLiteStorageManager instance
            arango_storage: ArangoStorageManager instance
            vectorizer: Object with .encode(text) -> List[float]  (embedding model)
        """
        self.sqlite = sqlite_storage
        self.arango = arango_storage
        self.vectorizer = vectorizer

    # ─────────────────────────────────────────────
    # Reasoning Traces
    # ─────────────────────────────────────────────

    def log_reasoning_trace(self, trace: ReasoningTrace):
        """Persist a structured reasoning trace for a conversation turn."""
        trace_data = trace.model_dump()
        self.sqlite.save_reasoning_trace(trace_data)

    def get_traces(self, session_id: str) -> List[Dict]:
        """Get all reasoning traces for a session."""
        return self.sqlite.get_reasoning_traces(session_id)

    def get_turn_count(self, session_id: str) -> int:
        """Get the number of completed turns in a session."""
        return self.sqlite.get_turn_count(session_id)

    # ─────────────────────────────────────────────
    # Memory Fragments
    # ─────────────────────────────────────────────

    def store_fragment(self, fragment: MemoryFragment, related_doc_ids: List[str] = None):
        """
        Store a memory fragment in both SQLite (metadata) and ArangoDB (embedding + graph).
        
        Args:
            fragment: The MemoryFragment to store
            related_doc_ids: Optional list of KB atom IDs to create graph edges to
        """
        # 1. Generate embedding
        try:
            embedding = self.vectorizer.encode(fragment.content)
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
        except Exception as e:
            logger.warning(f"Failed to generate embedding for fragment: {e}")
            embedding = []

        # 2. Store in ArangoDB (embedding + graph)
        arango_data = fragment.model_dump()
        arango_data["embedding"] = embedding
        arango_data["related_doc_ids"] = related_doc_ids or []
        embedding_id = self.arango.insert_memory_fragment(arango_data)
        
        # 3. Update fragment with embedding reference
        fragment_data = fragment.model_dump()
        fragment_data["embedding_id"] = embedding_id
        
        # 4. Store in SQLite (structured metadata)
        self.sqlite.save_memory_fragment(fragment_data)

    def search_fragments(
        self,
        query: str,
        namespaces: List[str],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search memory fragments by semantic similarity, scoped to namespaces.
        Uses ArangoDB vector search for semantic matching.
        """
        try:
            query_embedding = self.vectorizer.encode(query)
            if hasattr(query_embedding, 'tolist'):
                query_embedding = query_embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to encode query for fragment search: {e}")
            return []

        results = self.arango.search_memory_fragments(
            query_embedding=query_embedding,
            namespace_filter=namespaces,
            top_k=top_k
        )

        # Update access counts for retrieved fragments
        for result in results:
            self.sqlite.update_fragment_access(result["fragment_id"])

        return results

    def delete_fragment(self, fragment_id: str):
        """
        Soft-delete a fragment from SQLite and hard-delete from ArangoDB.
        """
        self.sqlite.delete_memory_fragment(fragment_id)
        self.arango.delete_single_memory_fragment(fragment_id)

    def get_session_fragments(self, session_id: str) -> List[Dict]:
        """Get all active fragments for a session from SQLite."""
        return self.sqlite.get_session_fragments(session_id)

    def get_fragments_by_namespace(self, namespaces: List[str]) -> List[Dict]:
        """Get all active fragments in the given namespaces from SQLite."""
        return self.sqlite.get_fragments_by_namespace(namespaces)

    # ─────────────────────────────────────────────
    # Session Memory Config
    # ─────────────────────────────────────────────

    def set_session_config(self, config: SessionMemoryConfig):
        """Save or update session memory configuration."""
        self.sqlite.save_session_memory_config(config.model_dump())

    def get_session_config(self, session_id: str) -> Optional[SessionMemoryConfig]:
        """Get memory configuration for a session. Returns default if none saved."""
        data = self.sqlite.get_session_memory_config(session_id)
        if data:
            return SessionMemoryConfig(**data)
        # Return default config
        return SessionMemoryConfig(
            session_id=session_id,
            active_namespaces=["global"],
            source_filters=[],
            include_web_history=True,
            include_fragments=True,
            updated_at=datetime.utcnow().isoformat()
        )

    # ─────────────────────────────────────────────
    # Cross-Session Combination
    # ─────────────────────────────────────────────

    def combine_sessions(
        self,
        source_session_ids: List[str],
        target_namespace: str,
        user_selected_fragment_ids: List[str] = None
    ) -> int:
        """
        Combine memory from multiple sessions into a target namespace.
        
        If user_selected_fragment_ids is provided, only those fragments are cloned.
        Otherwise, all active fragments from source sessions are combined.
        
        Returns the number of fragments copied.
        """
        if user_selected_fragment_ids:
            # User explicitly selected which fragments to merge
            self.sqlite.clone_fragments_to_namespace(user_selected_fragment_ids, target_namespace)
            return len(user_selected_fragment_ids)
        
        # Gather all fragment IDs from source sessions
        all_fragment_ids = []
        for sid in source_session_ids:
            fragments = self.sqlite.get_session_fragments(sid)
            all_fragment_ids.extend([f["fragment_id"] for f in fragments])
        
        if all_fragment_ids:
            self.sqlite.clone_fragments_to_namespace(all_fragment_ids, target_namespace)
        
        return len(all_fragment_ids)

    # ─────────────────────────────────────────────
    # Web Interaction Logs
    # ─────────────────────────────────────────────

    def log_web_interactions(
        self,
        session_id: str,
        turn_index: int,
        query: str,
        results: List
    ):
        """
        Log web search results for a turn.
        
        Args:
            results: List of SearchResult objects from web_search_engine
        """
        logs = []
        for r in results:
            logs.append({
                "session_id": session_id,
                "turn_index": turn_index,
                "provider": getattr(r, 'source_provider', 'unknown'),
                "query_sent": query,
                "url": getattr(r, 'url', None),
                "title": getattr(r, 'title', None),
                "snippet": getattr(r, 'snippet', None),
                "relevance_score": getattr(r, 'final_score', 0.0),
            })
        self.sqlite.save_web_interaction_logs(logs)

    def get_web_history(self, session_id: str) -> List[Dict]:
        """Get all web interaction logs for a session."""
        return self.sqlite.get_web_interaction_logs(session_id)

    # ─────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────

    def delete_session_memory(self, session_id: str):
        """
        Full cascade delete of all memory artifacts for a session.
        SQLite cascade handles: traces, fragments, web_logs, config.
        ArangoDB cleanup: remove fragment nodes and edges.
        """
        self.arango.delete_memory_fragments_by_session(session_id)
        # SQLite cascade runs when the chat is deleted via delete_chat()
        logger.info(f"Cleaned up memory artifacts for session {session_id}")
