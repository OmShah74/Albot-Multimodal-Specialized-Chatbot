"""
Namespace Resolver — Determines the active retrieval scope for a session.

Active scope = union of:
1. KB sources matching source_filters (e.g. ["report.pdf", "notes.txt"])
2. Memory fragments from active_namespaces (e.g. ["session_abc", "global"])
3. Optionally, past web search results from selected sessions
"""

from typing import Optional
from loguru import logger

from backend.models.memory import RetrievalScope, SessionMemoryConfig


class NamespaceResolver:
    """
    Resolves the active retrieval scope for a given session.
    
    Reads the session's memory configuration and produces a RetrievalScope
    that the retrieval engine uses to constrain searches.
    """

    def __init__(self, memory_manager):
        """
        Args:
            memory_manager: MemoryManager instance
        """
        self.memory = memory_manager

    def resolve(self, session_id: str) -> RetrievalScope:
        """
        Resolve the active retrieval scope for a session.
        
        Args:
            session_id: Chat session ID
            
        Returns:
            RetrievalScope with source_filters, active_namespaces, and flags
        """
        try:
            config = self.memory.get_session_config(session_id)
            
            if config is None:
                # No config saved — return fully open scope (default behavior)
                return RetrievalScope()

            return RetrievalScope(
                source_filters=config.source_filters,
                active_namespaces=config.active_namespaces,
                include_web_history=config.include_web_history,
                include_fragments=config.include_fragments
            )

        except Exception as e:
            logger.error(f"Failed to resolve namespace for session {session_id}: {e}")
            # Graceful fallback: return fully open scope
            return RetrievalScope()

    def get_available_namespaces(self, session_id: str = None) -> dict:
        """
        Get a summary of all available namespaces for the UI.
        
        Returns:
            Dict with namespace names and fragment counts
        """
        try:
            # "global" is always available
            namespaces = {"global": 0}

            # Count fragments per namespace
            global_frags = self.memory.get_fragments_by_namespace(["global"])
            namespaces["global"] = len(global_frags)

            # If a specific session is requested, include its namespace
            if session_id:
                session_frags = self.memory.get_session_fragments(session_id)
                # Session namespace is "session_{id_prefix}"
                ns_name = f"session_{session_id[:8]}"
                namespaces[ns_name] = len(session_frags)

            return namespaces

        except Exception as e:
            logger.error(f"Failed to get available namespaces: {e}")
            return {"global": 0}
