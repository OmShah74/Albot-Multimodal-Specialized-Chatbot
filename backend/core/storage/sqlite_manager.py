import sqlite3
import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Optional
from loguru import logger
from pathlib import Path

class SQLiteStorageManager:
    """
    SQLite storage manager for chat history persistence.
    Supports multiple chat sessions.
    """
    
    def __init__(self, db_path: str = "data/database/chat_history.db"):
        self.db_path = db_path
        self._ensure_db_dir()
        self._init_db()
        
    def _ensure_db_dir(self):
        """Ensure the directory for the database exists"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def _get_connection(self):
        """Get a database connection"""
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Initialize the database schema with chats, messages, and memory tables"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Enable foreign keys
                cursor.execute("PRAGMA foreign_keys = ON")
                
                # Check if messages table exists and has chat_id
                # Simplified migration: If old schema exists (no chat_id), drop it.
                cursor.execute("PRAGMA table_info(messages)")
                columns = [info[1] for info in cursor.fetchall()]
                
                # If table exists (columns not empty) and chat_id is missing
                if columns and 'chat_id' not in columns:
                    logger.warning(f"Old schema detected (columns={columns}). Dropping messages table to migrate.")
                    cursor.execute("DROP TABLE IF EXISTS messages")
                
                # ── Core tables ──────────────────────────────────

                # Create chats table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chats (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)

                # Migration: add namespace column if missing
                cursor.execute("PRAGMA table_info(chats)")
                chat_cols = [info[1] for info in cursor.fetchall()]
                if 'namespace' not in chat_cols:
                    cursor.execute("ALTER TABLE chats ADD COLUMN namespace TEXT DEFAULT 'default'")
                    logger.info("Migrated chats table: added namespace column")

                # Create messages table with chat_id foreign key
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        chat_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        sources TEXT,
                        metrics TEXT,
                        FOREIGN KEY (chat_id) REFERENCES chats (id) ON DELETE CASCADE
                    )
                """)
                
                # ── Memory tables ────────────────────────────────

                # Structured reasoning traces per conversation turn
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS reasoning_traces (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        turn_index INTEGER NOT NULL,
                        user_query TEXT NOT NULL,
                        reformulated_query TEXT,
                        retrieved_doc_ids TEXT,
                        retrieved_doc_sources TEXT,
                        retrieval_scores TEXT,
                        algorithms_used TEXT,
                        web_search_triggered INTEGER DEFAULT 0,
                        web_urls_searched TEXT,
                        web_snippets TEXT,
                        search_mode TEXT DEFAULT 'web_search',
                        synthesis_model TEXT,
                        answer_summary TEXT,
                        total_time_ms REAL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (session_id) REFERENCES chats (id) ON DELETE CASCADE
                    )
                """)

                # Memory fragments (solutions, knowledge, preferences, entities)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS memory_fragments (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        fragment_type TEXT NOT NULL,
                        content TEXT NOT NULL,
                        tags TEXT,
                        embedding_id TEXT,
                        namespace TEXT DEFAULT 'global',
                        importance_score REAL DEFAULT 0.5,
                        access_count INTEGER DEFAULT 0,
                        created_at TEXT NOT NULL,
                        last_accessed TEXT,
                        is_deleted INTEGER DEFAULT 0,
                        FOREIGN KEY (session_id) REFERENCES chats (id) ON DELETE CASCADE
                    )
                """)

                # Session memory configuration
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS session_memory_config (
                        session_id TEXT PRIMARY KEY,
                        active_namespaces TEXT,
                        source_filters TEXT,
                        include_web_history INTEGER DEFAULT 1,
                        include_fragments INTEGER DEFAULT 1,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY (session_id) REFERENCES chats (id) ON DELETE CASCADE
                    )
                """)

                # Web interaction logs
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS web_interaction_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        turn_index INTEGER NOT NULL,
                        provider TEXT NOT NULL,
                        query_sent TEXT NOT NULL,
                        url TEXT,
                        title TEXT,
                        snippet TEXT,
                        relevance_score REAL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (session_id) REFERENCES chats (id) ON DELETE CASCADE
                    )
                """)
                
                conn.commit()
                logger.info("SQLite database initialized (core + memory tables)")
        except Exception as e:
            logger.error(f"Failed to initialize SQLite DB: {e}")
            raise

    # --- Chat Management Methods ---

    def create_chat(self, title: str = "New Chat") -> Dict:
        """Create a new chat session"""
        try:
            chat_id = str(uuid.uuid4())
            now = datetime.utcnow().isoformat()
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO chats (id, title, created_at, updated_at, namespace)
                    VALUES (?, ?, ?, ?, ?)
                """, (chat_id, title, now, now, "default"))
                conn.commit()
            
            logger.info(f"Created new chat: {chat_id} ({title})")
            return {
                "id": chat_id,
                "title": title,
                "created_at": now,
                "updated_at": now,
                "namespace": "default"
            }
        except Exception as e:
            logger.error(f"Failed to create chat: {e}")
            raise

    def get_chats(self) -> List[Dict]:
        """Get all chats ordered by updated_at desc"""
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, title, created_at, updated_at, namespace
                    FROM chats
                    ORDER BY updated_at DESC
                """)
                rows = cursor.fetchall()
                
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get chats: {e}")
            return []

    def get_chat(self, chat_id: str) -> Optional[Dict]:
        """Get a specific chat by ID"""
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM chats WHERE id = ?", (chat_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"Failed to get chat {chat_id}: {e}")
            return None

    def rename_chat(self, chat_id: str, new_title: str):
        """Rename a chat"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.utcnow().isoformat()
                cursor.execute("""
                    UPDATE chats 
                    SET title = ?, updated_at = ?
                    WHERE id = ?
                """, (new_title, now, chat_id))
                conn.commit()
            logger.info(f"Renamed chat {chat_id} to '{new_title}'")
        except Exception as e:
            logger.error(f"Failed to rename chat {chat_id}: {e}")

    def delete_chat(self, chat_id: str):
        """Delete a chat and all its associated data (messages, traces, fragments, logs, config)"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA foreign_keys = ON")
                # CASCADE handles: messages, reasoning_traces, memory_fragments,
                # session_memory_config, web_interaction_logs
                cursor.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
                conn.commit()
            logger.info(f"Deleted chat {chat_id} (cascade: traces, fragments, logs, config)")
        except Exception as e:
            logger.error(f"Failed to delete chat {chat_id}: {e}")

    def update_chat_timestamp(self, chat_id: str):
        """Update the updated_at timestamp of a chat"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.utcnow().isoformat()
                cursor.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (now, chat_id))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to update timestamp for chat {chat_id}: {e}")

    def delete_message(self, message_id: int):
        """Delete a single message from chat history"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM messages WHERE id = ?", (message_id,))
                conn.commit()
            logger.info(f"Deleted message {message_id}")
        except Exception as e:
            logger.error(f"Failed to delete message {message_id}: {e}")

    # --- Message Management Methods ---

    def save_chat_message(self, chat_id: str, role: str, content: str, sources: List[str] = None, metrics: Dict = None):
        """Save a chat message to a specific chat history"""
        try:
            sources_json = json.dumps(sources or [])
            
            # Sanitize metrics before JSON serialization
            def sanitize(obj):
                if isinstance(obj, dict):
                    return {str(k): sanitize(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [sanitize(i) for i in obj]
                elif hasattr(obj, 'item'): # Handle numpy types
                    return obj.item()
                return obj

            metrics_json = json.dumps(sanitize(metrics) if metrics else {})
            timestamp = datetime.utcnow().isoformat()

            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if chat exists, if not create default one (safety fallback)
                cursor.execute("SELECT 1 FROM chats WHERE id = ?", (chat_id,))
                if not cursor.fetchone():
                    logger.warning(f"Chat {chat_id} not found when saving message. Creating it.")
                    cursor.execute("INSERT INTO chats (id, title, created_at, updated_at, namespace) VALUES (?, ?, ?, ?, ?)", 
                                  (chat_id, "New Chat", timestamp, timestamp, "default"))

                cursor.execute("""
                    INSERT INTO messages (chat_id, role, content, timestamp, sources, metrics)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (chat_id, role, content, timestamp, sources_json, metrics_json))
                
                # Update chat timestamp
                cursor.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (timestamp, chat_id))
                
                conn.commit()
            
            logger.info(f"Saved {role} message to chat {chat_id}")
            
        except Exception as e:
            logger.error(f"Failed to save message to SQLite: {e}")

    def get_chat_history(self, chat_id: str, limit: int = 100) -> List[Dict]:
        """Get history for a specific chat"""
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, role, content, timestamp, sources, metrics
                    FROM messages
                    WHERE chat_id = ?
                    ORDER BY timestamp ASC
                """, (chat_id,))
                rows = cursor.fetchall()
                
                history = []
                for row in rows:
                    history.append({
                        "id": row["id"],
                        "role": row["role"],
                        "content": row["content"],
                        "timestamp": row["timestamp"],
                        "sources": json.loads(row["sources"] or "[]"),
                        "metrics": json.loads(row["metrics"] or "{}")
                    })
                
                if limit and len(history) > limit:
                     history = history[-limit:]
                     
                return history
                
        except Exception as e:
            logger.error(f"Failed to get history for chat {chat_id}: {e}")
            return []

    def clear_chat_history(self, chat_id: str):
        """Clear history for a specific chat"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
                conn.commit()
            logger.info(f"Cleared history for chat {chat_id}")
        except Exception as e:
            logger.error(f"Failed to clear history for chat {chat_id}: {e}")

    # ═══════════════════════════════════════════════════
    # Memory System Methods
    # ═══════════════════════════════════════════════════

    # --- Reasoning Traces ---

    def save_reasoning_trace(self, trace_data: Dict):
        """Save a structured reasoning trace for a conversation turn"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO reasoning_traces 
                    (id, session_id, turn_index, user_query, reformulated_query,
                     retrieved_doc_ids, retrieved_doc_sources, retrieval_scores,
                     algorithms_used, web_search_triggered, web_urls_searched,
                     web_snippets, search_mode, synthesis_model, answer_summary,
                     total_time_ms, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trace_data["trace_id"],
                    trace_data["session_id"],
                    trace_data["turn_index"],
                    trace_data["user_query"],
                    trace_data.get("reformulated_query"),
                    json.dumps(trace_data.get("retrieved_doc_ids", [])),
                    json.dumps(trace_data.get("retrieved_doc_sources", [])),
                    json.dumps(trace_data.get("retrieval_scores", {})),
                    json.dumps(trace_data.get("algorithms_used", [])),
                    1 if trace_data.get("web_search_triggered") else 0,
                    json.dumps(trace_data.get("web_urls_searched", [])),
                    json.dumps(trace_data.get("web_snippets", [])),
                    trace_data.get("search_mode", "web_search"),
                    trace_data.get("synthesis_model", ""),
                    trace_data.get("answer_summary", ""),
                    trace_data.get("total_time_ms", 0.0),
                    trace_data.get("created_at", datetime.utcnow().isoformat())
                ))
                conn.commit()
            logger.info(f"Saved reasoning trace {trace_data['trace_id']} for session {trace_data['session_id']}")
        except Exception as e:
            logger.error(f"Failed to save reasoning trace: {e}")

    def get_reasoning_traces(self, session_id: str) -> List[Dict]:
        """Get all reasoning traces for a session"""
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM reasoning_traces
                    WHERE session_id = ?
                    ORDER BY turn_index ASC
                """, (session_id,))
                rows = cursor.fetchall()
                
                traces = []
                for row in rows:
                    traces.append({
                        "trace_id": row["id"],
                        "session_id": row["session_id"],
                        "turn_index": row["turn_index"],
                        "user_query": row["user_query"],
                        "reformulated_query": row["reformulated_query"],
                        "retrieved_doc_ids": json.loads(row["retrieved_doc_ids"] or "[]"),
                        "retrieved_doc_sources": json.loads(row["retrieved_doc_sources"] or "[]"),
                        "retrieval_scores": json.loads(row["retrieval_scores"] or "{}"),
                        "algorithms_used": json.loads(row["algorithms_used"] or "[]"),
                        "web_search_triggered": bool(row["web_search_triggered"]),
                        "web_urls_searched": json.loads(row["web_urls_searched"] or "[]"),
                        "web_snippets": json.loads(row["web_snippets"] or "[]"),
                        "search_mode": row["search_mode"],
                        "synthesis_model": row["synthesis_model"],
                        "answer_summary": row["answer_summary"],
                        "total_time_ms": row["total_time_ms"],
                        "created_at": row["created_at"]
                    })
                return traces
        except Exception as e:
            logger.error(f"Failed to get reasoning traces for {session_id}: {e}")
            return []

    def get_turn_count(self, session_id: str) -> int:
        """Get the number of completed turns in a session"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM reasoning_traces WHERE session_id = ?",
                    (session_id,)
                )
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Failed to get turn count for {session_id}: {e}")
            return 0

    def delete_reasoning_trace(self, trace_id: str):
        """Delete a reasoning trace by ID"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM reasoning_traces WHERE id = ?", (trace_id,))
                conn.commit()
            logger.info(f"Deleted reasoning trace {trace_id}")
        except Exception as e:
            logger.error(f"Failed to delete reasoning trace {trace_id}: {e}")

    # --- Memory Fragments ---

    def save_memory_fragment(self, fragment_data: Dict):
        """Save a memory fragment"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO memory_fragments
                    (id, session_id, fragment_type, content, tags, embedding_id,
                     namespace, importance_score, access_count, created_at, last_accessed, is_deleted)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    fragment_data["fragment_id"],
                    fragment_data["session_id"],
                    fragment_data["fragment_type"],
                    fragment_data["content"],
                    json.dumps(fragment_data.get("tags", [])),
                    fragment_data.get("embedding_id"),
                    fragment_data.get("namespace", "global"),
                    fragment_data.get("importance_score", 0.5),
                    fragment_data.get("access_count", 0),
                    fragment_data.get("created_at", datetime.utcnow().isoformat()),
                    fragment_data.get("last_accessed"),
                    1 if fragment_data.get("is_deleted") else 0
                ))
                conn.commit()
            logger.info(f"Saved memory fragment {fragment_data['fragment_id']}")
        except Exception as e:
            logger.error(f"Failed to save memory fragment: {e}")

    def get_session_fragments(self, session_id: str, include_deleted: bool = False) -> List[Dict]:
        """Get all memory fragments for a session"""
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                query = "SELECT * FROM memory_fragments WHERE session_id = ?"
                if not include_deleted:
                    query += " AND is_deleted = 0"
                query += " ORDER BY created_at ASC"
                cursor.execute(query, (session_id,))
                rows = cursor.fetchall()
                
                return [{
                    "fragment_id": row["id"],
                    "session_id": row["session_id"],
                    "fragment_type": row["fragment_type"],
                    "content": row["content"],
                    "tags": json.loads(row["tags"] or "[]"),
                    "embedding_id": row["embedding_id"],
                    "namespace": row["namespace"],
                    "importance_score": row["importance_score"],
                    "access_count": row["access_count"],
                    "created_at": row["created_at"],
                    "last_accessed": row["last_accessed"],
                    "is_deleted": bool(row["is_deleted"])
                } for row in rows]
        except Exception as e:
            logger.error(f"Failed to get fragments for session {session_id}: {e}")
            return []

    def get_fragments_by_namespace(self, namespaces: List[str]) -> List[Dict]:
        """Get all active fragments in the given namespaces"""
        if not namespaces:
            return []
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                placeholders = ",".join("?" for _ in namespaces)
                cursor.execute(f"""
                    SELECT * FROM memory_fragments
                    WHERE namespace IN ({placeholders}) AND is_deleted = 0
                    ORDER BY importance_score DESC
                """, namespaces)
                rows = cursor.fetchall()
                
                return [{
                    "fragment_id": row["id"],
                    "session_id": row["session_id"],
                    "fragment_type": row["fragment_type"],
                    "content": row["content"],
                    "tags": json.loads(row["tags"] or "[]"),
                    "embedding_id": row["embedding_id"],
                    "namespace": row["namespace"],
                    "importance_score": row["importance_score"],
                    "access_count": row["access_count"],
                    "created_at": row["created_at"],
                    "last_accessed": row["last_accessed"]
                } for row in rows]
        except Exception as e:
            logger.error(f"Failed to get fragments by namespace {namespaces}: {e}")
            return []

    def delete_memory_fragment(self, fragment_id: str):
        """Soft-delete a memory fragment"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE memory_fragments SET is_deleted = 1 WHERE id = ?",
                    (fragment_id,)
                )
                conn.commit()
            logger.info(f"Soft-deleted memory fragment {fragment_id}")
        except Exception as e:
            logger.error(f"Failed to delete fragment {fragment_id}: {e}")

    def hard_delete_fragment(self, fragment_id: str):
        """Permanently delete a memory fragment"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM memory_fragments WHERE id = ?", (fragment_id,))
                conn.commit()
            logger.info(f"Hard-deleted memory fragment {fragment_id}")
        except Exception as e:
            logger.error(f"Failed to hard-delete fragment {fragment_id}: {e}")

    def update_fragment_access(self, fragment_id: str):
        """Increment access count and update last_accessed timestamp"""
        try:
            now = datetime.utcnow().isoformat()
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE memory_fragments
                    SET access_count = access_count + 1, last_accessed = ?
                    WHERE id = ?
                """, (now, fragment_id))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to update fragment access {fragment_id}: {e}")

    def update_fragment_importance(self, fragment_id: str, new_score: float):
        """Update the importance score of a fragment"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE memory_fragments SET importance_score = ? WHERE id = ?",
                    (new_score, fragment_id)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to update importance for fragment {fragment_id}: {e}")

    def get_decayed_fragments(self, max_age_days: int = 90, min_score: float = 0.1) -> List[Dict]:
        """Get fragments eligible for decay pruning"""
        try:
            from datetime import timedelta
            cutoff = (datetime.utcnow() - timedelta(days=max_age_days)).isoformat()
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, importance_score, created_at FROM memory_fragments
                    WHERE is_deleted = 0
                      AND importance_score < ?
                      AND created_at < ?
                """, (min_score, cutoff))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get decayed fragments: {e}")
            return []

    # --- Session Memory Config ---

    def save_session_memory_config(self, config_data: Dict):
        """Save or update session memory configuration"""
        try:
            now = datetime.utcnow().isoformat()
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO session_memory_config
                    (session_id, active_namespaces, source_filters, include_web_history, include_fragments, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        active_namespaces = excluded.active_namespaces,
                        source_filters = excluded.source_filters,
                        include_web_history = excluded.include_web_history,
                        include_fragments = excluded.include_fragments,
                        updated_at = excluded.updated_at
                """, (
                    config_data["session_id"],
                    json.dumps(config_data.get("active_namespaces", ["global"])),
                    json.dumps(config_data.get("source_filters", [])),
                    1 if config_data.get("include_web_history", True) else 0,
                    1 if config_data.get("include_fragments", True) else 0,
                    now
                ))
                conn.commit()
            logger.info(f"Saved memory config for session {config_data['session_id']}")
        except Exception as e:
            logger.error(f"Failed to save session memory config: {e}")

    def get_session_memory_config(self, session_id: str) -> Optional[Dict]:
        """Get memory configuration for a session"""
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM session_memory_config WHERE session_id = ?",
                    (session_id,)
                )
                row = cursor.fetchone()
                if not row:
                    return None
                return {
                    "session_id": row["session_id"],
                    "active_namespaces": json.loads(row["active_namespaces"] or '["global"]'),
                    "source_filters": json.loads(row["source_filters"] or "[]"),
                    "include_web_history": bool(row["include_web_history"]),
                    "include_fragments": bool(row["include_fragments"]),
                    "updated_at": row["updated_at"]
                }
        except Exception as e:
            logger.error(f"Failed to get memory config for {session_id}: {e}")
            return None

    # --- Web Interaction Logs ---

    def save_web_interaction_logs(self, logs: List[Dict]):
        """Batch-save web interaction logs for a turn"""
        if not logs:
            return
        try:
            now = datetime.utcnow().isoformat()
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany("""
                    INSERT INTO web_interaction_logs
                    (session_id, turn_index, provider, query_sent, url, title, snippet, relevance_score, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [(
                    log["session_id"],
                    log["turn_index"],
                    log["provider"],
                    log["query_sent"],
                    log.get("url"),
                    log.get("title"),
                    log.get("snippet"),
                    log.get("relevance_score", 0.0),
                    now
                ) for log in logs])
                conn.commit()
            logger.info(f"Saved {len(logs)} web interaction logs")
        except Exception as e:
            logger.error(f"Failed to save web interaction logs: {e}")

    def get_web_interaction_logs(self, session_id: str) -> List[Dict]:
        """Get all web interaction logs for a session"""
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM web_interaction_logs
                    WHERE session_id = ?
                    ORDER BY turn_index ASC, id ASC
                """, (session_id,))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get web logs for {session_id}: {e}")
            return []

    def delete_web_interaction_log(self, log_id: int):
        """Delete a web interaction log by ID"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM web_interaction_logs WHERE id = ?", (log_id,))
                conn.commit()
            logger.info(f"Deleted web log {log_id}")
        except Exception as e:
            logger.error(f"Failed to delete web log {log_id}: {e}")

    # --- Bulk Memory Ops ---

    def clone_fragments_to_namespace(self, fragment_ids: List[str], target_namespace: str):
        """Clone selected fragments into a new namespace (for cross-session combination)"""
        if not fragment_ids:
            return
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                placeholders = ",".join("?" for _ in fragment_ids)
                cursor.execute(f"""
                    SELECT * FROM memory_fragments
                    WHERE id IN ({placeholders}) AND is_deleted = 0
                """, fragment_ids)
                rows = cursor.fetchall()
                
                now = datetime.utcnow().isoformat()
                for row in rows:
                    new_id = str(uuid.uuid4())
                    cursor.execute("""
                        INSERT INTO memory_fragments
                        (id, session_id, fragment_type, content, tags, embedding_id,
                         namespace, importance_score, access_count, created_at, last_accessed, is_deleted)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                    """, (
                        new_id,
                        row["session_id"],
                        row["fragment_type"],
                        row["content"],
                        row["tags"],
                        row["embedding_id"],
                        target_namespace,
                        row["importance_score"],
                        0,
                        now,
                        None
                    ))
                conn.commit()
            logger.info(f"Cloned {len(fragment_ids)} fragments to namespace '{target_namespace}'")
        except Exception as e:
            logger.error(f"Failed to clone fragments: {e}")

