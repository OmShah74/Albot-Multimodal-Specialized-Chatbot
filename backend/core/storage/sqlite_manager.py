import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Optional
from loguru import logger
from pathlib import Path

class SQLiteStorageManager:
    """
    SQLite storage manager specifically for chat history persistence.
    Replaces ArangoDB for chat history to ensure robust persistence across reloads.
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
        """Initialize the database schema"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # Create messages table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        sources TEXT,  -- JSON list
                        metrics TEXT   -- JSON dict
                    )
                """)
                conn.commit()
                logger.info("SQLite chat history database initialized")
        except Exception as e:
            logger.error(f"Failed to initialize SQLite DB: {e}")
            raise

    def save_chat_message(self, role: str, content: str, sources: List[str] = None, metrics: Dict = None):
        """Save a chat message to history"""
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
                cursor.execute("""
                    INSERT INTO messages (role, content, timestamp, sources, metrics)
                    VALUES (?, ?, ?, ?, ?)
                """, (role, content, timestamp, sources_json, metrics_json))
                conn.commit()
            
            logger.info(f"Saved {role} message to SQLite history")
            
        except Exception as e:
            logger.error(f"Failed to save message to SQLite: {e}")

    def get_chat_history(self, limit: int = 100) -> List[Dict]:
        """Get recent chat history"""
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                # Get last 'limit' messages, ordered by timestamp
                cursor.execute("""
                    SELECT role, content, timestamp, sources, metrics
                    FROM messages
                    ORDER BY timestamp ASC
                """)
                rows = cursor.fetchall()
                
                history = []
                for row in rows:
                    history.append({
                        "role": row["role"],
                        "content": row["content"],
                        "timestamp": row["timestamp"],
                        "sources": json.loads(row["sources"] or "[]"),
                        "metrics": json.loads(row["metrics"] or "{}")
                    })
                
                # If we need to limit strictly to the *last* N messages but keep them chronological:
                # The query gets ALL, so we slice the last 'limit'
                if limit and len(history) > limit:
                     history = history[-limit:]
                     
                return history
                
        except Exception as e:
            logger.error(f"Failed to get history from SQLite: {e}")
            return []

    def clear_chat_history(self):
        """Clear entire chat history"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM messages")
                conn.commit()
            logger.info("SQLite chat history cleared")
        except Exception as e:
            logger.error(f"Failed to clear SQLite history: {e}")
