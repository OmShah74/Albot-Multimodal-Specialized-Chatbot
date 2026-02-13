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
        """Initialize the database schema with chats and messages tables"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Enable foreign keys
                cursor.execute("PRAGMA foreign_keys = ON")
                
                # Check if messages table exists and has chat_id
                # Simplified migration: If old schema exists (no chat_id), drop it.
                # In production, we would migrate data.
                cursor.execute("PRAGMA table_info(messages)")
                columns = [info[1] for info in cursor.fetchall()]
                
                # If table exists (columns not empty) and chat_id is missing
                if columns and 'chat_id' not in columns:
                    logger.warning(f"Old schema detected (columns={columns}). Dropping messages table to migrate.")
                    cursor.execute("DROP TABLE IF EXISTS messages")
                
                # Create chats table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chats (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)

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
                
                conn.commit()
                logger.info("SQLite chat history database initialized")
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
                    INSERT INTO chats (id, title, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                """, (chat_id, title, now, now))
                conn.commit()
            
            logger.info(f"Created new chat: {chat_id} ({title})")
            return {
                "id": chat_id,
                "title": title,
                "created_at": now,
                "updated_at": now
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
                    SELECT id, title, created_at, updated_at
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
        """Delete a chat and its messages"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # Cascade delete should handle messages, but explicit is safer in some SQLite versions/configs
                cursor.execute("PRAGMA foreign_keys = ON")
                cursor.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
                conn.commit()
            logger.info(f"Deleted chat {chat_id}")
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
                    # Insert the requested chat_id manually
                    cursor.execute("INSERT INTO chats (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)", 
                                  (chat_id, "New Chat", timestamp, timestamp))

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
                    SELECT role, content, timestamp, sources, metrics
                    FROM messages
                    WHERE chat_id = ?
                    ORDER BY timestamp ASC
                """, (chat_id,))
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
