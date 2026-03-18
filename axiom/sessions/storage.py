"""Session Persistence & State Management for axiom

Save and restore complete browser sessions (cookies, localStorage, sessionStorage)
across runs with automatic session rotation and cookie jar management.
"""

import json
import sqlite3
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import threading
from contextlib import contextmanager
import logging
import zlib
import base64

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from axiom.core.custom_types import SessionData, Cookie, StorageItem
from axiom.core.storage import BaseStorage
from axiom.core.utils._utils import get_logger

logger = get_logger(__name__)


class SessionStorageError(Exception):
    """Base exception for session storage operations."""
    pass


class SessionExpiredError(SessionStorageError):
    """Raised when trying to access an expired session."""
    pass


class SessionStorage(BaseStorage):
    """Manages persistent storage for browser sessions.
    
    Supports SQLite and Redis backends with automatic session aging,
    rotation, and HAR file import/export capabilities.
    """
    
    # Session aging defaults
    DEFAULT_SESSION_TTL = 86400 * 7  # 7 days in seconds
    DEFAULT_MAX_SESSIONS = 1000
    DEFAULT_CLEANUP_INTERVAL = 3600  # 1 hour
    
    def __init__(
        self,
        backend: str = "sqlite",
        connection_string: Optional[str] = None,
        session_ttl: int = DEFAULT_SESSION_TTL,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        auto_cleanup: bool = True,
        compression: bool = True,
        **kwargs
    ):
        """Initialize session storage.
        
        Args:
            backend: Storage backend ('sqlite' or 'redis')
            connection_string: Connection string for backend
            session_ttl: Session time-to-live in seconds
            max_sessions: Maximum number of sessions to store
            auto_cleanup: Enable automatic cleanup of expired sessions
            compression: Enable compression for session data
            **kwargs: Additional backend-specific arguments
        """
        super().__init__()
        self.backend = backend.lower()
        self.session_ttl = session_ttl
        self.max_sessions = max_sessions
        self.auto_cleanup = auto_cleanup
        self.compression = compression
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
        
        # Initialize backend
        if self.backend == "sqlite":
            self._init_sqlite(connection_string or "sessions.db")
        elif self.backend == "redis":
            if not REDIS_AVAILABLE:
                raise SessionStorageError("Redis backend requires redis-py package")
            self._init_redis(connection_string or "redis://localhost:6379/0", **kwargs)
        else:
            raise SessionStorageError(f"Unsupported backend: {backend}")
        
        # Start cleanup thread if enabled
        if auto_cleanup:
            self._start_cleanup_thread()
    
    def _init_sqlite(self, db_path: str):
        """Initialize SQLite backend."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        
        # Create tables
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    metadata TEXT,
                    compressed INTEGER DEFAULT 0
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cookies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    value TEXT,
                    domain TEXT,
                    path TEXT,
                    expires REAL,
                    http_only INTEGER,
                    secure INTEGER,
                    same_site TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS storage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    storage_type TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE,
                    UNIQUE(session_id, storage_type, key)
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cookies_session ON cookies(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_storage_session ON storage(session_id)")
            
            conn.commit()
    
    def _init_redis(self, connection_string: str, **kwargs):
        """Initialize Redis backend."""
        self.redis_client = redis.from_url(connection_string, **kwargs)
        self.redis_prefix = "axiom:session:"
    
    @contextmanager
    def _get_connection(self):
        """Get SQLite connection with thread safety."""
        if not hasattr(self._local, "connection"):
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0
            )
            self._local.connection.row_factory = sqlite3.Row
        
        try:
            yield self._local.connection
        except Exception:
            self._local.connection.rollback()
            raise
    
    def _compress_data(self, data: str) -> str:
        """Compress data if compression is enabled."""
        if not self.compression:
            return data
        
        compressed = zlib.compress(data.encode('utf-8'))
        return base64.b64encode(compressed).decode('ascii')
    
    def _decompress_data(self, data: str) -> str:
        """Decompress data if it was compressed."""
        if not self.compression:
            return data
        
        try:
            compressed = base64.b64decode(data.encode('ascii'))
            return zlib.decompress(compressed).decode('utf-8')
        except Exception:
            # Data might not be compressed
            return data
    
    def _session_to_dict(self, session: SessionData) -> Dict[str, Any]:
        """Convert SessionData to dictionary for storage."""
        return {
            "cookies": [cookie.dict() for cookie in session.cookies],
            "local_storage": session.local_storage,
            "session_storage": session.session_storage,
            "user_agent": session.user_agent,
            "viewport": session.viewport,
            "extra": session.extra
        }
    
    def _dict_to_session(self, session_id: str, data: Dict[str, Any]) -> SessionData:
        """Convert dictionary to SessionData."""
        cookies = [Cookie(**cookie) for cookie in data.get("cookies", [])]
        return SessionData(
            session_id=session_id,
            cookies=cookies,
            local_storage=data.get("local_storage", {}),
            session_storage=data.get("session_storage", {}),
            user_agent=data.get("user_agent"),
            viewport=data.get("viewport"),
            extra=data.get("extra", {})
        )
    
    def create_session(
        self,
        session_id: Optional[str] = None,
        cookies: Optional[List[Cookie]] = None,
        local_storage: Optional[Dict[str, str]] = None,
        session_storage: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
        **kwargs
    ) -> str:
        """Create a new session.
        
        Args:
            session_id: Optional session ID (generated if not provided)
            cookies: List of cookies
            local_storage: Local storage key-value pairs
            session_storage: Session storage key-value pairs
            metadata: Additional metadata
            ttl: Session TTL in seconds (overrides default)
            **kwargs: Additional session attributes
            
        Returns:
            Session ID
        """
        session_id = session_id or str(uuid.uuid4())
        now = time.time()
        expires_at = now + (ttl or self.session_ttl)
        
        session_data = SessionData(
            session_id=session_id,
            cookies=cookies or [],
            local_storage=local_storage or {},
            session_storage=session_storage or {},
            **kwargs
        )
        
        # Check max sessions limit
        self._enforce_max_sessions()
        
        # Store session
        if self.backend == "sqlite":
            self._store_session_sqlite(session_id, session_data, metadata, expires_at, now)
        else:
            self._store_session_redis(session_id, session_data, metadata, expires_at, now)
        
        logger.info(f"Created session: {session_id}")
        return session_id
    
    def _store_session_sqlite(
        self,
        session_id: str,
        session_data: SessionData,
        metadata: Optional[Dict[str, Any]],
        expires_at: float,
        now: float
    ):
        """Store session in SQLite."""
        with self._get_connection() as conn:
            # Store session metadata
            session_dict = self._session_to_dict(session_data)
            data_json = json.dumps(session_dict)
            compressed = 1 if self.compression else 0
            
            if self.compression:
                data_json = self._compress_data(data_json)
            
            conn.execute(
                """INSERT OR REPLACE INTO sessions 
                (session_id, created_at, updated_at, expires_at, metadata, compressed)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (session_id, now, now, expires_at,
                 json.dumps(metadata) if metadata else None, compressed)
            )
            
            # Clear existing cookies and storage
            conn.execute("DELETE FROM cookies WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM storage WHERE session_id = ?", (session_id,))
            
            # Store cookies
            for cookie in session_data.cookies:
                conn.execute(
                    """INSERT INTO cookies 
                    (session_id, name, value, domain, path, expires, http_only, secure, same_site)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (session_id, cookie.name, cookie.value, cookie.domain, cookie.path,
                     cookie.expires, int(cookie.http_only), int(cookie.secure), cookie.same_site)
                )
            
            # Store local storage
            for key, value in session_data.local_storage.items():
                conn.execute(
                    """INSERT INTO storage (session_id, storage_type, key, value)
                    VALUES (?, 'local', ?, ?)""",
                    (session_id, key, value)
                )
            
            # Store session storage
            for key, value in session_data.session_storage.items():
                conn.execute(
                    """INSERT INTO storage (session_id, storage_type, key, value)
                    VALUES (?, 'session', ?, ?)""",
                    (session_id, key, value)
                )
            
            conn.commit()
    
    def _store_session_redis(
        self,
        session_id: str,
        session_data: SessionData,
        metadata: Optional[Dict[str, Any]],
        expires_at: float,
        now: float
    ):
        """Store session in Redis."""
        pipe = self.redis_client.pipeline()
        
        # Store session data
        session_dict = self._session_to_dict(session_data)
        session_key = f"{self.redis_prefix}{session_id}"
        
        # Use hash for session metadata
        pipe.hset(session_key, mapping={
            "data": json.dumps(session_dict),
            "created_at": str(now),
            "updated_at": str(now),
            "expires_at": str(expires_at),
            "metadata": json.dumps(metadata) if metadata else ""
        })
        
        # Set TTL on session
        ttl = int(expires_at - now)
        pipe.expire(session_key, ttl)
        
        # Store cookies in a sorted set by domain
        cookie_key = f"{session_key}:cookies"
        for cookie in session_data.cookies:
            cookie_data = json.dumps(cookie.dict())
            pipe.hset(cookie_key, cookie.name, cookie_data)
        
        pipe.expire(cookie_key, ttl)
        
        # Store local storage
        local_key = f"{session_key}:local"
        if session_data.local_storage:
            pipe.hset(local_key, mapping=session_data.local_storage)
            pipe.expire(local_key, ttl)
        
        # Store session storage
        session_storage_key = f"{session_key}:session"
        if session_data.session_storage:
            pipe.hset(session_storage_key, mapping=session_data.session_storage)
            pipe.expire(session_storage_key, ttl)
        
        # Add to session index
        pipe.zadd(f"{self.redis_prefix}index", {session_id: expires_at})
        
        pipe.execute()
    
    def get_session(self, session_id: str) -> Optional[SessionData]:
        """Retrieve a session by ID.
        
        Args:
            session_id: Session ID to retrieve
            
        Returns:
            SessionData if found and not expired, None otherwise
            
        Raises:
            SessionExpiredError: If session exists but has expired
        """
        if self.backend == "sqlite":
            return self._get_session_sqlite(session_id)
        else:
            return self._get_session_redis(session_id)
    
    def _get_session_sqlite(self, session_id: str) -> Optional[SessionData]:
        """Get session from SQLite."""
        with self._get_connection() as conn:
            # Check if session exists and is not expired
            cursor = conn.execute(
                """SELECT * FROM sessions 
                WHERE session_id = ? AND expires_at > ?""",
                (session_id, time.time())
            )
            session_row = cursor.fetchone()
            
            if not session_row:
                # Check if session exists but is expired
                cursor = conn.execute(
                    "SELECT expires_at FROM sessions WHERE session_id = ?",
                    (session_id,)
                )
                expired_row = cursor.fetchone()
                if expired_row:
                    raise SessionExpiredError(f"Session {session_id} has expired")
                return None
            
            # Get session data
            data_json = session_row["data"]
            if session_row["compressed"]:
                data_json = self._decompress_data(data_json)
            
            session_dict = json.loads(data_json)
            
            # Get cookies
            cursor = conn.execute(
                "SELECT * FROM cookies WHERE session_id = ?",
                (session_id,)
            )
            cookies = []
            for cookie_row in cursor.fetchall():
                cookies.append(Cookie(
                    name=cookie_row["name"],
                    value=cookie_row["value"],
                    domain=cookie_row["domain"],
                    path=cookie_row["path"],
                    expires=cookie_row["expires"],
                    http_only=bool(cookie_row["http_only"]),
                    secure=bool(cookie_row["secure"]),
                    same_site=cookie_row["same_site"]
                ))
            
            session_dict["cookies"] = cookies
            
            # Get local storage
            cursor = conn.execute(
                """SELECT key, value FROM storage 
                WHERE session_id = ? AND storage_type = 'local'""",
                (session_id,)
            )
            local_storage = {row["key"]: row["value"] for row in cursor.fetchall()}
            session_dict["local_storage"] = local_storage
            
            # Get session storage
            cursor = conn.execute(
                """SELECT key, value FROM storage 
                WHERE session_id = ? AND storage_type = 'session'""",
                (session_id,)
            )
            session_storage = {row["key"]: row["value"] for row in cursor.fetchall()}
            session_dict["session_storage"] = session_storage
            
            return self._dict_to_session(session_id, session_dict)
    
    def _get_session_redis(self, session_id: str) -> Optional[SessionData]:
        """Get session from Redis."""
        session_key = f"{self.redis_prefix}{session_id}"
        
        # Check if session exists
        if not self.redis_client.exists(session_key):
            return None
        
        # Check expiration
        expires_at = float(self.redis_client.hget(session_key, "expires_at") or 0)
        if expires_at < time.time():
            raise SessionExpiredError(f"Session {session_id} has expired")
        
        # Get session data
        data_json = self.redis_client.hget(session_key, "data")
        session_dict = json.loads(data_json)
        
        # Get cookies
        cookie_key = f"{session_key}:cookies"
        cookies_data = self.redis_client.hgetall(cookie_key)
        cookies = []
        for cookie_json in cookies_data.values():
            cookies.append(Cookie(**json.loads(cookie_json)))
        session_dict["cookies"] = cookies
        
        # Get local storage
        local_key = f"{session_key}:local"
        local_storage = self.redis_client.hgetall(local_key)
        session_dict["local_storage"] = local_storage
        
        # Get session storage
        session_storage_key = f"{session_key}:session"
        session_storage = self.redis_client.hgetall(session_storage_key)
        session_dict["session_storage"] = session_storage
        
        return self._dict_to_session(session_id, session_dict)
    
    def update_session(
        self,
        session_id: str,
        cookies: Optional[List[Cookie]] = None,
        local_storage: Optional[Dict[str, str]] = None,
        session_storage: Optional[Dict[str, str]] = None,
        extend_ttl: Optional[int] = None,
        **kwargs
    ) -> bool:
        """Update an existing session.
        
        Args:
            session_id: Session ID to update
            cookies: New cookies (replaces existing)
            local_storage: New local storage (merged with existing)
            session_storage: New session storage (merged with existing)
            extend_ttl: Extend session TTL by this many seconds
            **kwargs: Additional session attributes to update
            
        Returns:
            True if session was updated, False if not found
        """
        # Get existing session
        session = self.get_session(session_id)
        if not session:
            return False
        
        # Update fields
        if cookies is not None:
            session.cookies = cookies
        
        if local_storage is not None:
            session.local_storage.update(local_storage)
        
        if session_storage is not None:
            session.session_storage.update(session_storage)
        
        for key, value in kwargs.items():
            if hasattr(session, key):
                setattr(session, key, value)
        
        # Calculate new expiration
        now = time.time()
        if extend_ttl:
            expires_at = now + extend_ttl
        else:
            # Keep existing expiration
            if self.backend == "sqlite":
                with self._get_connection() as conn:
                    cursor = conn.execute(
                        "SELECT expires_at FROM sessions WHERE session_id = ?",
                        (session_id,)
                    )
                    row = cursor.fetchone()
                    expires_at = row["expires_at"] if row else now + self.session_ttl
            else:
                expires_at = float(
                    self.redis_client.hget(
                        f"{self.redis_prefix}{session_id}", "expires_at"
                    ) or now + self.session_ttl
                )
        
        # Store updated session
        if self.backend == "sqlite":
            self._store_session_sqlite(session_id, session, None, expires_at, now)
            
            # Update timestamp
            with self._get_connection() as conn:
                conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                    (now, session_id)
                )
                conn.commit()
        else:
            self._store_session_redis(session_id, session, None, expires_at, now)
        
        logger.info(f"Updated session: {session_id}")
        return True
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session.
        
        Args:
            session_id: Session ID to delete
            
        Returns:
            True if session was deleted, False if not found
        """
        if self.backend == "sqlite":
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM sessions WHERE session_id = ?",
                    (session_id,)
                )
                conn.commit()
                return cursor.rowcount > 0
        else:
            session_key = f"{self.redis_prefix}{session_id}"
            pipe = self.redis_client.pipeline()
            
            # Delete all session data
            pipe.delete(session_key)
            pipe.delete(f"{session_key}:cookies")
            pipe.delete(f"{session_key}:local")
            pipe.delete(f"{session_key}:session")
            
            # Remove from index
            pipe.zrem(f"{self.redis_prefix}index", session_id)
            
            results = pipe.execute()
            return results[0] > 0
    
    def list_sessions(
        self,
        include_expired: bool = False,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List all sessions.
        
        Args:
            include_expired: Include expired sessions
            limit: Maximum number of sessions to return
            offset: Offset for pagination
            
        Returns:
            List of session metadata dictionaries
        """
        if self.backend == "sqlite":
            return self._list_sessions_sqlite(include_expired, limit, offset)
        else:
            return self._list_sessions_redis(include_expired, limit, offset)
    
    def _list_sessions_sqlite(
        self,
        include_expired: bool,
        limit: int,
        offset: int
    ) -> List[Dict[str, Any]]:
        """List sessions from SQLite."""
        with self._get_connection() as conn:
            now = time.time()
            
            if include_expired:
                query = """
                    SELECT session_id, created_at, updated_at, expires_at, metadata
                    FROM sessions
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                """
                params = (limit, offset)
            else:
                query = """
                    SELECT session_id, created_at, updated_at, expires_at, metadata
                    FROM sessions
                    WHERE expires_at > ?
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                """
                params = (now, limit, offset)
            
            cursor = conn.execute(query, params)
            sessions = []
            
            for row in cursor.fetchall():
                metadata = json.loads(row["metadata"]) if row["metadata"] else {}
                sessions.append({
                    "session_id": row["session_id"],
                    "created_at": datetime.fromtimestamp(row["created_at"]).isoformat(),
                    "updated_at": datetime.fromtimestamp(row["updated_at"]).isoformat(),
                    "expires_at": datetime.fromtimestamp(row["expires_at"]).isoformat(),
                    "expired": row["expires_at"] <= now,
                    **metadata
                })
            
            return sessions
    
    def _list_sessions_redis(
        self,
        include_expired: bool,
        limit: int,
        offset: int
    ) -> List[Dict[str, Any]]:
        """List sessions from Redis."""
        now = time.time()
        
        if include_expired:
            # Get all sessions
            session_ids = self.redis_client.zrange(
                f"{self.redis_prefix}index", 0, -1
            )
        else:
            # Get non-expired sessions
            session_ids = self.redis_client.zrangebyscore(
                f"{self.redis_prefix}index", now, "+inf"
            )
        
        # Apply pagination
        session_ids = session_ids[offset:offset + limit]
        
        sessions = []
        for session_id in session_ids:
            session_key = f"{self.redis_prefix}{session_id}"
            data = self.redis_client.hgetall(session_key)
            
            if data:
                created_at = float(data.get("created_at", 0))
                updated_at = float(data.get("updated_at", 0))
                expires_at = float(data.get("expires_at", 0))
                metadata = json.loads(data.get("metadata", "{}"))
                
                sessions.append({
                    "session_id": session_id.decode() if isinstance(session_id, bytes) else session_id,
                    "created_at": datetime.fromtimestamp(created_at).isoformat(),
                    "updated_at": datetime.fromtimestamp(updated_at).isoformat(),
                    "expires_at": datetime.fromtimestamp(expires_at).isoformat(),
                    "expired": expires_at <= now,
                    **metadata
                })
        
        return sessions
    
    def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions.
        
        Returns:
            Number of sessions removed
        """
        if self.backend == "sqlite":
            return self._cleanup_expired_sqlite()
        else:
            return self._cleanup_expired_redis()
    
    def _cleanup_expired_sqlite(self) -> int:
        """Cleanup expired sessions in SQLite."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE expires_at <= ?",
                (time.time(),)
            )
            conn.commit()
            count = cursor.rowcount
            
            if count > 0:
                logger.info(f"Cleaned up {count} expired sessions")
            
            return count
    
    def _cleanup_expired_redis(self) -> int:
        """Cleanup expired sessions in Redis."""
        now = time.time()
        
        # Get expired session IDs
        expired_ids = self.redis_client.zrangebyscore(
            f"{self.redis_prefix}index", "-inf", now
        )
        
        if not expired_ids:
            return 0
        
        pipe = self.redis_client.pipeline()
        
        for session_id in expired_ids:
            session_key = f"{self.redis_prefix}{session_id}"
            
            # Delete all session data
            pipe.delete(session_key)
            pipe.delete(f"{session_key}:cookies")
            pipe.delete(f"{session_key}:local")
            pipe.delete(f"{session_key}:session")
        
        # Remove from index
        pipe.zremrangebyscore(f"{self.redis_prefix}index", "-inf", now)
        
        pipe.execute()
        
        count = len(expired_ids)
        logger.info(f"Cleaned up {count} expired sessions")
        return count
    
    def _enforce_max_sessions(self):
        """Enforce maximum sessions limit by removing oldest sessions."""
        if self.backend == "sqlite":
            self._enforce_max_sessions_sqlite()
        else:
            self._enforce_max_sessions_redis()
    
    def _enforce_max_sessions_sqlite(self):
        """Enforce max sessions in SQLite."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) as count FROM sessions")
            count = cursor.fetchone()["count"]
            
            if count >= self.max_sessions:
                # Remove oldest sessions
                to_remove = count - self.max_sessions + 1
                conn.execute(
                    """DELETE FROM sessions WHERE session_id IN (
                        SELECT session_id FROM sessions 
                        ORDER BY created_at ASC 
                        LIMIT ?
                    )""",
                    (to_remove,)
                )
                conn.commit()
                logger.info(f"Removed {to_remove} old sessions to enforce limit")
    
    def _enforce_max_sessions_redis(self):
        """Enforce max sessions in Redis."""
        count = self.redis_client.zcard(f"{self.redis_prefix}index")
        
        if count >= self.max_sessions:
            # Remove oldest sessions
            to_remove = count - self.max_sessions + 1
            oldest_sessions = self.redis_client.zrange(
                f"{self.redis_prefix}index", 0, to_remove - 1
            )
            
            pipe = self.redis_client.pipeline()
            for session_id in oldest_sessions:
                session_key = f"{self.redis_prefix}{session_id}"
                pipe.delete(session_key)
                pipe.delete(f"{session_key}:cookies")
                pipe.delete(f"{session_key}:local")
                pipe.delete(f"{session_key}:session")
            
            pipe.zremrangebyrank(f"{self.redis_prefix}index", 0, to_remove - 1)
            pipe.execute()
            
            logger.info(f"Removed {to_remove} old sessions to enforce limit")
    
    def rotate_session(
        self,
        old_session_id: str,
        new_session_id: Optional[str] = None,
        copy_data: bool = True
    ) -> Optional[str]:
        """Rotate a session to a new ID.
        
        Args:
            old_session_id: Current session ID
            new_session_id: New session ID (generated if not provided)
            copy_data: Copy data from old session to new
            
        Returns:
            New session ID if successful, None if old session not found
        """
        # Get old session
        old_session = self.get_session(old_session_id)
        if not old_session:
            return None
        
        new_session_id = new_session_id or str(uuid.uuid4())
        
        if copy_data:
            # Create new session with same data
            self.create_session(
                session_id=new_session_id,
                cookies=old_session.cookies,
                local_storage=old_session.local_storage,
                session_storage=old_session.session_storage,
                user_agent=old_session.user_agent,
                viewport=old_session.viewport,
                extra=old_session.extra
            )
        else:
            # Create empty new session
            self.create_session(session_id=new_session_id)
        
        # Delete old session
        self.delete_session(old_session_id)
        
        logger.info(f"Rotated session {old_session_id} to {new_session_id}")
        return new_session_id
    
    def import_from_har(
        self,
        har_path: Union[str, Path],
        session_id: Optional[str] = None,
        extract_storage: bool = True
    ) -> str:
        """Import session from HAR file.
        
        Args:
            har_path: Path to HAR file
            session_id: Session ID for imported data (generated if not provided)
            extract_storage: Attempt to extract storage from HAR entries
            
        Returns:
            Session ID
        """
        har_path = Path(har_path)
        if not har_path.exists():
            raise SessionStorageError(f"HAR file not found: {har_path}")
        
        try:
            with open(har_path, 'r', encoding='utf-8') as f:
                har_data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise SessionStorageError(f"Invalid HAR file: {e}")
        
        # Extract cookies from HAR
        cookies = []
        cookie_jar = set()  # Track unique cookies
        
        # Get cookies from response headers
        for entry in har_data.get("log", {}).get("entries", []):
            response = entry.get("response", {})
            headers = response.get("headers", [])
            
            for header in headers:
                if header.get("name", "").lower() == "set-cookie":
                    cookie_str = header.get("value", "")
                    cookie = self._parse_cookie_string(cookie_str)
                    if cookie:
                        cookie_key = f"{cookie.name}:{cookie.domain}:{cookie.path}"
                        if cookie_key not in cookie_jar:
                            cookies.append(cookie)
                            cookie_jar.add(cookie_key)
        
        # Extract storage if enabled
        local_storage = {}
        session_storage = {}
        
        if extract_storage:
            # Try to extract from page context if available
            pages = har_data.get("log", {}).get("pages", [])
            for page in pages:
                page_title = page.get("title", "")
                # Some HAR files include storage in page timings or custom fields
                if "localStorage" in page:
                    local_storage.update(page["localStorage"])
                if "sessionStorage" in page:
                    session_storage.update(page["sessionStorage"])
        
        # Create session
        return self.create_session(
            session_id=session_id,
            cookies=cookies,
            local_storage=local_storage,
            session_storage=session_storage,
            metadata={
                "source": "har_import",
                "har_file": str(har_path),
                "imported_at": datetime.now().isoformat()
            }
        )
    
    def export_to_har(
        self,
        session_id: str,
        har_path: Union[str, Path],
        include_storage: bool = True
    ) -> bool:
        """Export session to HAR file.
        
        Args:
            session_id: Session ID to export
            har_path: Output HAR file path
            include_storage: Include storage in export
            
        Returns:
            True if successful, False if session not found
        """
        session = self.get_session(session_id)
        if not session:
            return False
        
        har_path = Path(har_path)
        har_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create HAR structure
        har_data = {
            "log": {
                "version": "1.2",
                "creator": {
                    "name": "axiom Session Storage",
                    "version": "1.0.0"
                },
                "pages": [{
                    "startedDateTime": datetime.now().isoformat(),
                    "id": session_id,
                    "title": f"axiom Session {session_id}",
                    "pageTimings": {}
                }],
                "entries": []
            }
        }
        
        # Add storage to page if requested
        if include_storage:
            har_data["log"]["pages"][0]["localStorage"] = session.local_storage
            har_data["log"]["pages"][0]["sessionStorage"] = session.session_storage
        
        # Create a dummy entry with cookies
        entry = {
            "startedDateTime": datetime.now().isoformat(),
            "time": 0,
            "request": {
                "method": "GET",
                "url": "http://localhost/",
                "httpVersion": "HTTP/1.1",
                "cookies": [],
                "headers": [],
                "queryString": [],
                "headersSize": -1,
                "bodySize": -1
            },
            "response": {
                "status": 200,
                "statusText": "OK",
                "httpVersion": "HTTP/1.1",
                "cookies": [],
                "headers": [],
                "content": {
                    "size": 0,
                    "mimeType": "text/html"
                },
                "redirectURL": "",
                "headersSize": -1,
                "bodySize": -1
            },
            "cache": {},
            "timings": {
                "send": 0,
                "wait": 0,
                "receive": 0
            }
        }
        
        # Add cookies to response
        for cookie in session.cookies:
            entry["response"]["cookies"].append({
                "name": cookie.name,
                "value": cookie.value,
                "domain": cookie.domain or "",
                "path": cookie.path or "/",
                "expires": cookie.expires or "",
                "httpOnly": cookie.http_only,
                "secure": cookie.secure,
                "sameSite": cookie.same_site or ""
            })
        
        har_data["log"]["entries"].append(entry)
        
        # Write HAR file
        try:
            with open(har_path, 'w', encoding='utf-8') as f:
                json.dump(har_data, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to write HAR file: {e}")
            return False
        
        logger.info(f"Exported session {session_id} to {har_path}")
        return True
    
    def _parse_cookie_string(self, cookie_string: str) -> Optional[Cookie]:
        """Parse a Set-Cookie header string into a Cookie object."""
        try:
            parts = [p.strip() for p in cookie_string.split(';')]
            if not parts:
                return None
            
            # Parse name=value
            name_value = parts[0]
            if '=' not in name_value:
                return None
            
            name, value = name_value.split('=', 1)
            
            # Initialize cookie with defaults
            cookie = Cookie(
                name=name,
                value=value,
                domain="",
                path="/",
                expires=None,
                http_only=False,
                secure=False,
                same_site="Lax"
            )
            
            # Parse attributes
            for part in parts[1:]:
                if '=' in part:
                    key, val = part.split('=', 1)
                    key = key.strip().lower()
                    val = val.strip()
                    
                    if key == "domain":
                        cookie.domain = val
                    elif key == "path":
                        cookie.path = val
                    elif key == "expires":
                        try:
                            # Try to parse date string
                            from email.utils import parsedate_to_datetime
                            cookie.expires = parsedate_to_datetime(val).timestamp()
                        except Exception:
                            pass
                    elif key == "max-age":
                        try:
                            cookie.expires = time.time() + int(val)
                        except ValueError:
                            pass
                    elif key == "samesite":
                        cookie.same_site = val
                else:
                    key = part.strip().lower()
                    if key == "httponly":
                        cookie.http_only = True
                    elif key == "secure":
                        cookie.secure = True
            
            return cookie
        except Exception as e:
            logger.warning(f"Failed to parse cookie string: {e}")
            return None
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        def cleanup_worker():
            while not self._stop_cleanup.is_set():
                try:
                    self.cleanup_expired_sessions()
                except Exception as e:
                    logger.error(f"Cleanup error: {e}")
                
                # Wait for next cleanup interval or stop signal
                self._stop_cleanup.wait(self.DEFAULT_CLEANUP_INTERVAL)
        
        self._cleanup_thread = threading.Thread(
            target=cleanup_worker,
            daemon=True,
            name="SessionCleanup"
        )
        self._cleanup_thread.start()
    
    def close(self):
        """Close storage and cleanup resources."""
        self._stop_cleanup.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5.0)
        
        if hasattr(self, '_local') and hasattr(self._local, 'connection'):
            self._local.connection.close()
        
        logger.info("Session storage closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# Factory function for easy instantiation
def create_session_storage(
    backend: str = "sqlite",
    connection_string: Optional[str] = None,
    **kwargs
) -> SessionStorage:
    """Create a session storage instance.
    
    Args:
        backend: Storage backend ('sqlite' or 'redis')
        connection_string: Connection string for backend
        **kwargs: Additional arguments for SessionStorage
        
    Returns:
        Configured SessionStorage instance
    """
    return SessionStorage(
        backend=backend,
        connection_string=connection_string,
        **kwargs
    )


# Convenience class for session management
class SessionManager:
    """High-level session management interface.
    
    Provides simplified methods for common session operations
    with automatic rotation and state management.
    """
    
    def __init__(
        self,
        storage: Optional[SessionStorage] = None,
        **storage_kwargs
    ):
        """Initialize session manager.
        
        Args:
            storage: Existing SessionStorage instance
            **storage_kwargs: Arguments to create new SessionStorage
        """
        self.storage = storage or create_session_storage(**storage_kwargs)
        self._active_sessions = {}
    
    def get_or_create_session(
        self,
        session_id: Optional[str] = None,
        auto_rotate: bool = True,
        **kwargs
    ) -> str:
        """Get existing session or create new one.
        
        Args:
            session_id: Existing session ID
            auto_rotate: Automatically rotate expired sessions
            **kwargs: Arguments for create_session
            
        Returns:
            Session ID
        """
        if session_id:
            try:
                session = self.storage.get_session(session_id)
                if session:
                    return session_id
            except SessionExpiredError:
                if auto_rotate:
                    new_id = self.storage.rotate_session(session_id)
                    if new_id:
                        return new_id
        
        # Create new session
        return self.storage.create_session(**kwargs)
    
    def with_session(self, session_id: Optional[str] = None):
        """Context manager for session operations.
        
        Args:
            session_id: Session ID to use
            
        Returns:
            Context manager that yields session ID
        """
        class SessionContext:
            def __init__(self, manager, sid):
                self.manager = manager
                self.session_id = sid
                self._session = None
            
            def __enter__(self):
                self.session_id = self.manager.get_or_create_session(self.session_id)
                self._session = self.manager.storage.get_session(self.session_id)
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self._session:
                    self.manager.storage.update_session(
                        self.session_id,
                        cookies=self._session.cookies,
                        local_storage=self._session.local_storage,
                        session_storage=self._session.session_storage
                    )
            
            @property
            def session(self):
                return self._session
            
            def add_cookie(self, **kwargs):
                if self._session:
                    from axiom.core.custom_types import Cookie
                    self._session.cookies.append(Cookie(**kwargs))
            
            def set_local_storage(self, key: str, value: str):
                if self._session:
                    self._session.local_storage[key] = value
            
            def set_session_storage(self, key: str, value: str):
                if self._session:
                    self._session.session_storage[key] = value
        
        return SessionContext(self, session_id)
    
    def cleanup(self):
        """Cleanup expired sessions."""
        return self.storage.cleanup_expired_sessions()
    
    def close(self):
        """Close session manager."""
        self.storage.close()


# Export public API
__all__ = [
    'SessionStorage',
    'SessionManager',
    'SessionStorageError',
    'SessionExpiredError',
    'create_session_storage',
]