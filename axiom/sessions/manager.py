# axiom/sessions/manager.py
"""
Session Persistence & State Management for axiom
Save and restore complete browser sessions (cookies, localStorage, sessionStorage) across runs
with automatic session rotation and cookie jar management.
"""

import json
import time
import sqlite3
import hashlib
import pickle
import zlib
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager
from urllib.parse import urlparse
import threading
from collections import defaultdict

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from axiom.core.custom_types import CookieType, LocalStorageType, SessionStorageType
from axiom.core.storage import BaseStorage, SQLiteStorage, StorageError
from axiom.core.utils._utils import generate_unique_id, sanitize_filename
from axiom.core.mixins import LoggerMixin


@dataclass
class SessionState:
    """Represents a complete browser session state"""
    session_id: str
    cookies: List[CookieType] = field(default_factory=list)
    local_storage: Dict[str, LocalStorageType] = field(default_factory=dict)
    session_storage: Dict[str, SessionStorageType] = field(default_factory=dict)
    user_agent: Optional[str] = None
    viewport: Optional[Dict[str, int]] = None
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session state to dictionary for serialization"""
        data = asdict(self)
        data['tags'] = list(self.tags)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionState':
        """Create session state from dictionary"""
        if 'tags' in data and isinstance(data['tags'], list):
            data['tags'] = set(data['tags'])
        return cls(**data)
    
    @property
    def age_seconds(self) -> float:
        """Get session age in seconds"""
        return time.time() - self.created_at
    
    @property
    def idle_seconds(self) -> float:
        """Get time since last use in seconds"""
        return time.time() - self.last_used
    
    def touch(self):
        """Update last used timestamp and increment use count"""
        self.last_used = time.time()
        self.use_count += 1
    
    def add_cookie(self, cookie: CookieType):
        """Add or update a cookie"""
        # Remove existing cookie with same name and domain
        self.cookies = [c for c in self.cookies 
                       if not (c.get('name') == cookie.get('name') and 
                              c.get('domain') == cookie.get('domain'))]
        self.cookies.append(cookie)
    
    def remove_cookie(self, name: str, domain: Optional[str] = None):
        """Remove a cookie by name and optionally domain"""
        if domain:
            self.cookies = [c for c in self.cookies 
                           if not (c.get('name') == name and c.get('domain') == domain)]
        else:
            self.cookies = [c for c in self.cookies if c.get('name') != name]
    
    def get_cookies_for_domain(self, domain: str) -> List[CookieType]:
        """Get all cookies for a specific domain"""
        return [c for c in self.cookies 
                if domain.endswith(c.get('domain', '')) or 
                c.get('domain', '').endswith(domain)]
    
    def set_local_storage(self, origin: str, key: str, value: str):
        """Set localStorage item for an origin"""
        if origin not in self.local_storage:
            self.local_storage[origin] = {}
        self.local_storage[origin][key] = value
    
    def get_local_storage(self, origin: str, key: str) -> Optional[str]:
        """Get localStorage item for an origin"""
        return self.local_storage.get(origin, {}).get(key)
    
    def set_session_storage(self, origin: str, key: str, value: str):
        """Set sessionStorage item for an origin"""
        if origin not in self.session_storage:
            self.session_storage[origin] = {}
        self.session_storage[origin][key] = value
    
    def get_session_storage(self, origin: str, key: str) -> Optional[str]:
        """Get sessionStorage item for an origin"""
        return self.session_storage.get(origin, {}).get(key)


class SessionStorageError(Exception):
    """Exception raised for session storage errors"""
    pass


class SessionStorage(BaseStorage):
    """Abstract base class for session storage backends"""
    
    def save_session(self, session: SessionState) -> bool:
        """Save a session state"""
        raise NotImplementedError
    
    def load_session(self, session_id: str) -> Optional[SessionState]:
        """Load a session state by ID"""
        raise NotImplementedError
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session by ID"""
        raise NotImplementedError
    
    def list_sessions(self, 
                     tags: Optional[Set[str]] = None,
                     max_age_seconds: Optional[float] = None,
                     min_uses: Optional[int] = None,
                     limit: Optional[int] = None) -> List[SessionState]:
        """List sessions with optional filtering"""
        raise NotImplementedError
    
    def cleanup_expired(self, max_age_seconds: float) -> int:
        """Remove sessions older than max_age_seconds, returns count removed"""
        raise NotImplementedError
    
    def count_sessions(self) -> int:
        """Count total sessions"""
        raise NotImplementedError
    
    def export_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Export session to dictionary"""
        raise NotImplementedError
    
    def import_session(self, session_data: Dict[str, Any]) -> Optional[str]:
        """Import session from dictionary, returns session_id"""
        raise NotImplementedError


class SQLiteSessionStorage(SessionStorage, LoggerMixin):
    """SQLite-based session storage"""
    
    def __init__(self, db_path: Union[str, Path] = "sessions.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()
        self.logger.info(f"Initialized SQLite session storage at {self.db_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection"""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0
            )
            self._local.connection.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            self._local.connection.execute("PRAGMA journal_mode=WAL")
            self._local.connection.execute("PRAGMA synchronous=NORMAL")
        return self._local.connection
    
    def _init_db(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    data BLOB NOT NULL,
                    compressed_data BLOB,
                    created_at REAL NOT NULL,
                    last_used REAL NOT NULL,
                    use_count INTEGER DEFAULT 0,
                    user_agent TEXT,
                    tags TEXT DEFAULT '[]',
                    checksum TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_created 
                ON sessions(created_at)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_last_used 
                ON sessions(last_used)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_use_count 
                ON sessions(use_count)
            """)
            
            # Tags are stored as JSON array, create index for common queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_tags 
                ON sessions(tags)
            """)
            
            conn.commit()
    
    def _compute_checksum(self, data: bytes) -> str:
        """Compute checksum for data integrity verification"""
        return hashlib.sha256(data).hexdigest()
    
    def _compress_data(self, data: bytes) -> bytes:
        """Compress data using zlib"""
        return zlib.compress(data, level=6)
    
    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress zlib-compressed data"""
        return zlib.decompress(data)
    
    def save_session(self, session: SessionState) -> bool:
        """Save a session state to SQLite"""
        try:
            session_dict = session.to_dict()
            session_data = pickle.dumps(session_dict)
            compressed_data = self._compress_data(session_data)
            checksum = self._compute_checksum(session_data)
            
            tags_json = json.dumps(list(session.tags))
            
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO sessions 
                    (session_id, data, compressed_data, created_at, last_used, 
                     use_count, user_agent, tags, checksum)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session.session_id,
                    session_data,
                    compressed_data,
                    session.created_at,
                    session.last_used,
                    session.use_count,
                    session.user_agent,
                    tags_json,
                    checksum
                ))
                conn.commit()
            
            self.logger.debug(f"Saved session {session.session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save session {session.session_id}: {e}")
            raise SessionStorageError(f"Failed to save session: {e}")
    
    def load_session(self, session_id: str) -> Optional[SessionState]:
        """Load a session state from SQLite"""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT data, compressed_data, checksum 
                    FROM sessions 
                    WHERE session_id = ?
                """, (session_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                # Try to use compressed data if available, otherwise use uncompressed
                if row['compressed_data']:
                    session_data = self._decompress_data(row['compressed_data'])
                else:
                    session_data = row['data']
                
                # Verify data integrity
                computed_checksum = self._compute_checksum(session_data)
                if computed_checksum != row['checksum']:
                    self.logger.warning(f"Checksum mismatch for session {session_id}")
                    # Try to use uncompressed data as fallback
                    if row['data']:
                        session_data = row['data']
                        computed_checksum = self._compute_checksum(session_data)
                        if computed_checksum != row['checksum']:
                            raise SessionStorageError(f"Data corruption detected for session {session_id}")
                
                session_dict = pickle.loads(session_data)
                session = SessionState.from_dict(session_dict)
                
                # Update last used timestamp
                session.touch()
                self.save_session(session)  # Update the timestamp in DB
                
                self.logger.debug(f"Loaded session {session_id}")
                return session
                
        except Exception as e:
            self.logger.error(f"Failed to load session {session_id}: {e}")
            raise SessionStorageError(f"Failed to load session: {e}")
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session from SQLite"""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM sessions WHERE session_id = ?",
                    (session_id,)
                )
                conn.commit()
                deleted = cursor.rowcount > 0
                
                if deleted:
                    self.logger.debug(f"Deleted session {session_id}")
                
                return deleted
                
        except Exception as e:
            self.logger.error(f"Failed to delete session {session_id}: {e}")
            raise SessionStorageError(f"Failed to delete session: {e}")
    
    def list_sessions(self,
                     tags: Optional[Set[str]] = None,
                     max_age_seconds: Optional[float] = None,
                     min_uses: Optional[int] = None,
                     limit: Optional[int] = None) -> List[SessionState]:
        """List sessions with optional filtering"""
        try:
            query = "SELECT session_id FROM sessions WHERE 1=1"
            params = []
            
            if tags:
                # SQLite doesn't have native array support, so we use JSON functions
                tag_conditions = []
                for tag in tags:
                    tag_conditions.append("tags LIKE ?")
                    params.append(f'%"{tag}"%')
                query += " AND (" + " OR ".join(tag_conditions) + ")"
            
            if max_age_seconds is not None:
                min_created = time.time() - max_age_seconds
                query += " AND created_at >= ?"
                params.append(min_created)
            
            if min_uses is not None:
                query += " AND use_count >= ?"
                params.append(min_uses)
            
            query += " ORDER BY last_used DESC"
            
            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)
            
            with self._get_connection() as conn:
                cursor = conn.execute(query, params)
                session_ids = [row['session_id'] for row in cursor.fetchall()]
            
            sessions = []
            for session_id in session_ids:
                session = self.load_session(session_id)
                if session:
                    sessions.append(session)
            
            return sessions
            
        except Exception as e:
            self.logger.error(f"Failed to list sessions: {e}")
            raise SessionStorageError(f"Failed to list sessions: {e}")
    
    def cleanup_expired(self, max_age_seconds: float) -> int:
        """Remove sessions older than max_age_seconds"""
        try:
            min_created = time.time() - max_age_seconds
            
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM sessions WHERE created_at < ?",
                    (min_created,)
                )
                conn.commit()
                deleted_count = cursor.rowcount
                
                if deleted_count > 0:
                    self.logger.info(f"Cleaned up {deleted_count} expired sessions")
                
                return deleted_count
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired sessions: {e}")
            raise SessionStorageError(f"Failed to cleanup expired sessions: {e}")
    
    def count_sessions(self) -> int:
        """Count total sessions"""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT COUNT(*) as count FROM sessions")
                row = cursor.fetchone()
                return row['count'] if row else 0
                
        except Exception as e:
            self.logger.error(f"Failed to count sessions: {e}")
            raise SessionStorageError(f"Failed to count sessions: {e}")
    
    def export_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Export session to dictionary"""
        session = self.load_session(session_id)
        if not session:
            return None
        
        return {
            'version': '1.0',
            'exported_at': time.time(),
            'session': session.to_dict()
        }
    
    def import_session(self, session_data: Dict[str, Any]) -> Optional[str]:
        """Import session from dictionary"""
        try:
            if 'session' not in session_data:
                raise SessionStorageError("Invalid session data format")
            
            session_dict = session_data['session']
            
            # Generate new session ID to avoid conflicts
            if 'session_id' in session_dict:
                original_id = session_dict['session_id']
                session_dict['session_id'] = generate_unique_id()
                self.logger.info(f"Imported session with new ID {session_dict['session_id']} (original: {original_id})")
            
            session = SessionState.from_dict(session_dict)
            self.save_session(session)
            
            return session.session_id
            
        except Exception as e:
            self.logger.error(f"Failed to import session: {e}")
            raise SessionStorageError(f"Failed to import session: {e}")
    
    def close(self):
        """Close database connection"""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            delattr(self._local, 'connection')


class RedisSessionStorage(SessionStorage, LoggerMixin):
    """Redis-based session storage (requires redis-py)"""
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379/0",
                 prefix: str = "axiom:session:",
                 expire_seconds: int = 86400 * 30):  # 30 days default
        if not REDIS_AVAILABLE:
            raise ImportError("Redis storage requires redis-py package. Install with: pip install redis")
        
        self.redis_url = redis_url
        self.prefix = prefix
        self.expire_seconds = expire_seconds
        self._redis = redis.Redis.from_url(redis_url, decode_responses=False)
        self.logger.info(f"Initialized Redis session storage with prefix {prefix}")
    
    def _get_key(self, session_id: str) -> str:
        """Get Redis key for session ID"""
        return f"{self.prefix}{session_id}"
    
    def _get_index_key(self) -> str:
        """Get Redis key for session index"""
        return f"{self.prefix}index"
    
    def save_session(self, session: SessionState) -> bool:
        """Save a session state to Redis"""
        try:
            session_dict = session.to_dict()
            session_data = pickle.dumps(session_dict)
            
            key = self._get_key(session.session_id)
            index_key = self._get_index_key()
            
            # Use pipeline for atomic operations
            pipe = self._redis.pipeline()
            pipe.set(key, session_data, ex=self.expire_seconds)
            pipe.zadd(index_key, {session.session_id: session.last_used})
            pipe.execute()
            
            self.logger.debug(f"Saved session {session.session_id} to Redis")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save session to Redis: {e}")
            raise SessionStorageError(f"Failed to save session to Redis: {e}")
    
    def load_session(self, session_id: str) -> Optional[SessionState]:
        """Load a session state from Redis"""
        try:
            key = self._get_key(session_id)
            session_data = self._redis.get(key)
            
            if not session_data:
                return None
            
            session_dict = pickle.loads(session_data)
            session = SessionState.from_dict(session_dict)
            
            # Update last used timestamp
            session.touch()
            self.save_session(session)
            
            self.logger.debug(f"Loaded session {session_id} from Redis")
            return session
            
        except Exception as e:
            self.logger.error(f"Failed to load session from Redis: {e}")
            raise SessionStorageError(f"Failed to load session from Redis: {e}")
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session from Redis"""
        try:
            key = self._get_key(session_id)
            index_key = self._get_index_key()
            
            pipe = self._redis.pipeline()
            pipe.delete(key)
            pipe.zrem(index_key, session_id)
            results = pipe.execute()
            
            deleted = results[0] > 0
            if deleted:
                self.logger.debug(f"Deleted session {session_id} from Redis")
            
            return deleted
            
        except Exception as e:
            self.logger.error(f"Failed to delete session from Redis: {e}")
            raise SessionStorageError(f"Failed to delete session from Redis: {e}")
    
    def list_sessions(self,
                     tags: Optional[Set[str]] = None,
                     max_age_seconds: Optional[float] = None,
                     min_uses: Optional[int] = None,
                     limit: Optional[int] = None) -> List[SessionState]:
        """List sessions with optional filtering"""
        try:
            index_key = self._get_index_key()
            
            # Get all session IDs sorted by last_used (descending)
            session_ids = self._redis.zrevrange(index_key, 0, -1)
            
            sessions = []
            for session_id_bytes in session_ids:
                session_id = session_id_bytes.decode('utf-8')
                session = self.load_session(session_id)
                
                if not session:
                    # Clean up orphaned index entry
                    self._redis.zrem(index_key, session_id)
                    continue
                
                # Apply filters
                if tags and not session.tags.intersection(tags):
                    continue
                
                if max_age_seconds and session.age_seconds > max_age_seconds:
                    continue
                
                if min_uses and session.use_count < min_uses:
                    continue
                
                sessions.append(session)
                
                if limit and len(sessions) >= limit:
                    break
            
            return sessions
            
        except Exception as e:
            self.logger.error(f"Failed to list sessions from Redis: {e}")
            raise SessionStorageError(f"Failed to list sessions from Redis: {e}")
    
    def cleanup_expired(self, max_age_seconds: float) -> int:
        """Remove sessions older than max_age_seconds"""
        # Redis handles expiration automatically via TTL
        # But we need to clean up the index
        try:
            index_key = self._get_index_key()
            min_timestamp = time.time() - max_age_seconds
            
            # Get sessions older than max_age_seconds
            old_sessions = self._redis.zrangebyscore(index_key, 0, min_timestamp)
            
            if not old_sessions:
                return 0
            
            pipe = self._redis.pipeline()
            for session_id_bytes in old_sessions:
                session_id = session_id_bytes.decode('utf-8')
                key = self._get_key(session_id)
                pipe.delete(key)
                pipe.zrem(index_key, session_id)
            
            pipe.execute()
            
            deleted_count = len(old_sessions)
            if deleted_count > 0:
                self.logger.info(f"Cleaned up {deleted_count} expired sessions from Redis")
            
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired sessions in Redis: {e}")
            raise SessionStorageError(f"Failed to cleanup expired sessions in Redis: {e}")
    
    def count_sessions(self) -> int:
        """Count total sessions"""
        try:
            index_key = self._get_index_key()
            return self._redis.zcard(index_key)
            
        except Exception as e:
            self.logger.error(f"Failed to count sessions in Redis: {e}")
            raise SessionStorageError(f"Failed to count sessions in Redis: {e}")
    
    def export_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Export session to dictionary"""
        session = self.load_session(session_id)
        if not session:
            return None
        
        return {
            'version': '1.0',
            'exported_at': time.time(),
            'session': session.to_dict()
        }
    
    def import_session(self, session_data: Dict[str, Any]) -> Optional[str]:
        """Import session from dictionary"""
        try:
            if 'session' not in session_data:
                raise SessionStorageError("Invalid session data format")
            
            session_dict = session_data['session']
            
            # Generate new session ID to avoid conflicts
            if 'session_id' in session_dict:
                original_id = session_dict['session_id']
                session_dict['session_id'] = generate_unique_id()
                self.logger.info(f"Imported session with new ID {session_dict['session_id']} (original: {original_id})")
            
            session = SessionState.from_dict(session_dict)
            self.save_session(session)
            
            return session.session_id
            
        except Exception as e:
            self.logger.error(f"Failed to import session to Redis: {e}")
            raise SessionStorageError(f"Failed to import session to Redis: {e}")
    
    def close(self):
        """Close Redis connection"""
        self._redis.close()


class HarSessionHandler:
    """Handles import/export of sessions to/from HAR files"""
    
    @staticmethod
    def export_to_har(session: SessionState, 
                     har_path: Union[str, Path],
                     include_storage: bool = False) -> bool:
        """Export session to HAR file
        
        Args:
            session: Session state to export
            har_path: Path to save HAR file
            include_storage: Whether to include localStorage/sessionStorage
                           (Note: HAR spec doesn't support this, we use custom field)
        
        Returns:
            True if successful
        """
        try:
            har_path = Path(har_path)
            har_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Build HAR structure
            har = {
                "log": {
                    "version": "1.2",
                    "creator": {
                        "name": "axiom Session Manager",
                        "version": "1.0.0"
                    },
                    "pages": [],
                    "entries": [],
                    "comment": f"axiom session export - {session.session_id}"
                }
            }
            
            # Add cookies to HAR
            if session.cookies:
                har["log"]["cookies"] = session.cookies
            
            # Add custom storage if requested
            if include_storage:
                har["log"]["_axiomStorage"] = {
                    "localStorage": session.local_storage,
                    "sessionStorage": session.session_storage,
                    "metadata": session.metadata
                }
            
            # Write HAR file
            with open(har_path, 'w', encoding='utf-8') as f:
                json.dump(har, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            raise SessionStorageError(f"Failed to export session to HAR: {e}")
    
    @staticmethod
    def import_from_har(har_path: Union[str, Path],
                       session_id: Optional[str] = None) -> SessionState:
        """Import session from HAR file
        
        Args:
            har_path: Path to HAR file
            session_id: Optional session ID (generated if not provided)
        
        Returns:
            SessionState imported from HAR
        """
        try:
            har_path = Path(har_path)
            
            if not har_path.exists():
                raise FileNotFoundError(f"HAR file not found: {har_path}")
            
            with open(har_path, 'r', encoding='utf-8') as f:
                har = json.load(f)
            
            # Validate HAR structure
            if "log" not in har:
                raise ValueError("Invalid HAR file: missing 'log' key")
            
            log = har["log"]
            
            # Extract cookies
            cookies = log.get("cookies", [])
            
            # Extract custom storage if present
            local_storage = {}
            session_storage = {}
            metadata = {}
            
            if "_axiomStorage" in log:
                axiom_data = log["_axiomStorage"]
                local_storage = axiom_data.get("localStorage", {})
                session_storage = axiom_data.get("sessionStorage", {})
                metadata = axiom_data.get("metadata", {})
            
            # Create session
            session = SessionState(
                session_id=session_id or generate_unique_id(),
                cookies=cookies,
                local_storage=local_storage,
                session_storage=session_storage,
                metadata=metadata
            )
            
            return session
            
        except Exception as e:
            raise SessionStorageError(f"Failed to import session from HAR: {e}")


class SessionManager(LoggerMixin):
    """Main session manager with automatic rotation and cookie jar management"""
    
    def __init__(self,
                 storage: Optional[SessionStorage] = None,
                 max_sessions: int = 1000,
                 max_session_age_days: int = 30,
                 rotation_strategy: str = "age",  # "age", "uses", "both"
                 max_uses_per_session: int = 1000,
                 auto_cleanup_interval: int = 3600):  # 1 hour
        """
        Initialize session manager
        
        Args:
            storage: Storage backend (default: SQLite)
            max_sessions: Maximum number of sessions to keep
            max_session_age_days: Maximum age of sessions in days
            rotation_strategy: When to rotate sessions
            max_uses_per_session: Maximum uses before rotation
            auto_cleanup_interval: Interval for automatic cleanup in seconds
        """
        if storage is None:
            storage = SQLiteSessionStorage()
        
        self.storage = storage
        self.max_sessions = max_sessions
        self.max_session_age_seconds = max_session_age_days * 86400
        self.rotation_strategy = rotation_strategy
        self.max_uses_per_session = max_uses_per_session
        self.auto_cleanup_interval = auto_cleanup_interval
        
        self._last_cleanup = time.time()
        self._rotation_lock = threading.RLock()
        
        self.logger.info(f"Session manager initialized with {type(storage).__name__}")
    
    def create_session(self,
                      cookies: Optional[List[CookieType]] = None,
                      local_storage: Optional[Dict[str, LocalStorageType]] = None,
                      session_storage: Optional[Dict[str, SessionStorageType]] = None,
                      user_agent: Optional[str] = None,
                      viewport: Optional[Dict[str, int]] = None,
                      tags: Optional[Set[str]] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> SessionState:
        """Create a new session
        
        Args:
            cookies: Initial cookies
            local_storage: Initial localStorage
            session_storage: Initial sessionStorage
            user_agent: User agent string
            viewport: Viewport dimensions
            tags: Tags for categorization
            metadata: Additional metadata
        
        Returns:
            New session state
        """
        session = SessionState(
            session_id=generate_unique_id(),
            cookies=cookies or [],
            local_storage=local_storage or {},
            session_storage=session_storage or {},
            user_agent=user_agent,
            viewport=viewport,
            tags=tags or set(),
            metadata=metadata or {}
        )
        
        self.storage.save_session(session)
        self.logger.info(f"Created new session {session.session_id}")
        
        # Trigger cleanup if needed
        self._maybe_cleanup()
        
        return session
    
    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get a session by ID
        
        Args:
            session_id: Session ID
        
        Returns:
            Session state or None if not found
        """
        session = self.storage.load_session(session_id)
        
        if session:
            # Check if session needs rotation
            if self._should_rotate(session):
                self.logger.info(f"Session {session_id} needs rotation")
                return self._rotate_session(session)
        
        return session
    
    def get_or_create_session(self,
                            session_id: Optional[str] = None,
                            tags: Optional[Set[str]] = None,
                            **kwargs) -> SessionState:
        """Get existing session or create new one
        
        Args:
            session_id: Optional session ID to retrieve
            tags: Tags for new session
            **kwargs: Arguments for create_session if creating new
        
        Returns:
            Session state
        """
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session
        
        # Create new session
        return self.create_session(tags=tags, **kwargs)
    
    def update_session(self, session: SessionState) -> bool:
        """Update an existing session
        
        Args:
            session: Session state to update
        
        Returns:
            True if successful
        """
        session.touch()
        return self.storage.save_session(session)
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session
        
        Args:
            session_id: Session ID to delete
        
        Returns:
            True if deleted
        """
        return self.storage.delete_session(session_id)
    
    def list_sessions(self,
                     tags: Optional[Set[str]] = None,
                     max_age_seconds: Optional[float] = None,
                     min_uses: Optional[int] = None,
                     limit: Optional[int] = None) -> List[SessionState]:
        """List sessions with filtering
        
        Args:
            tags: Filter by tags
            max_age_seconds: Maximum age in seconds
            min_uses: Minimum use count
            limit: Maximum number to return
        
        Returns:
            List of session states
        """
        return self.storage.list_sessions(
            tags=tags,
            max_age_seconds=max_age_seconds,
            min_uses=min_uses,
            limit=limit
        )
    
    def get_session_for_domain(self, domain: str, tags: Optional[Set[str]] = None) -> Optional[SessionState]:
        """Get a session that has cookies for a specific domain
        
        Args:
            domain: Domain to check for cookies
            tags: Optional tags to filter by
        
        Returns:
            Session with cookies for domain, or None
        """
        sessions = self.list_sessions(tags=tags)
        
        for session in sessions:
            if session.get_cookies_for_domain(domain):
                return session
        
        return None
    
    def rotate_session(self, session_id: str) -> Optional[SessionState]:
        """Manually rotate a session
        
        Args:
            session_id: Session ID to rotate
        
        Returns:
            New session state, or None if original not found
        """
        with self._rotation_lock:
            session = self.storage.load_session(session_id)
            if not session:
                return None
            
            return self._rotate_session(session)
    
    def _rotate_session(self, old_session: SessionState) -> SessionState:
        """Rotate a session (internal implementation)
        
        Args:
            old_session: Session to rotate
        
        Returns:
            New session state
        """
        # Create new session with same properties
        new_session = SessionState(
            session_id=generate_unique_id(),
            cookies=old_session.cookies.copy(),
            local_storage={k: v.copy() for k, v in old_session.local_storage.items()},
            session_storage={k: v.copy() for k, v in old_session.session_storage.items()},
            user_agent=old_session.user_agent,
            viewport=old_session.viewport,
            tags=old_session.tags.copy(),
            metadata={
                **old_session.metadata,
                'rotated_from': old_session.session_id,
                'rotated_at': time.time()
            }
        )
        
        # Save new session
        self.storage.save_session(new_session)
        
        # Delete old session
        self.storage.delete_session(old_session.session_id)
        
        self.logger.info(f"Rotated session {old_session.session_id} -> {new_session.session_id}")
        
        return new_session
    
    def _should_rotate(self, session: SessionState) -> bool:
        """Check if a session should be rotated
        
        Args:
            session: Session to check
        
        Returns:
            True if session should be rotated
        """
        if self.rotation_strategy == "age":
            return session.age_seconds > self.max_session_age_seconds
        elif self.rotation_strategy == "uses":
            return session.use_count >= self.max_uses_per_session
        elif self.rotation_strategy == "both":
            return (session.age_seconds > self.max_session_age_seconds or
                    session.use_count >= self.max_uses_per_session)
        return False
    
    def _maybe_cleanup(self):
        """Perform cleanup if interval has passed"""
        current_time = time.time()
        if current_time - self._last_cleanup > self.auto_cleanup_interval:
            self.cleanup()
            self._last_cleanup = current_time
    
    def cleanup(self) -> Dict[str, int]:
        """Perform cleanup of old sessions
        
        Returns:
            Dictionary with cleanup statistics
        """
        stats = {
            'expired_removed': 0,
            'excess_removed': 0,
            'total_before': 0,
            'total_after': 0
        }
        
        try:
            stats['total_before'] = self.storage.count_sessions()
            
            # Remove expired sessions
            stats['expired_removed'] = self.storage.cleanup_expired(self.max_session_age_seconds)
            
            # Remove excess sessions if over limit
            current_count = self.storage.count_sessions()
            if current_count > self.max_sessions:
                # Get oldest sessions
                sessions = self.storage.list_sessions(limit=current_count - self.max_sessions)
                for session in sessions:
                    self.storage.delete_session(session.session_id)
                    stats['excess_removed'] += 1
            
            stats['total_after'] = self.storage.count_sessions()
            
            if stats['expired_removed'] > 0 or stats['excess_removed'] > 0:
                self.logger.info(
                    f"Cleanup completed: removed {stats['expired_removed']} expired, "
                    f"{stats['excess_removed']} excess sessions"
                )
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            raise
    
    def export_session(self,
                      session_id: str,
                      export_path: Union[str, Path],
                      format: str = "json") -> bool:
        """Export a session to file
        
        Args:
            session_id: Session ID to export
            export_path: Path to export to
            format: Export format ("json" or "har")
        
        Returns:
            True if successful
        """
        export_path = Path(export_path)
        
        if format == "har":
            session = self.storage.load_session(session_id)
            if not session:
                return False
            return HarSessionHandler.export_to_har(session, export_path)
        elif format == "json":
            session_data = self.storage.export_session(session_id)
            if not session_data:
                return False
            
            export_path.parent.mkdir(parents=True, exist_ok=True)
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            return True
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def import_session(self,
                      import_path: Union[str, Path],
                      format: str = "json") -> Optional[str]:
        """Import a session from file
        
        Args:
            import_path: Path to import from
            format: Import format ("json" or "har")
        
        Returns:
            Imported session ID, or None if failed
        """
        import_path = Path(import_path)
        
        if not import_path.exists():
            raise FileNotFoundError(f"Import file not found: {import_path}")
        
        if format == "har":
            session = HarSessionHandler.import_from_har(import_path)
            self.storage.save_session(session)
            return session.session_id
        elif format == "json":
            with open(import_path, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            return self.storage.import_session(session_data)
        else:
            raise ValueError(f"Unsupported import format: {format}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session manager statistics
        
        Returns:
            Dictionary with statistics
        """
        sessions = self.list_sessions()
        
        total_sessions = len(sessions)
        total_cookies = sum(len(s.cookies) for s in sessions)
        total_local_storage = sum(len(s.local_storage) for s in sessions)
        total_session_storage = sum(len(s.session_storage) for s in sessions)
        
        # Calculate age distribution
        ages = [s.age_seconds for s in sessions]
        avg_age = sum(ages) / len(ages) if ages else 0
        
        # Calculate use distribution
        uses = [s.use_count for s in sessions]
        avg_uses = sum(uses) / len(uses) if uses else 0
        
        # Tag distribution
        tag_counts = defaultdict(int)
        for session in sessions:
            for tag in session.tags:
                tag_counts[tag] += 1
        
        return {
            'total_sessions': total_sessions,
            'total_cookies': total_cookies,
            'total_local_storage_entries': total_local_storage,
            'total_session_storage_entries': total_session_storage,
            'average_age_seconds': avg_age,
            'average_use_count': avg_uses,
            'tag_distribution': dict(tag_counts),
            'storage_type': type(self.storage).__name__,
            'max_sessions': self.max_sessions,
            'max_session_age_days': self.max_session_age_seconds / 86400,
            'rotation_strategy': self.rotation_strategy
        }
    
    def close(self):
        """Close session manager and underlying storage"""
        if hasattr(self.storage, 'close'):
            self.storage.close()
        self.logger.info("Session manager closed")


# Factory functions for common configurations
def create_sqlite_session_manager(db_path: Union[str, Path] = "sessions.db", **kwargs) -> SessionManager:
    """Create session manager with SQLite storage"""
    storage = SQLiteSessionStorage(db_path)
    return SessionManager(storage=storage, **kwargs)


def create_redis_session_manager(redis_url: str = "redis://localhost:6379/0", **kwargs) -> SessionManager:
    """Create session manager with Redis storage"""
    storage = RedisSessionStorage(redis_url)
    return SessionManager(storage=storage, **kwargs)


# Context manager for session management
@contextmanager
def managed_session(manager: SessionManager, 
                   session_id: Optional[str] = None,
                   **kwargs):
    """Context manager for automatic session management
    
    Args:
        manager: Session manager instance
        session_id: Optional existing session ID
        **kwargs: Arguments for get_or_create_session
    
    Yields:
        Session state
    """
    session = manager.get_or_create_session(session_id=session_id, **kwargs)
    try:
        yield session
    finally:
        manager.update_session(session)


# Integration with existing axiom components
class SessionAwareFetcher:
    """Mixin to add session awareness to fetchers"""
    
    def __init__(self, session_manager: Optional[SessionManager] = None, **kwargs):
        self.session_manager = session_manager or create_sqlite_session_manager()
        self._current_session_id = None
        super().__init__(**kwargs)
    
    def set_session(self, session_id: Optional[str] = None, **kwargs) -> SessionState:
        """Set current session"""
        session = self.session_manager.get_or_create_session(session_id, **kwargs)
        self._current_session_id = session.session_id
        return session
    
    def get_current_session(self) -> Optional[SessionState]:
        """Get current session"""
        if self._current_session_id:
            return self.session_manager.get_session(self._current_session_id)
        return None
    
    def clear_session(self):
        """Clear current session"""
        self._current_session_id = None
    
    def save_cookies_to_session(self, cookies: List[CookieType]):
        """Save cookies to current session"""
        session = self.get_current_session()
        if session:
            for cookie in cookies:
                session.add_cookie(cookie)
            self.session_manager.update_session(session)
    
    def load_cookies_from_session(self, domain: Optional[str] = None) -> List[CookieType]:
        """Load cookies from current session"""
        session = self.get_current_session()
        if not session:
            return []
        
        if domain:
            return session.get_cookies_for_domain(domain)
        return session.cookies


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    manager = create_sqlite_session_manager("test_sessions.db")
    
    # Create a session
    session = manager.create_session(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        tags={"test", "example"}
    )
    
    # Add some cookies
    session.add_cookie({
        "name": "session_id",
        "value": "abc123",
        "domain": ".example.com",
        "path": "/",
        "expires": time.time() + 3600
    })
    
    # Save session
    manager.update_session(session)
    
    # Export to HAR
    manager.export_session(session.session_id, "session_export.har", format="har")
    
    # Get statistics
    stats = manager.get_stats()
    print(f"Sessions: {stats['total_sessions']}")
    print(f"Cookies: {stats['total_cookies']}")
    
    # Cleanup
    manager.cleanup()
    manager.close()