import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_DB_PATH = Path("prompt_cache.db")
DEFAULT_TTL_SECONDS = 24 * 60 * 60


class PromptCache:
    """Manages an SQLite cache for LLM prompts and responses."""

    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        default_ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ):
        self.db_path = db_path
        self.default_ttl_seconds = default_ttl_seconds
        self._conn: Optional[sqlite3.Connection] = None
        with self._get_connection() as conn:
            self._ensure_db_and_table_with_conn(conn)

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a connection to the SQLite database."""
        if self.db_path == Path(":memory:"):
            if self._conn is None or self._is_connection_closed(self._conn):
                self._conn = sqlite3.connect(
                    self.db_path, check_same_thread=False, isolation_level=None
                )
            return self._conn
        else:
            return sqlite3.connect(
                self.db_path, check_same_thread=False, isolation_level=None
            )

    def _is_connection_closed(self, conn: Optional[sqlite3.Connection]) -> bool:
        """Checks if the SQLite connection is closed."""
        if conn is None:
            return True
        try:
            conn.execute("PRAGMA user_version")
            return False
        except sqlite3.ProgrammingError:
            return True

    def _ensure_db_and_table_with_conn(self, conn: sqlite3.Connection):
        """Ensures the cache table exists using a provided connection."""
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompt_cache (
                fingerprint TEXT PRIMARY KEY,
                response_data TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                ttl_seconds INTEGER NOT NULL,
                adapter_name TEXT,
                model_used TEXT
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_fingerprint ON prompt_cache (fingerprint)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON prompt_cache (timestamp)"
        )
        if self.db_path != Path(":memory:") or conn.isolation_level is not None:
            conn.commit()

    def _generate_fingerprint(
        self,
        prompt: str,
        adapter_name: str,
        model_id: Optional[str],
        options: Dict[str, Any],
    ) -> str:
        """Generates a unique fingerprint for a given prompt and its configuration."""
        # Input validation for security
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if not isinstance(adapter_name, str):
            raise TypeError("adapter_name must be a string")
        if model_id is not None and not isinstance(model_id, str):
            raise TypeError("model_id must be a string or None")

        # Limit input sizes to prevent DoS attacks
        if len(prompt) > 1_000_000:  # 1MB limit
            raise ValueError("prompt exceeds maximum allowed size")
        if len(adapter_name) > 1000:
            raise ValueError("adapter_name exceeds maximum allowed size")
        if model_id and len(model_id) > 1000:
            raise ValueError("model_id exceeds maximum allowed size")

        try:
            if isinstance(options, frozenset):
                sorted_options_list = sorted(list(options))
                options_str_for_hash = json.dumps(sorted_options_list, sort_keys=False)
            else:
                options_str_for_hash = json.dumps(options, sort_keys=True, default=str)
        except (TypeError, ValueError) as e:
            raise ValueError(f"options cannot be serialized to JSON: {e}")

        hasher = hashlib.sha256()
        hasher.update(prompt.encode("utf-8"))
        hasher.update(adapter_name.encode("utf-8"))
        if model_id:
            hasher.update(model_id.encode("utf-8"))
        hasher.update(options_str_for_hash.encode("utf-8"))
        return hasher.hexdigest()

    def get(
        self,
        prompt: str,
        adapter_name: str,
        model_id: Optional[str],
        options: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Retrieves a cached response if available and not expired."""
        fingerprint = self._generate_fingerprint(
            prompt, adapter_name, model_id, options
        )
        current_time = int(time.time())

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT response_data, timestamp, ttl_seconds FROM prompt_cache WHERE fingerprint = ?",
                (fingerprint,),
            )
            row = cursor.fetchone()

            if row:
                response_data_str, timestamp, ttl_seconds = row
                if (timestamp + ttl_seconds) > current_time:
                    try:
                        return json.loads(response_data_str)
                    except json.JSONDecodeError:
                        self.delete(fingerprint)
                        return None
                else:
                    self.delete(fingerprint)
                    return None
        return None

    def put(
        self,
        prompt: str,
        adapter_name: str,
        model_id: Optional[str],
        options: Dict[str, Any],
        response_data: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ):
        """Stores a response in the cache."""
        # Input validation
        if not isinstance(response_data, dict):
            raise TypeError("response_data must be a dictionary")
        if ttl_seconds is not None and (not isinstance(ttl_seconds, int) or ttl_seconds <= 0):
            raise ValueError("ttl_seconds must be a positive integer")

        fingerprint = self._generate_fingerprint(
            prompt, adapter_name, model_id, options
        )
        current_time = int(time.time())
        effective_ttl = (
            ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
        )

        try:
            response_data_str = json.dumps(response_data)
            # Limit response size to prevent database bloat
            if len(response_data_str) > 10_000_000:  # 10MB limit
                raise ValueError("response_data exceeds maximum allowed size")
        except (TypeError, ValueError) as e:
            raise ValueError(f"Failed to serialize response_data: {e}")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO prompt_cache
                (fingerprint, response_data, timestamp, ttl_seconds, adapter_name, model_used)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    fingerprint,
                    response_data_str,
                    current_time,
                    effective_ttl,
                    adapter_name,
                    model_id,
                ),
            )
            conn.commit()

    def delete(self, fingerprint: str):
        """Deletes a specific cache entry by its fingerprint."""
        # Input validation
        if not isinstance(fingerprint, str):
            raise TypeError("fingerprint must be a string")
        if len(fingerprint) != 64:  # SHA256 hex digest length
            raise ValueError("fingerprint must be a valid SHA256 hash")
        if not all(c in '0123456789abcdef' for c in fingerprint.lower()):
            raise ValueError("fingerprint must contain only hexadecimal characters")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM prompt_cache WHERE fingerprint = ?", (fingerprint,)
            )
            conn.commit()

    def purge_expired(self):
        """Removes all expired cache entries from the database."""
        current_time = int(time.time())
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # More efficient: delete expired entries directly without fetching first
            cursor.execute(
                "DELETE FROM prompt_cache WHERE (timestamp + ttl_seconds) <= ?",
                (current_time,)
            )
            conn.commit()

    def clear_all(self):
        """Clears all entries from the cache table."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM prompt_cache")
            conn.commit()

    def get_stats(self) -> Dict[str, Any]:
        """Returns cache statistics for monitoring and debugging."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get total count
            cursor.execute("SELECT COUNT(*) FROM prompt_cache")
            total_count = cursor.fetchone()[0]

            # Get expired count
            current_time = int(time.time())
            cursor.execute(
                "SELECT COUNT(*) FROM prompt_cache WHERE (timestamp + ttl_seconds) <= ?",
                (current_time,)
            )
            expired_count = cursor.fetchone()[0]

            # Get size statistics
            cursor.execute("SELECT SUM(LENGTH(response_data)) FROM prompt_cache")
            total_size_bytes = cursor.fetchone()[0] or 0

            # Get adapter breakdown
            cursor.execute(
                "SELECT adapter_name, COUNT(*) FROM prompt_cache GROUP BY adapter_name"
            )
            adapter_breakdown = dict(cursor.fetchall())

            return {
                "total_entries": total_count,
                "active_entries": total_count - expired_count,
                "expired_entries": expired_count,
                "total_size_bytes": total_size_bytes,
                "adapter_breakdown": adapter_breakdown,
                "db_path": str(self.db_path),
            }

    def close(self):
        """Closes the persistent database connection, if any."""
        if self._conn:
            self._conn.close()
            self._conn = None
