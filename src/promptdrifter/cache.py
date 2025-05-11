import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_DB_PATH = Path("prompt_cache.db")  # Or perhaps in a user-specific cache dir
DEFAULT_TTL_SECONDS = 24 * 60 * 60  # 24 hours


class PromptCache:
    """Manages an SQLite cache for LLM prompts and responses."""

    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        default_ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ):
        self.db_path = db_path
        self.default_ttl_seconds = default_ttl_seconds
        self._conn: Optional[sqlite3.Connection] = (
            None  # For persistent in-memory connection
        )
        # self._ensure_db_and_table() # Call after connection logic is set
        # Initialize and ensure table with the first connection
        with self._get_connection() as conn:
            self._ensure_db_and_table_with_conn(conn)

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a connection to the SQLite database."""
        if self.db_path == Path(":memory:"):
            if self._conn is None or self._is_connection_closed(self._conn):
                # For in-memory, create one connection and reuse it.
                self._conn = sqlite3.connect(
                    self.db_path, check_same_thread=False, isolation_level=None
                )  # autocommit for :memory: simplicity
            return self._conn
        else:
            # For file-based DBs, connect each time.
            # isolation_level=None for autocommit can simplify some operations if not using explicit transactions widely.
            return sqlite3.connect(
                self.db_path, check_same_thread=False, isolation_level=None
            )

    def _is_connection_closed(self, conn: Optional[sqlite3.Connection]) -> bool:
        """Checks if the SQLite connection is closed."""
        if conn is None:
            return True
        try:
            # Attempt a simplepragma call. If it fails with ProgrammingError, connection is likely closed.
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
            conn.commit()  # Commit only if not in autocommit mode (memory often is by default or by our isolation_level)

    def _generate_fingerprint(
        self,
        prompt: str,
        adapter_name: str,
        model_id: Optional[str],
        options: Dict[str, Any],
    ) -> str:
        """Generates a unique fingerprint for a given prompt and its configuration."""
        # Normalize options by sorting keys to ensure consistent hash
        # Convert all option values to strings to avoid issues with unhashable types in complex dicts
        # A more robust solution might involve canonical JSON string representation for options.
        sorted_options_str = json.dumps(options, sort_keys=True, default=str)

        hasher = hashlib.sha256()
        hasher.update(prompt.encode("utf-8"))
        hasher.update(adapter_name.encode("utf-8"))
        if model_id:
            hasher.update(model_id.encode("utf-8"))
        hasher.update(sorted_options_str.encode("utf-8"))
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
                        # Cached data is corrupted, treat as miss and potentially remove
                        self.delete(fingerprint)  # Or just let it expire
                        return None
                else:
                    # Cache entry has expired, delete it
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
        fingerprint = self._generate_fingerprint(
            prompt, adapter_name, model_id, options
        )
        current_time = int(time.time())
        effective_ttl = (
            ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
        )

        try:
            response_data_str = json.dumps(response_data)
        except TypeError:
            # Handle cases where response_data might not be JSON serializable
            # print(f"Error serializing response data for caching: {e}")
            return  # Or raise an error

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
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM prompt_cache WHERE fingerprint = ?", (fingerprint,)
            )
            conn.commit()

    def purge_expired(self):
        """Removes all expired cache entries from the database."""
        current_time = int(time.time())
        with self._get_connection() as conn:  # Ensures conn is open for this operation
            cursor = conn.cursor()
            cursor.execute(
                "SELECT fingerprint, timestamp, ttl_seconds FROM prompt_cache"
            )
            expired_fingerprints = []
            for row in cursor.fetchall():
                fp, ts, ttl = row
                if (ts + ttl) <= current_time:
                    expired_fingerprints.append(fp)

            if expired_fingerprints:
                # Delete in batches or one by one
                # Using executemany for efficiency if supported and list of tuples for fingerprints
                placeholders = ",".join(["?"] * len(expired_fingerprints))
                cursor.execute(
                    f"DELETE FROM prompt_cache WHERE fingerprint IN ({placeholders})",
                    expired_fingerprints,
                )
                conn.commit()
                # print(f"Purged {len(expired_fingerprints)} expired cache entries.")
            # else:
            # print("No expired cache entries to purge.")

    def clear_all(self):
        """Clears all entries from the cache table."""
        with self._get_connection() as conn:  # Ensures conn is open
            cursor = conn.cursor()
            cursor.execute("DELETE FROM prompt_cache")
            conn.commit()

    def close(self):
        """Closes the persistent database connection, if any."""
        if self._conn:
            self._conn.close()
            self._conn = None


# Example usage (optional, for quick testing):
# if __name__ == '__main__':
#     cache = PromptCache(db_path=Path("test_cache.db"))
#     cache.clear_all() # Start fresh for example

#     prompt1 = "What is the capital of France?"
#     options1 = {"temp": 0.7}
#     response1_data = {"answer": "Paris", "confidence": 0.9}
#     adapter1 = "openai"
#     model1 = "gpt-3.5"

#     print(f"Cache GET (miss): {cache.get(prompt1, adapter1, model1, options1)}")
#     cache.put(prompt1, adapter1, model1, options1, response1_data, ttl_seconds=5) # Short TTL for testing
#     print(f"Cache GET (hit): {cache.get(prompt1, adapter1, model1, options1)}")

#     print("Waiting for cache to expire...")
#     time.sleep(6)
#     print(f"Cache GET (expired): {cache.get(prompt1, adapter1, model1, options1)}")
#     # At this point, the entry for prompt1 should have been deleted by the get method due to expiry.

#     # Test purge_expired (manually create an expired entry)
#     prompt2 = "Another prompt"
#     response2_data = {"answer": "Another answer"}
#     cache.put(prompt2, adapter1, model1, options1, response2_data, ttl_seconds=1) # Expired almost immediately
#     time.sleep(2) # Ensure it's expired
#     print(f"Cache size before purge: {len(cache._get_connection().execute('SELECT * FROM prompt_cache').fetchall())}")
#     cache.purge_expired()
#     print(f"Cache size after purge (should be 0 if only prompt2 was left and expired): {len(cache._get_connection().execute('SELECT * FROM prompt_cache').fetchall())}")

#     cache.clear_all()
#     Path("test_cache.db").unlink() # Clean up test db
