import json
import time
from pathlib import Path
from typing import Generator

import pytest

from promptdrifter.cache import PromptCache

# Use a fixed temporary path for the test database to simplify inspection if needed,
# but ensure it's cleaned up. Or use :memory: for true isolation per test if preferred.
# Using a named temp file can be good if tests need to run in parallel & :memory: gets tricky.
TEST_DB_NAME = "test_prompt_cache.db"


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Provides a temporary path for the test SQLite database file."""
    return tmp_path / TEST_DB_NAME


@pytest.fixture
def cache(temp_db_path: Path) -> Generator[PromptCache, None, None]:
    """Provides a PromptCache instance using a temporary database file, clears it afterwards."""
    # Ensure a clean state for each test that uses this fixture
    if temp_db_path.exists():
        temp_db_path.unlink()

    pc = PromptCache(
        db_path=temp_db_path, default_ttl_seconds=3600
    )  # Default 1 hour TTL for tests unless overridden
    yield pc
    # Teardown: remove the test database file after the test run
    if temp_db_path.exists():
        try:
            # Attempt to close any connections if an explicit close method were available on PromptCache
            # pc.close_db_connection() # If implemented
            pass
        finally:
            temp_db_path.unlink()


@pytest.fixture
def cache_in_memory() -> Generator[PromptCache, None, None]:
    """Provides a PromptCache instance using an in-memory SQLite database."""
    pc = PromptCache(db_path=Path(":memory:"), default_ttl_seconds=3600)
    yield pc
    pc.close()  # Ensure the in-memory connection is closed after the test


# Use the in-memory cache for most tests for speed and isolation
@pytest.fixture
def sut(cache_in_memory: PromptCache) -> PromptCache:
    """System Under Test - an alias for the in-memory cache fixture."""
    return cache_in_memory


# --- Test Cases ---


def test_cache_initialization(sut: PromptCache):
    """Test that the database and table are created on initialization."""
    # The _ensure_db_and_table is called in __init__.
    # Check if the table exists by trying to query it.
    try:
        with sut._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT count(*) FROM prompt_cache")
            count = cursor.fetchone()[0]
            assert count == 0
    except Exception as e:
        pytest.fail(f"Database table not created or query failed: {e}")


def test_generate_fingerprint_consistency(sut: PromptCache):
    prompt = "Hello world"
    adapter = "test_adapter"
    model = "test_model"
    options = {"temp": 0.5, "max_tokens": 100}
    fp1 = sut._generate_fingerprint(prompt, adapter, model, options)
    fp2 = sut._generate_fingerprint(prompt, adapter, model, options)
    assert fp1 == fp2


def test_generate_fingerprint_difference(sut: PromptCache):
    prompt = "Hello world"
    adapter = "test_adapter"
    model = "test_model"
    options1 = {"temp": 0.5, "max_tokens": 100}
    options2 = {"temp": 0.6, "max_tokens": 100}  # Different temp
    fp1 = sut._generate_fingerprint(prompt, adapter, model, options1)
    fp2 = sut._generate_fingerprint(prompt, adapter, model, options2)
    assert fp1 != fp2

    fp3 = sut._generate_fingerprint("Different prompt", adapter, model, options1)
    assert fp1 != fp3


def test_put_and_get_item(sut: PromptCache):
    prompt = "What is the capital of Testland?"
    adapter = "test_adapter"
    model = "test_model_pg"
    options = {"setting": "value"}
    response_data = {"answer": "Testville", "details": [1, 2, 3]}

    # Cache miss initially
    assert sut.get(prompt, adapter, model, options) is None

    # Put item
    sut.put(prompt, adapter, model, options, response_data)

    # Cache hit
    cached_item = sut.get(prompt, adapter, model, options)
    assert cached_item is not None
    assert cached_item == response_data


def test_item_expiry(sut: PromptCache):
    prompt = "Short-lived prompt"
    adapter = "expiry_adapter"
    model = "expiry_model"
    options = {"expiry_test": True}
    response_data = {"data": "this will expire"}
    ttl_short = 1  # 1 second TTL

    sut.put(prompt, adapter, model, options, response_data, ttl_seconds=ttl_short)

    # Cache hit immediately
    assert sut.get(prompt, adapter, model, options) == response_data

    time.sleep(ttl_short + 1)  # Wait for item to expire

    # Cache miss after expiry, and item should be deleted by get()
    assert sut.get(prompt, adapter, model, options) is None

    # Verify it was deleted from DB after the get() call found it expired
    fingerprint = sut._generate_fingerprint(prompt, adapter, model, options)
    with sut._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM prompt_cache WHERE fingerprint = ?", (fingerprint,)
        )
        assert cursor.fetchone() is None


def test_purge_expired(sut: PromptCache):
    prompt_A = "Prompt A (will expire)"
    response_A = {"data": "A"}
    options_A = {"id": "A"}
    adapter_A = "adapter_purge"
    model_A = "model_purge"

    prompt_B = "Prompt B (will not expire)"
    response_B = {"data": "B"}
    options_B = {"id": "B"}

    # Put one item with short TTL, one with default (long) TTL
    sut.put(prompt_A, adapter_A, model_A, options_A, response_A, ttl_seconds=1)
    sut.put(
        prompt_B, adapter_A, model_A, options_B, response_B
    )  # Uses default_ttl_seconds

    assert sut.get(prompt_A, adapter_A, model_A, options_A) is not None  # Hit A
    assert sut.get(prompt_B, adapter_A, model_A, options_B) is not None  # Hit B

    time.sleep(2)  # Ensure prompt_A is expired

    sut.purge_expired()

    # Prompt A should now be purged (None)
    # Note: get() itself also deletes expired items, so this primarily tests purge_expired's direct effect
    # For a stricter test of purge_expired, we'd query the DB directly before/after.
    with sut._get_connection() as conn:
        cursor = conn.cursor()
        fp_A = sut._generate_fingerprint(prompt_A, adapter_A, model_A, options_A)
        cursor.execute("SELECT * FROM prompt_cache WHERE fingerprint = ?", (fp_A,))
        assert cursor.fetchone() is None, "Expired item A was not purged"

    # Prompt B should still be there
    assert sut.get(prompt_B, adapter_A, model_A, options_B) is not None, (
        "Non-expired item B was purged"
    )


def test_delete_item(sut: PromptCache):
    prompt = "To be deleted"
    adapter = "delete_adapter"
    model = "delete_model"
    options = {"key": "val"}
    response_data = {"message": "delete me"}
    fingerprint = sut._generate_fingerprint(prompt, adapter, model, options)

    sut.put(prompt, adapter, model, options, response_data)
    assert sut.get(prompt, adapter, model, options) is not None  # Ensure it's there

    sut.delete(fingerprint)
    assert sut.get(prompt, adapter, model, options) is None  # Should be gone


def test_clear_all(sut: PromptCache):
    sut.put("p1", "a1", "m1", {}, {"r": 1})
    sut.put("p2", "a2", "m2", {}, {"r": 2})
    assert sut.get("p1", "a1", "m1", {}) is not None

    sut.clear_all()
    assert sut.get("p1", "a1", "m1", {}) is None
    assert sut.get("p2", "a2", "m2", {}) is None
    with sut._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM prompt_cache")
        assert cursor.fetchone()[0] == 0


def test_get_corrupted_json(sut: PromptCache):
    prompt = "Corrupted data test"
    adapter = "corrupt_adapter"
    model = "corrupt_model"
    options = {}
    fingerprint = sut._generate_fingerprint(prompt, adapter, model, options)

    # Manually insert corrupted JSON
    with sut._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO prompt_cache (fingerprint, response_data, timestamp, ttl_seconds) VALUES (?, ?, ?, ?)",
            (fingerprint, "this is not valid json", int(time.time()), 3600),
        )
        conn.commit()

    # Get should return None and delete the corrupted entry
    assert sut.get(prompt, adapter, model, options) is None
    # Verify it was deleted
    with sut._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM prompt_cache WHERE fingerprint = ?", (fingerprint,)
        )
        assert cursor.fetchone() is None


def test_put_non_serializable_data(sut: PromptCache):
    prompt = "Non-serializable test"
    adapter = "non_serial_adapter"
    model = "non_serial_model"
    options = {}
    # Functions are not JSON serializable by default
    non_serializable_response = {"data": lambda x: x}

    # The current .put() method has a try-except for TypeError and returns.
    # It doesn't raise an error, so we check that no entry is made.
    sut.put(prompt, adapter, model, options, non_serializable_response)
    assert sut.get(prompt, adapter, model, options) is None

    # Verify that the DB does not contain an entry for this (it shouldn't if put failed silently)
    fingerprint = sut._generate_fingerprint(prompt, adapter, model, options)
    with sut._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT count(*) FROM prompt_cache WHERE fingerprint = ?", (fingerprint,)
        )
        assert cursor.fetchone()[0] == 0
