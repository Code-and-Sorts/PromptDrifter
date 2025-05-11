import json  # For constructing mock stream responses
from unittest.mock import (
    AsyncMock,
    MagicMock,
    # patch, # patch is not used if we directly inject client
)

import httpx
import pytest

from promptdrifter.adapters.ollama import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    OllamaAdapter,
)

# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_httpx_client(mocker, mock_streaming_response_generator):  # Add new fixture
    mock_client = MagicMock()  # Removed spec to simplify
    mock_client.post = AsyncMock()

    # .stream() is a synchronous method that returns an async context manager
    mock_client.stream = MagicMock()

    # Setup the async context manager that mock_client.stream will return
    async_context_manager_mock = AsyncMock()

    # Configure __aenter__ for the context manager
    mock_response_for_stream = MagicMock(spec=httpx.Response)
    mock_response_for_stream.status_code = 200
    # aiter_bytes is a method that returns an async generator.
    # It should be a MagicMock that, when called, returns the async generator instance.
    mock_response_for_stream.aiter_bytes = MagicMock(
        return_value=mock_streaming_response_generator
    )

    def simple_do_nothing_sync_callable(*args, **kwargs):
        return None

    mock_response_for_stream.raise_for_status = (
        simple_do_nothing_sync_callable  # Use a simple sync callable
    )

    aenter_mock = AsyncMock(return_value=mock_response_for_stream)
    async_context_manager_mock.__aenter__ = aenter_mock

    # Configure __aexit__ for the context manager
    async def mock_aexit(exc_type, exc_val, exc_tb):
        return None

    async_context_manager_mock.__aexit__ = AsyncMock(side_effect=mock_aexit)

    mock_client.stream.return_value = async_context_manager_mock

    mock_client.aclose = AsyncMock()
    return mock_client


@pytest.fixture
def ollama_adapter(mock_httpx_client):
    adapter = OllamaAdapter()
    adapter.client = mock_httpx_client
    return adapter


# --- Non-Streaming Tests (stream=False) ---
async def test_execute_non_streaming_successful(ollama_adapter, mock_httpx_client):
    mock_response_data = {
        "model": "test-model",
        "created_at": "2023-10-13T10:00:00Z",
        "response": "This is a test response from Ollama.",
        "done": True,
        # ... other fields Ollama might return
    }
    mock_httpx_client.post.return_value = httpx.Response(
        200, json=mock_response_data, request=httpx.Request("POST", "/api/generate")
    )

    prompt = "Hello, Ollama!"
    result = await ollama_adapter.execute(
        prompt, model="test-model", temperature=0.6, max_tokens=60, stream=False
    )

    mock_httpx_client.post.assert_called_once()
    call_args = mock_httpx_client.post.call_args
    assert call_args[0][0] == "/api/generate"
    payload = call_args[1]["json"]
    assert payload["model"] == "test-model"
    assert payload["prompt"] == prompt
    assert payload["stream"] is False
    assert payload["options"]["temperature"] == 0.6
    assert payload["options"]["num_predict"] == 60

    assert result["text_response"] == "This is a test response from Ollama."
    assert result["raw_response"] == mock_response_data
    assert result["model_used"] == "test-model"
    assert "error" not in result
    await ollama_adapter.close()


async def test_execute_non_streaming_uses_default_model_and_url(mock_httpx_client):
    # Test __init__ and default model usage without overriding them in execute
    adapter = OllamaAdapter()  # Uses default base_url
    adapter.client = mock_httpx_client  # Inject mock client

    mock_httpx_client.post.return_value = httpx.Response(
        200,
        json={"response": "Default model response"},
        request=httpx.Request("POST", "/"),
    )
    await adapter.execute("A prompt", stream=False)
    payload = mock_httpx_client.post.call_args[1]["json"]
    assert payload["model"] == DEFAULT_OLLAMA_MODEL
    assert adapter.base_url == DEFAULT_OLLAMA_BASE_URL
    await adapter.close()


async def test_execute_non_streaming_custom_options(ollama_adapter, mock_httpx_client):
    mock_httpx_client.post.return_value = httpx.Response(
        200, json={"response": "ok"}, request=httpx.Request("POST", "/")
    )
    await ollama_adapter.execute(
        "A prompt", model="custom", stream=False, stop=["\n", "User:"], top_k=30
    )

    payload = mock_httpx_client.post.call_args[1]["json"]
    assert payload["options"]["stop"] == ["\n", "User:"]
    assert payload["options"]["top_k"] == 30
    assert "temperature" not in payload["options"]  # Not specified, so not included
    await ollama_adapter.close()


# --- Streaming Tests (stream=True) ---
@pytest.fixture
def mock_streaming_response_generator():  # Renamed, returns generator instance
    chunks = [
        {
            "model": "test-stream-model",
            "created_at": "time1",
            "response": "Hello ",
            "done": False,
        },
        {
            "model": "test-stream-model",
            "created_at": "time2",
            "response": "Ollama",
            "done": False,
        },
        {
            "model": "test-stream-model",
            "created_at": "time3",
            "response": "!",
            "done": False,
        },
        {
            "model": "test-stream-model",
            "created_at": "time4",
            "response": "",
            "done": True,
            "total_duration": 1000,
            "context": [1, 2, 3],
        },
    ]

    async def byte_stream():
        for chunk_data in chunks:
            yield json.dumps(chunk_data).encode("utf-8") + b"\n"

    return byte_stream()  # Return the generator INSTANCE


async def test_execute_streaming_successful(
    ollama_adapter,
    mock_httpx_client,  # Removed mock_streaming_response_chunks, using generator from client setup
):
    # mock_httpx_client is now pre-configured with stream handling by the mock_httpx_client fixture

    # Add assertions to verify mock identity
    assert ollama_adapter.client is mock_httpx_client, (
        "ollama_adapter.client is not the mock_httpx_client instance"
    )
    assert ollama_adapter.client.stream is mock_httpx_client.stream, (
        "ollama_adapter.client.stream is not the mock_httpx_client.stream instance"
    )
    assert isinstance(mock_httpx_client.stream, MagicMock), (
        "mock_httpx_client.stream is not a MagicMock"
    )

    prompt = "Stream this!"
    result = await ollama_adapter.execute(
        prompt, model="test-stream-model", stream=True
    )

    mock_httpx_client.stream.assert_called_once()  # This is the key assertion

    # We can also check arguments passed to stream if needed
    call_args = mock_httpx_client.stream.call_args
    assert call_args[0][0] == "POST"  # method
    assert call_args[0][1] == "/api/generate"  # url
    payload = call_args[1]["json"]
    assert payload["model"] == "test-stream-model"
    assert payload["prompt"] == prompt
    assert payload["stream"] is True

    assert result["text_response"] == "Hello Ollama!"
    assert len(result["raw_response"]["parts"]) == 4
    assert result["raw_response"]["parts"][0]["response"] == "Hello "
    assert result["raw_response"]["final_context"]["context"] == [1, 2, 3]
    assert "error" not in result
    await ollama_adapter.close()


# --- Error Handling Tests (Applicable to both streaming and non-streaming where sensible) ---
async def test_execute_http_status_error(ollama_adapter, mock_httpx_client):
    error_response_content = '{"error": "model not found"}'
    mock_httpx_client.post.return_value = httpx.Response(
        404,
        content=error_response_content.encode("utf-8"),
        request=httpx.Request("POST", "/"),
    )
    result = await ollama_adapter.execute("A prompt", stream=False)
    assert "error" in result
    assert "HTTP error 404" in result["error"]
    assert "model not found" in result["error"]
    await ollama_adapter.close()


async def test_execute_request_error(ollama_adapter, mock_httpx_client):
    mock_httpx_client.post.side_effect = httpx.ConnectError("Connection failed")
    result = await ollama_adapter.execute("A prompt", stream=False)
    assert "error" in result
    assert "Request error connecting to Ollama" in result["error"]
    await ollama_adapter.close()


async def test_execute_json_decode_error_non_streaming(
    ollama_adapter, mock_httpx_client
):
    mock_httpx_client.post.return_value = httpx.Response(
        200, content="not json".encode("utf-8"), request=httpx.Request("POST", "/")
    )
    result = await ollama_adapter.execute("A prompt", stream=False)
    assert "error" in result
    assert "Failed to decode JSON response from Ollama" in result["error"]
    await ollama_adapter.close()


async def test_adapter_close_called(ollama_adapter, mock_httpx_client):
    await ollama_adapter.close()
    mock_httpx_client.aclose.assert_called_once()
