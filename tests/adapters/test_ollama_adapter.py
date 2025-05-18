import json
from unittest.mock import (
    AsyncMock,
    MagicMock,
    patch,
)

import httpx
import pytest

from promptdrifter.adapters.ollama import (
    OllamaAdapter,
    OllamaAdapterConfig,
)
from promptdrifter.config.adapter_settings import (
    DEFAULT_OLLAMA_BASE_URL as config_DEFAULT_OLLAMA_BASE_URL,
)
from promptdrifter.config.adapter_settings import (
    DEFAULT_OLLAMA_MODEL as config_DEFAULT_OLLAMA_MODEL,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_httpx_client_instance():
    client = MagicMock(spec=httpx.AsyncClient)
    client.post = AsyncMock()
    client.stream = MagicMock()
    client.aclose = AsyncMock()

    async_context_manager_mock = AsyncMock()
    mock_response_for_stream = MagicMock(spec=httpx.Response)
    mock_response_for_stream.status_code = 200
    mock_response_for_stream.aiter_bytes = MagicMock()
    mock_response_for_stream.raise_for_status = MagicMock()

    async_context_manager_mock.__aenter__ = AsyncMock(
        return_value=mock_response_for_stream
    )
    async_context_manager_mock.__aexit__ = AsyncMock(return_value=None)
    client.stream.return_value = async_context_manager_mock
    return client


@pytest.fixture
def patch_httpx_client(mock_httpx_client_instance):
    with patch(
        "promptdrifter.adapters.ollama.httpx.AsyncClient",
        return_value=mock_httpx_client_instance,
    ) as patched_class_mock:
        yield patched_class_mock


@pytest.fixture
def ollama_adapter(patch_httpx_client):
    adapter_instance = OllamaAdapter()
    patch_httpx_client.assert_called_once_with(
        base_url=adapter_instance.config.base_url
    )
    return adapter_instance


async def test_ollama_adapter_init_default(patch_httpx_client):
    adapter = OllamaAdapter()
    assert adapter.config.base_url == config_DEFAULT_OLLAMA_BASE_URL
    assert adapter.config.default_model == config_DEFAULT_OLLAMA_MODEL
    patch_httpx_client.assert_called_once_with(base_url=config_DEFAULT_OLLAMA_BASE_URL)


async def test_ollama_adapter_init_custom_params(patch_httpx_client):
    custom_url = "http://customhost:11435"
    custom_model = "customollamamodel"
    adapter = OllamaAdapter(base_url=custom_url, default_model=custom_model)
    assert adapter.config.base_url == custom_url
    assert adapter.config.default_model == custom_model
    patch_httpx_client.assert_called_once_with(base_url=custom_url)


async def test_ollama_adapter_init_with_config_object(patch_httpx_client):
    custom_url = "http://confighost:11433"
    custom_model = "configollamamodel"
    config = OllamaAdapterConfig(base_url=custom_url, default_model=custom_model)
    adapter = OllamaAdapter(config=config)
    assert adapter.config is config
    assert adapter.config.base_url == custom_url
    assert adapter.config.default_model == custom_model
    patch_httpx_client.assert_called_once_with(base_url=custom_url)


async def test_execute_non_streaming_successful(ollama_adapter, patch_httpx_client):
    mock_client_instance = patch_httpx_client.return_value
    mock_response_data = {
        "model": "test-model",
        "response": "This is a test response from Ollama.",
        "done": True,
    }
    mock_client_instance.post.return_value = httpx.Response(
        200, json=mock_response_data, request=httpx.Request("POST", "/api/generate")
    )

    prompt = "Hello, Ollama!"
    result = await ollama_adapter.execute(
        prompt, model="test-model", temperature=0.6, max_tokens=60, stream=False
    )

    mock_client_instance.post.assert_called_once()
    call_args = mock_client_instance.post.call_args
    assert call_args[0][0] == "/api/generate"
    payload = call_args[1]["json"]
    assert payload["model"] == "test-model"
    assert payload["prompt"] == prompt
    assert payload["stream"] is False
    assert payload["options"]["temperature"] == 0.6
    assert payload["options"]["num_predict"] == 60

    assert result["text_response"] == "This is a test response from Ollama."
    await ollama_adapter.close()
    mock_client_instance.aclose.assert_called_once()


async def test_execute_non_streaming_uses_default_model_and_url(patch_httpx_client):
    adapter = OllamaAdapter()
    mock_client_instance = patch_httpx_client.return_value
    mock_client_instance.post.return_value = httpx.Response(
        200,
        json={"response": "Default model response"},
        request=httpx.Request("POST", "/"),
    )
    await adapter.execute("A prompt", stream=False)
    payload = mock_client_instance.post.call_args[1]["json"]
    assert payload["model"] == adapter.config.default_model
    assert adapter.config.base_url == config_DEFAULT_OLLAMA_BASE_URL
    await adapter.close()
    mock_client_instance.aclose.assert_called_once()


async def test_execute_with_system_prompt(ollama_adapter, patch_httpx_client):
    """Test execution with a system prompt parameter."""
    mock_client_instance = patch_httpx_client.return_value
    mock_response_data = {
        "model": "llama3",
        "response": "Response with system instructions",
        "done": True,
    }
    mock_client_instance.post.return_value = httpx.Response(
        200, json=mock_response_data, request=httpx.Request("POST", "/api/generate")
    )

    system_prompt = "You are a helpful AI assistant that provides concise answers."
    prompt = "Tell me about the solar system"

    result = await ollama_adapter.execute(
        prompt,
        system=system_prompt,
        model="llama3",
        stream=False
    )

    mock_client_instance.post.assert_called_once()
    payload = mock_client_instance.post.call_args[1]["json"]
    assert payload["options"]["system"] == system_prompt
    assert payload["prompt"] == prompt

    assert result["text_response"] == "Response with system instructions"
    await ollama_adapter.close()
    mock_client_instance.aclose.assert_called_once()


@pytest.fixture
def mock_streaming_byte_chunks():
    chunks_data = [
        {"response": "Hello ", "done": False},
        {"response": "Ollama", "done": False},
        {"response": "!", "done": False},
        {"response": "", "done": True, "context": [1, 2, 3]},
    ]

    async def byte_stream():
        for chunk_data in chunks_data:
            yield json.dumps(chunk_data).encode("utf-8") + b"\n"

    return byte_stream()


async def test_execute_streaming_successful(
    ollama_adapter, patch_httpx_client, mock_streaming_byte_chunks
):
    mock_client_instance = patch_httpx_client.return_value

    mock_response_for_stream = (
        await mock_client_instance.stream.return_value.__aenter__()
    )
    mock_response_for_stream.aiter_bytes.return_value = mock_streaming_byte_chunks

    prompt = "Stream this!"
    result = await ollama_adapter.execute(
        prompt, model="test-stream-model", stream=True
    )

    mock_client_instance.stream.assert_called_once()
    call_args = mock_client_instance.stream.call_args
    assert call_args[0][0] == "POST"
    assert call_args[0][1] == "/api/generate"
    payload = call_args[1]["json"]
    assert payload["model"] == "test-stream-model"
    assert payload["stream"] is True

    assert result["text_response"] == "Hello Ollama!"
    assert len(result["raw_response"]["parts"]) == 4
    assert result["raw_response"]["final_context"]["context"] == [1, 2, 3]
    await ollama_adapter.close()
    mock_client_instance.aclose.assert_called_once()


async def test_execute_http_status_error_non_streaming(
    ollama_adapter, patch_httpx_client
):
    mock_client_instance = patch_httpx_client.return_value
    error_response_content = '{"error": "model not found"}'
    mock_client_instance.post.return_value = httpx.Response(
        404,
        content=error_response_content.encode("utf-8"),
        request=httpx.Request("POST", "/"),
    )
    result = await ollama_adapter.execute("A prompt", stream=False)
    assert "error" in result
    assert "HTTP error 404" in result["error"]
    await ollama_adapter.close()
    mock_client_instance.aclose.assert_called_once()


async def test_execute_http_status_error_streaming(ollama_adapter, patch_httpx_client):
    mock_client_instance = patch_httpx_client.return_value
    error_response_content = '{"error": "streaming model not found"}'

    bad_stream_response = MagicMock(spec=httpx.Response)
    bad_stream_response.status_code = 500
    bad_stream_response.text = error_response_content
    bad_stream_response.aiter_bytes = MagicMock(return_value=(b for b in []))

    def raise_http_error(*args, **kwargs):
        raise httpx.HTTPStatusError(
            "Error", request=MagicMock(), response=bad_stream_response
        )

    bad_stream_response.raise_for_status = MagicMock(side_effect=raise_http_error)

    mock_client_instance.stream.return_value.__aenter__ = AsyncMock(
        return_value=bad_stream_response
    )

    result = await ollama_adapter.execute("A prompt", stream=True)
    assert "error" in result
    assert "HTTP error 500" in result["error"]
    await ollama_adapter.close()
    mock_client_instance.aclose.assert_called_once()


async def test_execute_request_error_non_streaming(ollama_adapter, patch_httpx_client):
    mock_client_instance = patch_httpx_client.return_value
    mock_client_instance.post.side_effect = httpx.ConnectError("Connection failed")
    result = await ollama_adapter.execute("A prompt", stream=False)
    assert "error" in result
    assert "Request error connecting to Ollama" in result["error"]
    await ollama_adapter.close()
    mock_client_instance.aclose.assert_called_once()


async def test_execute_request_error_streaming(ollama_adapter, patch_httpx_client):
    mock_client_instance = patch_httpx_client.return_value
    mock_client_instance.stream.side_effect = httpx.ConnectError(
        "Stream connection failed"
    )
    result = await ollama_adapter.execute("A prompt", stream=True)
    assert "error" in result
    assert "Request error connecting to Ollama" in result["error"]
    await ollama_adapter.close()
    mock_client_instance.aclose.assert_called_once()


async def test_execute_timeout_error(ollama_adapter, patch_httpx_client):
    """Test handling of timeout errors during requests."""
    mock_client_instance = patch_httpx_client.return_value
    mock_client_instance.post.side_effect = httpx.ReadTimeout("Request timed out after 60s")

    result = await ollama_adapter.execute("A prompt", stream=False)

    assert "error" in result
    assert "Request error connecting to Ollama" in result["error"]
    assert "timed out" in result["error"].lower()
    await ollama_adapter.close()
    mock_client_instance.aclose.assert_called_once()


async def test_execute_empty_response_content(ollama_adapter, patch_httpx_client):
    """Test handling of empty response content."""
    mock_client_instance = patch_httpx_client.return_value

    mock_client_instance.post.return_value = httpx.Response(
        200,
        json={"model": "test-model", "response": "", "done": True},
        request=httpx.Request("POST", "/api/generate")
    )

    result = await ollama_adapter.execute("A prompt", stream=False)

    assert result["text_response"] == ""
    await ollama_adapter.close()
    mock_client_instance.aclose.assert_called_once()


async def test_adapter_close_called(ollama_adapter, patch_httpx_client):
    mock_client_instance = patch_httpx_client.return_value
    await ollama_adapter.close()
    mock_client_instance.aclose.assert_called_once()
