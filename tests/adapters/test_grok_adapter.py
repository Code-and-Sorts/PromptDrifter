import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from promptdrifter.adapters.grok import GrokAdapter

# Environment variable for Grok API Key
GROK_API_KEY_ENV_VAR = "GROK_API_KEY"


@pytest.fixture
def mock_response():
    """Creates a mock httpx.Response."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.json = MagicMock(return_value={})
    response.text = "Response text"
    response.raise_for_status = MagicMock()
    return response


@pytest.fixture
def patch_httpx_async_client(mock_response):
    """Patch httpx.AsyncClient to return a mock response directly from post."""
    with patch("httpx.AsyncClient") as mock_client_class:
        # Create a client instance mock
        mock_client = MagicMock()

        # Setup the post method to return our prepared response
        mock_client.post = AsyncMock(return_value=mock_response)

        # Setup the close method
        mock_client.aclose = AsyncMock()

        # Return the client when AsyncClient is instantiated
        mock_client_class.return_value = mock_client

        yield mock_client


@pytest.fixture
def mock_response_content():
    """Common response content fixture that can be configured by tests."""
    return {
        "id": "chat_123",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from Grok!"},
                "finish_reason": "stop",
            }
        ],
        "model": "grok-1",
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


# --- Initialization Tests ---


def test_grok_adapter_init_with_direct_key(monkeypatch):
    monkeypatch.delenv(GROK_API_KEY_ENV_VAR, raising=False)
    adapter = GrokAdapter(api_key="direct_grok_key")
    assert adapter.api_key == "direct_grok_key"
    assert adapter.headers["Authorization"] == "Bearer direct_grok_key"


def test_grok_adapter_init_with_env_key(monkeypatch):
    monkeypatch.setenv(GROK_API_KEY_ENV_VAR, "env_grok_key")
    adapter = GrokAdapter()
    assert adapter.api_key == "env_grok_key"
    assert adapter.headers["Authorization"] == "Bearer env_grok_key"


def test_grok_adapter_init_no_key_raises_error(monkeypatch):
    monkeypatch.delenv(GROK_API_KEY_ENV_VAR, raising=False)
    with pytest.raises(ValueError) as excinfo:
        GrokAdapter()
    assert GROK_API_KEY_ENV_VAR in str(excinfo.value)


# --- Execution Tests ---


@pytest.fixture
def grok_adapter_env_key(monkeypatch):
    """Provides a GrokAdapter instance with API key from env var."""
    monkeypatch.setenv(GROK_API_KEY_ENV_VAR, "test-grok-key")
    return GrokAdapter()


@pytest.mark.asyncio
async def test_execute_successful(
    grok_adapter_env_key,
    patch_httpx_async_client,
    mock_response,
    mock_response_content,
):
    # Set up the mock response
    mock_response.json.return_value = mock_response_content

    # Execute the adapter
    prompt = "Hi Grok"
    result = await grok_adapter_env_key.execute(
        prompt, model="grok-1", temperature=0.7, max_tokens=100
    )

    # Verify post was called with correct args
    patch_httpx_async_client.post.assert_called_once_with(
        GrokAdapter.API_ENDPOINT,
        headers=grok_adapter_env_key.headers,
        json={
            "model": "grok-1",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        },
        timeout=60.0,
    )

    # Verify the returned result
    assert result["text_response"] == "Hello from Grok!"
    assert result["raw_response"] == mock_response_content
    assert result["model_name"] == "grok-1"
    assert result["finish_reason"] == "stop"
    assert result["usage"] == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    assert result["error"] is None


@pytest.mark.asyncio
async def test_execute_uses_default_model(
    grok_adapter_env_key, patch_httpx_async_client, mock_response
):
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {"content": "Default model response"},
                "finish_reason": "max_tokens",
            }
        ]
    }

    await grok_adapter_env_key.execute("A prompt")

    payload = patch_httpx_async_client.post.call_args[1]["json"]
    assert payload["model"] == GrokAdapter.DEFAULT_MODEL
    assert payload["max_tokens"] == GrokAdapter.DEFAULT_MAX_TOKENS


@pytest.mark.asyncio
async def test_execute_with_system_prompt(
    grok_adapter_env_key, patch_httpx_async_client, mock_response
):
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {"content": "Sys prompt response"},
                "finish_reason": "stop",
            }
        ]
    }

    system_prompt_text = "You are a helpful assistant."
    await grok_adapter_env_key.execute(
        "User prompt", system_prompt=system_prompt_text
    )

    payload = patch_httpx_async_client.post.call_args[1]["json"]
    assert len(payload["messages"]) == 2
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][0]["content"] == system_prompt_text


@pytest.mark.asyncio
async def test_execute_no_text_in_response_content(
    grok_adapter_env_key, patch_httpx_async_client, mock_response
):
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {"role": "assistant"},  # No content field
                "finish_reason": "stop",
            }
        ]
    }

    result = await grok_adapter_env_key.execute("A prompt")
    assert result["text_response"] is None
    assert result["error"] == "No text content found in successful response."
    assert result["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_execute_empty_response_content(
    grok_adapter_env_key, patch_httpx_async_client, mock_response
):
    mock_response.json.return_value = {"choices": []}

    result = await grok_adapter_env_key.execute("A prompt")
    assert result["text_response"] is None
    assert result["error"] == "Unexpected response structure for 200 OK."


@pytest.mark.asyncio
async def test_execute_malformed_success_response(
    grok_adapter_env_key, patch_httpx_async_client, mock_response
):
    mock_response.json.return_value = {"unexpected_field": "no_choices_here"}

    result = await grok_adapter_env_key.execute("A prompt")
    assert result["text_response"] is None
    assert result["error"] == "Unexpected response structure for 200 OK."


# --- Error Handling Tests ---


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status_code, error_msg, expected_error_detail",
    [
        (400, "Invalid request parameters", "Invalid request parameters"),
        (401, "Invalid API key", "Invalid API key"),
        (429, "Rate limit exceeded", "Rate limit exceeded"),
        (500, "Internal server error", "Internal server error"),
    ],
)
async def test_execute_http_status_error_json_body(
    grok_adapter_env_key,
    patch_httpx_async_client,
    mock_response,
    status_code,
    error_msg,
    expected_error_detail,
):
    error_json = {"error": {"message": error_msg}}
    mock_response.status_code = status_code
    mock_response.json.return_value = error_json

    # Create an HTTPStatusError
    http_error = httpx.HTTPStatusError(
        message=f"HTTP Error {status_code}",
        request=MagicMock(),
        response=mock_response
    )

    # Have the raise_for_status method raise this error
    mock_response.raise_for_status.side_effect = http_error
    patch_httpx_async_client.post.return_value = mock_response

    result = await grok_adapter_env_key.execute("A prompt")

    assert result["error"] is not None
    assert f"API Error (HTTP {status_code})" in result["error"]
    assert expected_error_detail in result["error"]
    assert result["text_response"] is None


@pytest.mark.asyncio
async def test_execute_httpx_request_error(
    grok_adapter_env_key, patch_httpx_async_client
):
    error = httpx.RequestError("Connection refused", request=MagicMock())
    patch_httpx_async_client.post.side_effect = error

    result = await grok_adapter_env_key.execute("A prompt")

    assert result["error"] is not None
    assert "HTTP Client Error" in result["error"]
    assert "Connection refused" in result["error"]
    assert result["text_response"] is None
    assert result["finish_reason"] == "error"


@pytest.mark.asyncio
async def test_execute_timeout_error(
    grok_adapter_env_key, patch_httpx_async_client
):
    patch_httpx_async_client.post.side_effect = asyncio.TimeoutError()

    result = await grok_adapter_env_key.execute("A prompt")

    assert result["error"] is not None
    assert "Request timed out." in result["error"]
    assert result["text_response"] is None
    assert result["finish_reason"] == "error"


@pytest.mark.asyncio
async def test_execute_unexpected_exception(
    grok_adapter_env_key, patch_httpx_async_client
):
    error_msg = "Something unexpected happened"
    patch_httpx_async_client.post.side_effect = Exception(error_msg)

    result = await grok_adapter_env_key.execute("A prompt")

    assert result["error"] is not None
    assert "An unexpected error occurred" in result["error"]
    assert error_msg in result["error"]
    assert result["text_response"] is None
    assert result["finish_reason"] == "error"


@pytest.mark.asyncio
async def test_api_key_missing_during_execute():
    adapter = GrokAdapter(api_key="temp_key")
    adapter.api_key = None

    result = await adapter.execute("A prompt")
    assert result["error"] == "Grok API key not configured."
    assert result["text_response"] is None
    assert result["finish_reason"] == "error"
