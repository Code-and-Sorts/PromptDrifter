from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from promptdrifter.adapters.claude import ClaudeAdapter

ANTHROPIC_API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"


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
        mock_client = MagicMock()

        mock_client.post = AsyncMock(return_value=mock_response)

        mock_client.aclose = AsyncMock()

        mock_client_class.return_value = mock_client

        yield mock_client


@pytest.fixture
def mock_response_content():
    """Common response content fixture that can be configured by tests."""
    return {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello from Claude!"}],
        "model": "claude-3-opus-20240229",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }


def test_claude_adapter_init_with_direct_key(monkeypatch):
    monkeypatch.delenv(ANTHROPIC_API_KEY_ENV_VAR, raising=False)
    adapter = ClaudeAdapter(api_key="direct_claude_key")
    assert adapter.api_key == "direct_claude_key"
    assert adapter.headers["x-api-key"] == "direct_claude_key"


def test_claude_adapter_init_with_env_key(monkeypatch):
    monkeypatch.setenv(ANTHROPIC_API_KEY_ENV_VAR, "env_claude_key")
    adapter = ClaudeAdapter()
    assert adapter.api_key == "env_claude_key"
    assert adapter.headers["x-api-key"] == "env_claude_key"


def test_claude_adapter_init_no_key_raises_error(monkeypatch):
    monkeypatch.delenv(ANTHROPIC_API_KEY_ENV_VAR, raising=False)
    with pytest.raises(ValueError) as excinfo:
        ClaudeAdapter()
    assert ANTHROPIC_API_KEY_ENV_VAR in str(excinfo.value)


@pytest.fixture
def claude_adapter_env_key(monkeypatch):
    """Provides a ClaudeAdapter instance with API key from env var."""
    monkeypatch.setenv(ANTHROPIC_API_KEY_ENV_VAR, "test-anthropic-key")
    return ClaudeAdapter()


@pytest.mark.asyncio
async def test_execute_successful(
    claude_adapter_env_key,
    patch_httpx_async_client,
    mock_response,
    mock_response_content,
):
    mock_response.json.return_value = mock_response_content

    prompt = "Hi Claude"
    result = await claude_adapter_env_key.execute(
        prompt, model="claude-3-sonnet-20240229", temperature=0.7, max_tokens=100
    )

    patch_httpx_async_client.post.assert_called_once_with(
        ClaudeAdapter.API_ENDPOINT,
        headers=claude_adapter_env_key.headers,
        json={
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        },
        timeout=60.0,
    )

    assert result["text_response"] == "Hello from Claude!"
    assert result["raw_response"] == mock_response_content
    assert result["model_name"] == "claude-3-sonnet-20240229"
    assert result["finish_reason"] == "end_turn"
    assert result["usage"] == {"input_tokens": 10, "output_tokens": 20}
    assert result["error"] is None


@pytest.mark.asyncio
async def test_execute_uses_default_model(
    claude_adapter_env_key, patch_httpx_async_client, mock_response
):
    mock_response.json.return_value = {
        "content": [{"type": "text", "text": "Default model response"}],
        "stop_reason": "max_tokens",
    }

    await claude_adapter_env_key.execute("A prompt")

    payload = patch_httpx_async_client.post.call_args[1]["json"]
    assert payload["model"] == ClaudeAdapter.DEFAULT_MODEL
    assert payload["max_tokens"] == ClaudeAdapter.DEFAULT_MAX_TOKENS


@pytest.mark.asyncio
async def test_execute_with_system_prompt(
    claude_adapter_env_key, patch_httpx_async_client, mock_response
):
    mock_response.json.return_value = {
        "content": [{"type": "text", "text": "Sys prompt response"}],
        "stop_reason": "end_turn",
    }

    system_prompt_text = "You are a helpful assistant."
    await claude_adapter_env_key.execute(
        "User prompt", system_prompt=system_prompt_text
    )

    payload = patch_httpx_async_client.post.call_args[1]["json"]
    assert payload["system"] == system_prompt_text


@pytest.mark.asyncio
async def test_execute_no_text_in_response_content(
    claude_adapter_env_key, patch_httpx_async_client, mock_response
):
    mock_response.json.return_value = {
        "content": [{"type": "image", "source": "..."}],
        "stop_reason": "end_turn",
    }

    result = await claude_adapter_env_key.execute("A prompt")
    assert result["text_response"] is None
    assert result["error"] == "No text content found in successful response."
    assert result["finish_reason"] == "end_turn"


@pytest.mark.asyncio
async def test_execute_empty_response_content(
    claude_adapter_env_key, patch_httpx_async_client, mock_response
):
    mock_response.json.return_value = {"content": [], "stop_reason": "end_turn"}

    result = await claude_adapter_env_key.execute("A prompt")
    assert result["text_response"] is None
    assert result["error"] == "Unexpected response structure for 200 OK."


@pytest.mark.asyncio
async def test_execute_malformed_success_response(
    claude_adapter_env_key, patch_httpx_async_client, mock_response
):
    mock_response.json.return_value = {"unexpected_field": "no_content_here"}

    result = await claude_adapter_env_key.execute("A prompt")
    assert result["text_response"] is None
    assert result["error"] == "Unexpected response structure for 200 OK."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status_code, error_type, error_message, expected_error_detail",
    [
        (
            400,
            "invalid_request_error",
            "Missing 'model' parameter",
            "Missing 'model' parameter",
        ),
        (401, "authentication_error", "Invalid API Key", "Invalid API Key"),
        (
            403,
            "permission_error",
            "You don't have access to this model.",
            "You don't have access to this model.",
        ),
        (
            429,
            "rate_limit_error",
            "You have hit your rate limit.",
            "You have hit your rate limit.",
        ),
        (500, "api_error", "Internal server error", "Internal server error"),
        (503, "api_error", "Service unavailable", "Service unavailable"),
    ],
)
async def test_execute_http_status_error_json_body(
    claude_adapter_env_key,
    patch_httpx_async_client,
    mock_response,
    status_code,
    error_type,
    error_message,
    expected_error_detail,
):
    error_json = {"error": {"type": error_type, "message": error_message}}
    mock_response.status_code = status_code
    mock_response.json.return_value = error_json

    http_error = httpx.HTTPStatusError(
        message=f"HTTP Error {status_code}",
        request=MagicMock(),
        response=mock_response
    )

    mock_response.raise_for_status.side_effect = http_error
    patch_httpx_async_client.post.return_value = mock_response

    result = await claude_adapter_env_key.execute("A prompt")

    assert result["error"] is not None
    assert f"API Error (HTTP {status_code})" in result["error"]
    assert expected_error_detail in result["error"]
    assert result["text_response"] is None


@pytest.mark.asyncio
async def test_execute_http_status_error_non_json_body(
    claude_adapter_env_key, patch_httpx_async_client, mock_response
):
    status_code = 502
    error_text = "Bad Gateway"
    mock_response.status_code = status_code
    mock_response.json.side_effect = ValueError("Invalid JSON")
    mock_response.text = error_text
    mock_response.reason_phrase = "Bad Gateway"

    http_error = httpx.HTTPStatusError(
        message=f"HTTP Error {status_code}",
        request=MagicMock(),
        response=mock_response
    )

    mock_response.raise_for_status.side_effect = http_error
    patch_httpx_async_client.post.return_value = mock_response

    result = await claude_adapter_env_key.execute("A prompt")

    assert result["error"] is not None
    assert f"API Error (HTTP {status_code})" in result["error"]
    assert "Bad Gateway" in result["error"]
    assert result["text_response"] is None


@pytest.mark.asyncio
async def test_execute_httpx_request_error(
    claude_adapter_env_key, patch_httpx_async_client
):
    error = httpx.RequestError("Connection refused", request=MagicMock())
    patch_httpx_async_client.post.side_effect = error

    result = await claude_adapter_env_key.execute("A prompt")

    assert result["error"] is not None
    assert "HTTP Client Error" in result["error"]
    assert "Connection refused" in result["error"]
    assert result["text_response"] is None
    assert result["finish_reason"] == "error"


@pytest.mark.asyncio
async def test_execute_timeout_error(
    claude_adapter_env_key, patch_httpx_async_client
):
    patch_httpx_async_client.post.side_effect = httpx.ReadTimeout("Request timed out")

    result = await claude_adapter_env_key.execute("A prompt")

    assert result["error"] is not None
    assert "Request timed out" in result["error"]
    assert result["text_response"] is None
    assert result["finish_reason"] == "error"


@pytest.mark.asyncio
async def test_execute_unexpected_exception(
    claude_adapter_env_key, patch_httpx_async_client
):
    error_msg = "Something unexpected happened"
    patch_httpx_async_client.post.side_effect = Exception(error_msg)

    result = await claude_adapter_env_key.execute("A prompt")

    assert result["error"] is not None
    assert "An unexpected error occurred" in result["error"]
    assert error_msg in result["error"]
    assert result["text_response"] is None
    assert result["finish_reason"] == "error"


@pytest.mark.asyncio
async def test_api_key_missing_during_execute():
    adapter = ClaudeAdapter(api_key="key")
    adapter.api_key = None

    result = await adapter.execute("A prompt")

    assert result["error"] == "Anthropic API key not configured."
    assert result["finish_reason"] == "error"
    assert result["text_response"] is None
