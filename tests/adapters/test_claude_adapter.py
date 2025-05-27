import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from promptdrifter.adapters.claude import ClaudeAdapter, ClaudeAdapterConfig
from promptdrifter.config.adapter_settings import (
    API_KEY_ENV_VAR_CLAUDE,
    CLAUDE_API_BASE_URL,
    DEFAULT_CLAUDE_MODEL,
)

pytestmark = pytest.mark.asyncio

@pytest.fixture(autouse=True)
def auto_patch_httpx_client():
    mock_client_instance = MagicMock(spec=httpx.AsyncClient)
    mock_client_instance.post = AsyncMock()
    mock_client_instance.aclose = AsyncMock()
    with patch("promptdrifter.adapters.claude.httpx.AsyncClient", return_value=mock_client_instance) as patched_client:
        yield patched_client

@pytest.fixture
def adapter_config_data():
    return ClaudeAdapterConfig(
        api_key="test-claude-api-key",
        base_url=CLAUDE_API_BASE_URL,
        default_model=DEFAULT_CLAUDE_MODEL,
        max_tokens=1024,
        api_version="2023-06-01"
    )

@pytest.fixture
def adapter(adapter_config_data, monkeypatch, auto_patch_httpx_client):
    monkeypatch.setenv(API_KEY_ENV_VAR_CLAUDE, adapter_config_data.api_key)
    config = adapter_config_data
    adapter_instance = ClaudeAdapter(config=config)
    return adapter_instance

async def test_claude_adapter_init_with_direct_params(monkeypatch, auto_patch_httpx_client):
    monkeypatch.delenv(API_KEY_ENV_VAR_CLAUDE, raising=False)
    config = ClaudeAdapterConfig(
        api_key="direct_claude_key",
        base_url="custom_claude_url",
        default_model="custom_claude_model",
        max_tokens=500,
        api_version="2024-01-01"
    )
    adapter_instance = ClaudeAdapter(config=config)
    assert adapter_instance.config.api_key == "direct_claude_key"
    assert adapter_instance.config.base_url == "custom_claude_url"
    assert adapter_instance.config.default_model == "custom_claude_model"
    assert adapter_instance.config.max_tokens == 500
    assert adapter_instance.config.api_version == "2024-01-01"

async def test_claude_adapter_init_with_env_key_and_defaults(monkeypatch, auto_patch_httpx_client):
    monkeypatch.setenv(API_KEY_ENV_VAR_CLAUDE, "env_claude_key")
    adapter_instance = ClaudeAdapter()
    assert adapter_instance.config.api_key == "env_claude_key"
    assert adapter_instance.config.base_url == CLAUDE_API_BASE_URL
    assert adapter_instance.config.default_model == DEFAULT_CLAUDE_MODEL
    assert adapter_instance.config.max_tokens == ClaudeAdapterConfig().max_tokens
    assert adapter_instance.config.api_version == ClaudeAdapterConfig().api_version

async def test_claude_adapter_init_no_key_raises_error(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV_VAR_CLAUDE, raising=False)
    with pytest.raises(ValueError) as excinfo:
        ClaudeAdapter()
    assert API_KEY_ENV_VAR_CLAUDE in str(excinfo.value)

async def test_claude_adapter_init_with_config_object(monkeypatch, auto_patch_httpx_client):
    monkeypatch.delenv(API_KEY_ENV_VAR_CLAUDE, raising=False)
    config = ClaudeAdapterConfig(
        api_key="config_claude_key",
        base_url="config_claude_url",
        default_model="config_claude_model",
        max_tokens=600,
        api_version="2023-10-10"
    )
    adapter_instance = ClaudeAdapter(config=config)
    assert adapter_instance.config is config

@pytest.fixture
def mock_claude_successful_response_data():
    return {
        "id": "msg_claude_test_id",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello from Claude test!"}],
        "model": "claude-3-opus-test",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "output_tokens": 22}
    }

async def test_execute_successful(adapter, auto_patch_httpx_client, mock_claude_successful_response_data):
    mock_client = auto_patch_httpx_client.return_value
    mock_http_response = MagicMock(spec=httpx.Response)
    mock_http_response.status_code = 200
    mock_http_response.json = MagicMock(return_value=mock_claude_successful_response_data)
    mock_http_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_http_response

    config_override = ClaudeAdapterConfig(
        default_model="claude-3-haiku-override",
        temperature=0.6,
        max_tokens=150
    )
    prompt = "Hi Claude from test"
    result = await adapter.execute(prompt, config_override=config_override)

    expected_json_payload = {
        "model": "claude-3-haiku-override",
        "max_tokens": 150,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6
    }
    mock_client.post.assert_called_once_with(
        CLAUDE_API_BASE_URL,
        json=expected_json_payload,
        timeout=60.0
    )
    assert result.text_response == "Hello from Claude test!"
    assert result.raw_response == mock_claude_successful_response_data
    assert result.model_name == "claude-3-haiku-override"
    assert result.finish_reason == mock_claude_successful_response_data.get("stop_reason")
    assert result.error is None
    await adapter.close()
    mock_client.aclose.assert_called_once()

async def test_execute_uses_config_defaults(adapter, auto_patch_httpx_client, mock_claude_successful_response_data):
    mock_client = auto_patch_httpx_client.return_value
    mock_http_response = MagicMock(spec=httpx.Response)
    mock_http_response.status_code = 200
    mock_claude_successful_response_data["model"] = adapter.config.default_model
    mock_http_response.json = MagicMock(return_value=mock_claude_successful_response_data)
    mock_http_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_http_response

    await adapter.execute("Test prompt for Claude defaults")

    _, call_kwargs = mock_client.post.call_args
    payload = call_kwargs["json"]
    assert payload["model"] == adapter.config.default_model
    assert payload["max_tokens"] == adapter.config.max_tokens

@pytest.mark.parametrize(
    "status_code, error_json, expected_error_part",
    [
        (400, {"error": {"type": "invalid_request_error", "message": "Claude bad param"}}, "Claude bad param"),
        (401, {"error": {"type": "authentication_error", "message": "Claude auth error"}}, "Claude auth error"),
        (429, {"error": {"type": "rate_limit_error", "message": "Claude rate limited"}}, "Claude rate limited"),
    ]
)
async def test_execute_http_status_error_json_body(
    adapter, auto_patch_httpx_client, status_code, error_json, expected_error_part
):
    mock_client = auto_patch_httpx_client.return_value
    mock_error_http_response = MagicMock(spec=httpx.Response)
    mock_error_http_response.status_code = status_code
    mock_error_http_response.json = MagicMock(return_value=error_json)
    mock_error_http_response.text = json.dumps(error_json)
    mock_error_http_response.request = httpx.Request("POST", CLAUDE_API_BASE_URL)
    mock_error_http_response.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError(
        message=f"HTTP {status_code}", request=mock_error_http_response.request, response=mock_error_http_response
    ))
    mock_client.post.return_value = mock_error_http_response

    result = await adapter.execute("A failing Claude prompt")

    assert result.error is not None
    assert f"API Error (HTTP {status_code})" in result.error
    assert expected_error_part in result.error
    assert result.raw_response == error_json
    assert result.text_response is None
    assert result.model_name == adapter.config.default_model
    assert result.finish_reason == "error"

async def test_execute_request_error(adapter, auto_patch_httpx_client):
    mock_client = auto_patch_httpx_client.return_value
    error_message = "Claude connection error"
    mock_client.post.side_effect = httpx.ConnectError(error_message)
    result = await adapter.execute("A prompt")
    assert result.error is not None
    assert "HTTP Client Error" in result.error
    assert error_message in result.error
    assert result.raw_response == {"error_detail": error_message}

async def test_close_client(adapter, auto_patch_httpx_client):
    mock_client = auto_patch_httpx_client.return_value
    await adapter.close()
    mock_client.aclose.assert_called_once()
