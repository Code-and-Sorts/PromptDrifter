import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from promptdrifter.adapters.grok import GrokAdapter, GrokAdapterConfig
from promptdrifter.adapters.models import GrokResponse
from promptdrifter.config.adapter_settings import (
    API_KEY_ENV_VAR_GROK,
    DEFAULT_GROK_MODEL,
    GROK_API_BASE_URL,
)

pytestmark = pytest.mark.asyncio

@pytest.fixture
def mock_response():
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.raise_for_status = MagicMock()
    response.json = MagicMock()
    return response

@pytest.fixture
def mock_httpx_client():
    client = MagicMock(spec=httpx.AsyncClient)
    client.post = AsyncMock()
    client.aclose = AsyncMock()
    return client

@pytest.fixture(autouse=True)
def patch_shared_client(mock_httpx_client):
    async_mock = AsyncMock(return_value=mock_httpx_client)
    with patch(
        "promptdrifter.adapters.grok.get_shared_client",
        async_mock,
    ) as patched_get_shared_client:
        yield patched_get_shared_client

@pytest.fixture
def auto_patch_httpx_client(mock_httpx_client):
    """Compatibility fixture to maintain test interface"""
    return MagicMock(return_value=mock_httpx_client)

@pytest.fixture
def adapter_config_data():
    return {
        "api_key": "test-grok-api-key",
        "base_url": GROK_API_BASE_URL,
        "default_model": DEFAULT_GROK_MODEL,
        "max_tokens": 4096,
    }

@pytest.fixture
def adapter(adapter_config_data, monkeypatch, auto_patch_httpx_client):
    monkeypatch.setenv(API_KEY_ENV_VAR_GROK, adapter_config_data["api_key"])
    config = GrokAdapterConfig(
        api_key=adapter_config_data["api_key"],
        base_url=adapter_config_data["base_url"],
        default_model=adapter_config_data["default_model"],
        max_tokens=adapter_config_data["max_tokens"]
    )
    adapter_instance = GrokAdapter(config=config)
    return adapter_instance

async def test_grok_adapter_init_with_direct_params(monkeypatch, auto_patch_httpx_client):
    monkeypatch.delenv(API_KEY_ENV_VAR_GROK, raising=False)
    config = GrokAdapterConfig(
        api_key="direct_grok_key_param",
        base_url="custom_grok_url_param",
        default_model="custom_grok_model_param",
        max_tokens=1024
    )
    adapter_instance = GrokAdapter(config=config)
    assert adapter_instance.config.api_key == "direct_grok_key_param"
    assert adapter_instance.config.base_url == "custom_grok_url_param"
    assert adapter_instance.config.default_model == "custom_grok_model_param"
    assert adapter_instance.config.max_tokens == 1024

async def test_grok_adapter_init_with_env_key_and_defaults(monkeypatch, auto_patch_httpx_client):
    monkeypatch.setenv(API_KEY_ENV_VAR_GROK, "env_grok_key_defaults")
    config = GrokAdapterConfig()
    adapter_instance = GrokAdapter(config=config)
    assert adapter_instance.config.api_key == "env_grok_key_defaults"
    assert adapter_instance.config.base_url == GROK_API_BASE_URL
    assert adapter_instance.config.default_model == DEFAULT_GROK_MODEL
    assert adapter_instance.config.max_tokens == GrokAdapterConfig().max_tokens

async def test_grok_adapter_init_no_key_raises_error(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV_VAR_GROK, raising=False)
    with pytest.raises(ValueError) as excinfo:
        GrokAdapter()
    assert API_KEY_ENV_VAR_GROK in str(excinfo.value)

async def test_grok_adapter_init_with_config_object(monkeypatch, auto_patch_httpx_client):
    monkeypatch.delenv(API_KEY_ENV_VAR_GROK, raising=False)
    config = GrokAdapterConfig(
        api_key="config_grok_key", base_url="config_grok_url",
        default_model="config_grok_model", max_tokens=2000
    )
    adapter_instance = GrokAdapter(config=config)
    assert adapter_instance.config is config

@pytest.fixture
def mock_grok_successful_response_data():
    return {
        "id": "grok_chat_test_id",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "Hello from Grok test!"}, "finish_reason": "stop"}
        ],
        "model": "grok-1-test",
        "usage": {"prompt_tokens": 11, "completion_tokens": 21, "total_tokens": 32}
    }

async def test_execute_successful(adapter, auto_patch_httpx_client, mock_grok_successful_response_data):
    mock_client = auto_patch_httpx_client.return_value
    mock_http_response = MagicMock(spec=httpx.Response)
    mock_http_response.status_code = 200
    mock_http_response.json = MagicMock(return_value=mock_grok_successful_response_data)
    mock_http_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_http_response

    config_override = GrokAdapterConfig(
        default_model="grok-1-override",
        temperature=0.5,
        max_tokens=180
    )
    result = await adapter.execute("Hi Grok from test", config_override=config_override)

    expected_json_payload = {
        "model": "grok-1-override",
        "max_tokens": 180,
        "messages": [{"role": "user", "content": "Hi Grok from test"}],
        "temperature": 0.5,
    }
    mock_client.post.assert_called_once_with(
        "/v1/chat/completions",
        json=expected_json_payload,
        timeout=60.0,
    )
    assert isinstance(result, GrokResponse)
    assert result.text_response == "Hello from Grok test!"
    assert result.raw_response["choices"] == mock_grok_successful_response_data["choices"]
    assert result.raw_response["usage"] == mock_grok_successful_response_data["usage"]
    assert result.model_name == "grok-1-override"
    assert result.finish_reason == "stop"
    assert result.usage == mock_grok_successful_response_data["usage"]
    assert result.error is None

async def test_execute_uses_config_defaults(adapter, auto_patch_httpx_client, mock_grok_successful_response_data):
    mock_client = auto_patch_httpx_client.return_value
    mock_http_response = MagicMock(spec=httpx.Response)
    mock_http_response.status_code = 200
    mock_grok_successful_response_data["model"] = adapter.config.default_model
    mock_http_response.json = MagicMock(return_value=mock_grok_successful_response_data)
    mock_http_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_http_response

    result = await adapter.execute("Test prompt for Grok defaults")
    assert isinstance(result, GrokResponse)
    _, call_kwargs = mock_client.post.call_args
    payload = call_kwargs["json"]
    assert payload["model"] == adapter.config.default_model
    assert payload["max_tokens"] == adapter.config.max_tokens

@pytest.mark.parametrize(
    "status_code, error_json, expected_error_part",
    [
        (400, {"error": {"message": "Grok bad request param"}}, "Grok bad request param"),
        (401, {"error": {"message": "Grok auth failed horribly"}}, "Grok auth failed horribly"),
        (500, {"error": {"message": "Grok server is on fire"}}, "Grok server is on fire"),
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
    mock_error_http_response.request = httpx.Request("POST", "/v1/chat/completions")
    mock_error_http_response.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError(
        message=f"HTTP {status_code}", request=mock_error_http_response.request, response=mock_error_http_response
    ))
    mock_client.post.return_value = mock_error_http_response

    result = await adapter.execute("A failing Grok prompt")
    assert isinstance(result, GrokResponse)
    assert result.error is not None
    assert f"API Error (HTTP {status_code})" in result.error
    assert expected_error_part in result.error
    assert result.raw_response == error_json
    assert result.text_response is None
    assert result.model_name == adapter.config.default_model
    assert result.finish_reason == "error"

async def test_execute_request_error(adapter, auto_patch_httpx_client):
    mock_client = auto_patch_httpx_client.return_value
    error_message = "Grok connection error test"
    mock_client.post.side_effect = httpx.ConnectError(error_message)
    result = await adapter.execute("A prompt for Grok request error")
    assert isinstance(result, GrokResponse)
    assert result.error is not None
    assert "HTTP Client Error" in result.error
    assert error_message in result.error
    assert result.raw_response == {"error_detail": error_message}

async def test_close_client(adapter):
    # Test that close method completes without error
    # With shared client manager, we don't close individual clients
    await adapter.close()
    assert adapter._client is None
