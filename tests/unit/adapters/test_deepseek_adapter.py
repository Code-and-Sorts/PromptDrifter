import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from promptdrifter.adapters.deepseek import DeepSeekAdapter, DeepSeekAdapterConfig
from promptdrifter.config.adapter_settings import (
    API_KEY_ENV_VAR_DEEPSEEK,
    DEEPSEEK_API_BASE_URL,
    DEFAULT_DEEPSEEK_MODEL,
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
        "promptdrifter.adapters.deepseek.get_shared_client",
        async_mock,
    ) as patched_get_shared_client:
        yield patched_get_shared_client

@pytest.fixture
def auto_patch_httpx_client(mock_httpx_client):
    """Compatibility fixture to maintain test interface"""
    return MagicMock(return_value=mock_httpx_client)

@pytest.fixture
def adapter_config_data():
    return DeepSeekAdapterConfig(
        api_key="test-deepseek-api-key",
        base_url=DEEPSEEK_API_BASE_URL,
        default_model=DEFAULT_DEEPSEEK_MODEL,
        max_tokens=2048
    )

@pytest.fixture
def adapter(adapter_config_data, monkeypatch, patch_shared_client):
    monkeypatch.setenv(API_KEY_ENV_VAR_DEEPSEEK, adapter_config_data.api_key)
    config = adapter_config_data
    adapter_instance = DeepSeekAdapter(config=config)
    return adapter_instance

async def test_deepseek_adapter_init_with_direct_params(monkeypatch, patch_shared_client):
    monkeypatch.delenv(API_KEY_ENV_VAR_DEEPSEEK, raising=False)
    config = DeepSeekAdapterConfig(
        api_key="direct_ds_key",
        base_url="custom_ds_url",
        default_model="custom_ds_model",
        max_tokens=1000
    )
    adapter_instance = DeepSeekAdapter(config=config)
    assert adapter_instance.config.api_key == "direct_ds_key"
    assert adapter_instance.config.base_url == "custom_ds_url"
    assert adapter_instance.config.default_model == "custom_ds_model"
    assert adapter_instance.config.max_tokens == 1000

async def test_deepseek_adapter_init_with_env_key_and_defaults(monkeypatch, patch_shared_client):
    monkeypatch.setenv(API_KEY_ENV_VAR_DEEPSEEK, "env_ds_key")
    adapter_instance = DeepSeekAdapter()
    assert adapter_instance.config.api_key == "env_ds_key"
    assert adapter_instance.config.base_url == DEEPSEEK_API_BASE_URL
    assert adapter_instance.config.default_model == DEFAULT_DEEPSEEK_MODEL
    assert adapter_instance.config.max_tokens == DeepSeekAdapterConfig().max_tokens

async def test_deepseek_adapter_init_no_key_raises_error(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV_VAR_DEEPSEEK, raising=False)
    with pytest.raises(ValueError) as excinfo:
        DeepSeekAdapter()
    assert API_KEY_ENV_VAR_DEEPSEEK in str(excinfo.value)

async def test_deepseek_adapter_init_with_config_object(monkeypatch, auto_patch_httpx_client):
    monkeypatch.delenv(API_KEY_ENV_VAR_DEEPSEEK, raising=False)
    config = DeepSeekAdapterConfig(
        api_key="config_ds_key", base_url="config_ds_url", default_model="config_ds_model", max_tokens=500
    )
    adapter_instance = DeepSeekAdapter(config=config)
    assert adapter_instance.config is config

@pytest.fixture
def mock_successful_response_data():
    return {
        "id": "chatcmpl-mockdeepseekid",
        "object": "chat.completion",
        "created": 1699800000,
        "model": "deepseek-coder",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "Hello from DeepSeek test!"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40}
    }

async def test_execute_successful(adapter, auto_patch_httpx_client, mock_successful_response_data):
    mock_client = auto_patch_httpx_client.return_value
    mock_http_response = MagicMock(spec=httpx.Response)
    mock_http_response.status_code = 200
    mock_http_response.json = MagicMock(return_value=mock_successful_response_data)
    mock_http_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_http_response

    prompt = "Hi DeepSeek from test"
    config_override = DeepSeekAdapterConfig(
        default_model="deepseek-coder-override",
        temperature=0.8,
        max_tokens=120
    )
    result = await adapter.execute(prompt, config_override=config_override)

    mock_client.post.assert_called_once_with(
        "/chat/completions",
        json=adapter.config.get_payload(prompt, config_override),
        timeout=60.0,
    )
    assert result.text_response == "Hello from DeepSeek test!"
    assert result.raw_response == mock_successful_response_data
    assert result.model_name == "deepseek-coder-override"
    assert result.finish_reason == "stop"
    assert result.usage == mock_successful_response_data["usage"]
    assert result.error is None

async def test_execute_uses_config_defaults(adapter, auto_patch_httpx_client, mock_successful_response_data):
    mock_client = auto_patch_httpx_client.return_value
    mock_http_response = MagicMock(spec=httpx.Response)
    mock_http_response.status_code = 200
    mock_successful_response_data["model"] = adapter.config.default_model
    mock_http_response.json = MagicMock(return_value=mock_successful_response_data)
    mock_http_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_http_response

    result = await adapter.execute("Test prompt for DeepSeek defaults")

    _, call_kwargs = mock_client.post.call_args
    payload = call_kwargs["json"]
    assert payload["model"] == adapter.config.default_model
    assert payload["max_tokens"] == adapter.config.max_tokens
    assert result.model_name == adapter.config.default_model

@pytest.mark.parametrize(
    "status_code, error_json, expected_error_part",
    [
        (400, {"error": {"message": "Bad request to DeepSeek", "type": "invalid_param"}}, "Bad request to DeepSeek"),
        (401, {"error": {"message": "DeepSeek key invalid", "code": "auth_failed"}}, "DeepSeek key invalid"),
        (500, {"error": {"message": "DeepSeek server meltdown"}}, "DeepSeek server meltdown"),
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

    result = await adapter.execute("A failing prompt for DeepSeek")

    assert result.error is not None
    assert f"API Error (HTTP {status_code})" in result.error
    assert expected_error_part in result.error
    assert result.raw_response == error_json
    assert result.text_response is None
    assert result.model_name == adapter.config.default_model
    assert result.finish_reason == "error"

async def test_execute_request_error(adapter, auto_patch_httpx_client):
    mock_client = auto_patch_httpx_client.return_value
    error_message = "DeepSeek connection error"
    mock_client.post.side_effect = httpx.ConnectError(error_message)
    result = await adapter.execute("A prompt")
    assert result.error is not None
    assert "HTTP Client Error" in result.error
    assert error_message in result.error
    assert result.raw_response == {"error_detail": error_message}

async def test_execute_unexpected_exception(adapter, auto_patch_httpx_client):
    mock_client = auto_patch_httpx_client.return_value
    error_message = "Unexpected DeepSeek disaster"
    mock_client.post.side_effect = Exception(error_message)
    result = await adapter.execute("A prompt")
    assert result.error is not None
    assert "An unexpected error occurred" in result.error
    assert error_message in result.error
    assert result.raw_response == {"error": error_message}

async def test_close_client(adapter, auto_patch_httpx_client):
    await adapter.close()
    # With shared client manager, we don't close the client directly
    assert adapter._client is None
