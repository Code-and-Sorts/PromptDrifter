import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from promptdrifter.adapters.models.qwen_models import QwenStandardResponse
from promptdrifter.adapters.qwen import QwenAdapter, QwenAdapterConfig
from promptdrifter.config.adapter_settings import (
    API_KEY_ENV_VAR_QWEN,
    DEFAULT_QWEN_MODEL,
    QWEN_API_BASE_URL,
)

pytestmark = pytest.mark.asyncio

TEST_API_KEY = "test_qwen_api_key_fixture"
TEST_PROMPT = "Hello, Qwen from test!"
TEST_MODEL = "qwen-test-model"
CUSTOM_BASE_URL = "http://localhost:8001"

SUCCESS_RESPONSE_PAYLOAD = {
    "id": "chatcmpl-mockid",
    "object": "chat.completion",
    "created": 1677652288,
    "model": DEFAULT_QWEN_MODEL,
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "This is a test response from Qwen.",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
}

API_ERROR_RESPONSE_PAYLOAD = {
    "error": {
        "message": "The API key provided is invalid.",
        "type": "invalid_request_error",
        "param": None,
        "code": "invalid_api_key",
    }
}


@pytest.fixture(autouse=True)
def auto_patch_httpx_client():
    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_client.post = AsyncMock()
    mock_client.aclose = AsyncMock()
    with patch("promptdrifter.adapters.qwen.get_shared_client", return_value=mock_client) as patched_client:
        yield patched_client


@pytest.fixture
def adapter_config_data():
    return {
        "api_key": TEST_API_KEY,
        "base_url": QWEN_API_BASE_URL,
        "default_model": DEFAULT_QWEN_MODEL,
        "max_tokens": 2000,
    }


@pytest.fixture
def adapter(adapter_config_data, monkeypatch, auto_patch_httpx_client):
    monkeypatch.setenv(API_KEY_ENV_VAR_QWEN, adapter_config_data["api_key"])
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    config = QwenAdapterConfig(
        api_key=adapter_config_data["api_key"],
        base_url=adapter_config_data["base_url"],
        default_model=adapter_config_data["default_model"],
        max_tokens=adapter_config_data["max_tokens"]
    )
    adapter_instance = QwenAdapter(config=config)
    return adapter_instance


@pytest.fixture
def mock_qwen_successful_response_data():
    return {
        "id": "qwen-mock-id",
        "object": "chat.completion",
        "created": 1677652289,
        "model": "qwen-plus-test",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Qwen test response content."},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25}
    }


async def test_qwen_adapter_init_with_direct_params(monkeypatch, auto_patch_httpx_client):
    monkeypatch.delenv(API_KEY_ENV_VAR_QWEN, raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    config = QwenAdapterConfig(
        api_key="direct_q_key",
        base_url="custom_q_url",
        default_model="custom_q_model",
        max_tokens=555
    )
    adapter_instance = QwenAdapter(config=config)
    assert adapter_instance.config.api_key == "direct_q_key"
    assert adapter_instance.config.base_url == "custom_q_url"
    assert adapter_instance.config.default_model == "custom_q_model"
    assert adapter_instance.config.max_tokens == 555


async def test_qwen_adapter_init_with_qwen_env_var(monkeypatch, auto_patch_httpx_client):
    monkeypatch.setenv(API_KEY_ENV_VAR_QWEN, "qwen_env_key")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    adapter_instance = QwenAdapter()
    assert adapter_instance.config.api_key == "qwen_env_key"


async def test_qwen_adapter_init_with_dashscope_env_var(monkeypatch, auto_patch_httpx_client):
    monkeypatch.delenv(API_KEY_ENV_VAR_QWEN, raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dash_env_key")
    adapter_instance = QwenAdapter()
    assert adapter_instance.config.api_key == "dash_env_key"


async def test_qwen_adapter_init_api_key_priority(monkeypatch, auto_patch_httpx_client):
    monkeypatch.setenv(API_KEY_ENV_VAR_QWEN, "qwen_env_key_priority")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dash_env_key_priority")

    config_direct = QwenAdapterConfig(api_key="direct_priority_key")
    adapter_direct = QwenAdapter(config=config_direct)
    assert adapter_direct.config.api_key == "direct_priority_key"
    auto_patch_httpx_client.reset_mock()

    config_qwen_env = QwenAdapterConfig()
    adapter_qwen_env = QwenAdapter(config=config_qwen_env)
    assert adapter_qwen_env.config.api_key == "qwen_env_key_priority"
    auto_patch_httpx_client.reset_mock()

    monkeypatch.delenv(API_KEY_ENV_VAR_QWEN, raising=False)
    config_dash_env = QwenAdapterConfig()
    adapter_dash_env = QwenAdapter(config=config_dash_env)
    assert adapter_dash_env.config.api_key == "dash_env_key_priority"


async def test_qwen_adapter_init_missing_api_key(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV_VAR_QWEN, raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    with pytest.raises(ValueError) as exc_info:
        QwenAdapter()
    assert API_KEY_ENV_VAR_QWEN in str(exc_info.value)
    assert "DASHSCOPE_API_KEY" in str(exc_info.value)


async def test_qwen_adapter_init_with_config_object(monkeypatch, auto_patch_httpx_client):
    monkeypatch.delenv(API_KEY_ENV_VAR_QWEN, raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    config = QwenAdapterConfig(
        api_key="cfg_q_key", base_url="cfg_q_url", default_model="cfg_q_model", max_tokens=600
    )
    adapter_instance = QwenAdapter(config=config)
    assert adapter_instance.config is config


async def test_execute_successful(adapter, auto_patch_httpx_client):
    mock_client = auto_patch_httpx_client.return_value
    mock_http_response = MagicMock(spec=httpx.Response)
    mock_http_response.status_code = 200
    mock_http_response.json = MagicMock(return_value=SUCCESS_RESPONSE_PAYLOAD)
    mock_http_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_http_response

    config_override = QwenAdapterConfig(
        default_model="qwen-max-override",
        temperature=0.9,
        max_tokens=250
    )
    result = await adapter.execute(TEST_PROMPT, config_override=config_override)

    expected_payload = {
        "model": "qwen-max-override",
        "messages": [{"role": "user", "content": TEST_PROMPT}],
        "temperature": 0.9,
        "max_tokens": 250
    }
    mock_client.post.assert_called_once_with(
        "/chat/completions", json=expected_payload, timeout=180.0
    )
    assert isinstance(result, QwenStandardResponse)
    assert result.text_response == SUCCESS_RESPONSE_PAYLOAD["choices"][0]["message"]["content"]
    assert result.raw_response == SUCCESS_RESPONSE_PAYLOAD
    assert result.model_name == "qwen-max-override"
    assert result.finish_reason == "stop"
    assert result.usage == SUCCESS_RESPONSE_PAYLOAD["usage"]
    assert result.error is None


async def test_execute_uses_config_defaults(adapter, auto_patch_httpx_client):
    mock_client = auto_patch_httpx_client.return_value
    mock_http_response = MagicMock(spec=httpx.Response)
    mock_http_response.status_code = 200
    mock_http_response.json = MagicMock(return_value=SUCCESS_RESPONSE_PAYLOAD)
    mock_http_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_http_response

    await adapter.execute("Test prompt for Qwen defaults")
    _, call_kwargs = mock_client.post.call_args
    payload = call_kwargs["json"]
    assert payload["model"] == adapter.config.default_model
    assert payload["max_tokens"] == adapter.config.max_tokens


async def test_execute_api_error_in_json_response(adapter, auto_patch_httpx_client):
    mock_client = auto_patch_httpx_client.return_value
    mock_http_response = MagicMock(spec=httpx.Response)
    mock_http_response.status_code = 200
    mock_http_response.json = MagicMock(return_value=API_ERROR_RESPONSE_PAYLOAD)
    mock_http_response.text = json.dumps(API_ERROR_RESPONSE_PAYLOAD)
    mock_http_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_http_response

    result = await adapter.execute(TEST_PROMPT)

    assert isinstance(result, QwenStandardResponse)
    assert result.error.startswith("Qwen API error (type:")
    assert "invalid_request_error" in result.error
    assert "invalid_api_key" in result.error
    assert "The API key provided is invalid" in result.error
    assert result.finish_reason == "error"
    assert result.text_response is None
    assert result.raw_response == API_ERROR_RESPONSE_PAYLOAD


@pytest.mark.parametrize(
    "status_code, error_json_response, expected_error_text_part",
    [
        (400, {"error":{"message":"Qwen bad request","code":"invalid_param"}}, "Qwen bad request"),
        (401, {"error":{"message":"Qwen auth failed","code":"permission_denied"}}, "Qwen auth failed"),
        (503, {"error":{"message":"Qwen service unavailable"}}, "Qwen service unavailable"),
    ]
)
async def test_execute_http_status_error_json_body(
    adapter, auto_patch_httpx_client, status_code, error_json_response, expected_error_text_part
):
    mock_client = auto_patch_httpx_client.return_value
    mock_err_http_response = MagicMock(spec=httpx.Response)
    mock_err_http_response.status_code = status_code
    mock_err_http_response.json = MagicMock(return_value=error_json_response)
    mock_err_http_response.text = json.dumps(error_json_response)
    mock_err_http_response.request = httpx.Request("POST", "/chat/completions")
    mock_err_http_response.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError(
        message=f"HTTP {status_code}", request=mock_err_http_response.request, response=mock_err_http_response
    ))
    mock_client.post.return_value = mock_err_http_response

    result = await adapter.execute("A failing Qwen prompt")
    assert isinstance(result, QwenStandardResponse)
    assert result.error is not None
    assert f"HTTP error {status_code}" in result.error
    assert expected_error_text_part in result.error
    assert result.raw_response == {"error_detail": json.dumps(error_json_response)}
    assert result.finish_reason == "error"


async def test_execute_request_error(adapter, auto_patch_httpx_client):
    mock_client = auto_patch_httpx_client.return_value
    error_msg = "Qwen connection totally failed"
    mock_client.post.side_effect = httpx.ConnectError(error_msg)
    result = await adapter.execute(TEST_PROMPT)
    assert isinstance(result, QwenStandardResponse)
    assert "Request error connecting to Qwen API" in result.error
    assert error_msg in result.error
    assert result.raw_response == {"error_detail": error_msg}


async def test_close_client(adapter, auto_patch_httpx_client):
    mock_client = auto_patch_httpx_client.return_value
    mock_client.is_closed = False
    await adapter.close()
    # With shared client manager, we don't close the client directly
    assert adapter._client is None
