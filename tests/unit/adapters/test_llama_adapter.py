import json
import os
from unittest.mock import (
    AsyncMock,
    MagicMock,
    patch,
)

import httpx
import pytest
from pydantic import ValidationError

from promptdrifter.adapters.llama import LlamaAdapter, LlamaAdapterConfig
from promptdrifter.config.adapter_settings import (
    API_KEY_ENV_VAR_LLAMA,
    DEFAULT_LLAMA_MODEL,
    LLAMA_API_BASE_URL,
)

pytestmark = pytest.mark.asyncio

@pytest.fixture(autouse=True)
def auto_patch_httpx_client():
    mock_client_instance = MagicMock(spec=httpx.AsyncClient)
    mock_client_instance.post = AsyncMock()
    mock_client_instance.aclose = AsyncMock()
    with patch("promptdrifter.adapters.llama.get_shared_client", return_value=mock_client_instance) as patched_client:
        yield patched_client

@pytest.fixture
def adapter_config_data():
    return LlamaAdapterConfig(
        api_key="test-llama-api-key",
        base_url=LLAMA_API_BASE_URL,
        default_model=DEFAULT_LLAMA_MODEL,
        max_tokens=2048
    )

@pytest.fixture
def adapter(adapter_config_data, monkeypatch, auto_patch_httpx_client):
    monkeypatch.setenv(API_KEY_ENV_VAR_LLAMA, adapter_config_data.api_key)
    adapter_instance = LlamaAdapter(config=adapter_config_data)
    return adapter_instance

@pytest.fixture
def mock_http_response():
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.json.return_value = {
        "choices": [
            {
                "message": {"content": "Test response"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"total_tokens": 10},
    }
    return response

@pytest.fixture
def mock_http_client(mock_http_response):
    with patch("promptdrifter.adapters.llama.get_shared_client") as mock_client:
        mock_client.return_value.post = AsyncMock(return_value=mock_http_response)
        yield mock_client

async def test_llama_adapter_init_with_direct_params(monkeypatch, auto_patch_httpx_client):
    monkeypatch.delenv(API_KEY_ENV_VAR_LLAMA, raising=False)
    config = LlamaAdapterConfig(
        api_key="direct_llama_key",
        base_url="custom_llama_url",
        default_model="custom_llama_model",
        max_tokens=1000
    )
    adapter_instance = LlamaAdapter(config=config)
    assert adapter_instance.config.api_key == "direct_llama_key"
    assert adapter_instance.config.base_url == "custom_llama_url"
    assert adapter_instance.config.default_model == "custom_llama_model"
    assert adapter_instance.config.max_tokens == 1000

async def test_llama_adapter_init_with_env_key_and_defaults(monkeypatch, auto_patch_httpx_client):
    monkeypatch.setenv(API_KEY_ENV_VAR_LLAMA, "env_llama_key_for_defaults")
    adapter_instance = LlamaAdapter()
    assert adapter_instance.config.api_key == "env_llama_key_for_defaults"
    assert adapter_instance.config.base_url == LLAMA_API_BASE_URL
    assert adapter_instance.config.default_model == DEFAULT_LLAMA_MODEL
    assert adapter_instance.config.max_tokens == LlamaAdapterConfig().max_tokens

async def test_llama_adapter_init_no_key_raises_error(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV_VAR_LLAMA, raising=False)
    with pytest.raises(ValidationError) as excinfo:
        LlamaAdapter()
    assert API_KEY_ENV_VAR_LLAMA in str(excinfo.value)

async def test_llama_adapter_init_with_config_object(monkeypatch, auto_patch_httpx_client):
    monkeypatch.delenv(API_KEY_ENV_VAR_LLAMA, raising=False)
    config = LlamaAdapterConfig(
        api_key="config_llama_key", base_url="config_llama_url",
        default_model="config_llama_model", max_tokens=500
    )
    adapter_instance = LlamaAdapter(config=config)
    assert adapter_instance.config is config

@pytest.fixture
def mock_llama_successful_response_data():
    return {
        "choices": [
            {"message": {"content": "Test response from Llama test!"}, "finish_reason": "stop"}
        ],
        "usage": {"total_tokens": 25, "prompt_tokens": 10, "completion_tokens": 15},
        "model": "llama-3-test-model" # Actual model returned by API
    }

async def test_execute_successful(adapter, auto_patch_httpx_client, mock_llama_successful_response_data):
    mock_client = auto_patch_httpx_client.return_value
    mock_http_response = MagicMock(spec=httpx.Response)
    mock_http_response.status_code = 200
    mock_http_response.json = MagicMock(return_value=mock_llama_successful_response_data)
    mock_http_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_http_response

    prompt = "Hello, Llama from test!"
    result = await adapter.execute(
        prompt,
        config_override=LlamaAdapterConfig(
            default_model="llama-3-8b-override",
            temperature=0.6,
            max_tokens=120,
            system_prompt="You are a Llama test bot.",
            api_key=adapter.config.api_key,
            base_url=adapter.config.base_url
        )
    )

    expected_payload = {
        "model": "llama-3-8b-override",
        "messages": [
            {"role": "system", "content": "You are a Llama test bot."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.6,
        "max_tokens": 120,
        "system": "You are a Llama test bot."
    }
    mock_client.post.assert_called_once_with(
        "/chat/completions",
        json=expected_payload,
        timeout=60.0,
    )
    assert result.text_response == "Test response from Llama test!"
    assert result.raw_response["choices"] == mock_llama_successful_response_data["choices"]
    assert result.raw_response["usage"] == mock_llama_successful_response_data["usage"]
    assert result.model_name == "llama-3-8b-override"
    assert result.finish_reason == "stop"
    assert result.usage == mock_llama_successful_response_data["usage"]
    assert result.error is None

async def test_execute_uses_config_defaults(adapter, auto_patch_httpx_client, mock_llama_successful_response_data):
    mock_client = auto_patch_httpx_client.return_value
    mock_http_response = MagicMock(spec=httpx.Response)
    mock_http_response.status_code = 200
    mock_llama_successful_response_data["model"] = adapter.config.default_model
    mock_http_response.json = MagicMock(return_value=mock_llama_successful_response_data)
    mock_http_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_http_response

    await adapter.execute("Test prompt for Llama defaults")

    _, call_kwargs = mock_client.post.call_args
    payload = call_kwargs["json"]
    assert payload["model"] == adapter.config.default_model
    assert payload["max_tokens"] == adapter.config.max_tokens

@pytest.mark.parametrize(
    "status_code, error_json, expected_error_part",
    [
        (400, {"error": {"message": "Llama bad request syntax"}}, "Llama bad request syntax"),
        (401, {"error": {"message": "Llama auth failed, bad key"}}, "Llama auth failed, bad key"),
        (503, {"error": {"message": "Llama service overloaded"}}, "Llama service overloaded"),
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

    result = await adapter.execute("A failing Llama prompt")

    assert result.error is not None
    assert f"API Error (HTTP {status_code})" in result.error
    assert expected_error_part in result.error
    assert result.raw_response == error_json
    assert result.text_response is None
    assert result.model_name == adapter.config.default_model
    assert result.finish_reason == "error"

async def test_execute_request_error(adapter, auto_patch_httpx_client):
    mock_client = auto_patch_httpx_client.return_value
    error_message = "Llama connect error test"
    mock_client.post.side_effect = httpx.ConnectError(error_message)
    result = await adapter.execute("A prompt for Llama connect error")
    assert result.error is not None
    assert "HTTP Client Error" in result.error
    assert error_message in result.error
    assert result.raw_response == {"error_detail": error_message}

async def test_close_client(adapter, auto_patch_httpx_client):
    await adapter.close()
    # With shared client manager, we don't close the client directly
    assert adapter._client is None

async def test_llama_adapter_execute_success(mock_http_client):
    config = LlamaAdapterConfig(api_key="test_key")
    adapter = LlamaAdapter(config=config)
    result = await adapter.execute("Test prompt")
    assert result.text_response == "Test response"
    assert result.finish_reason == "stop"
    assert result.model_name == DEFAULT_LLAMA_MODEL
    assert result.usage == {"total_tokens": 10}
    assert result.error is None

async def test_llama_adapter_execute_with_all_params(mock_http_client):
    config = LlamaAdapterConfig(api_key="test_key")
    adapter = LlamaAdapter(config=config)
    result = await adapter.execute(
        prompt="Test prompt",
        config_override=LlamaAdapterConfig(
            default_model="test-model",
            temperature=0.7,
            max_tokens=100,
            system_prompt="Test system prompt",
            api_key="test_key"
        )
    )
    assert result.text_response == "Test response"
    mock_http_client.return_value.post.assert_called_once()
    call_args = mock_http_client.return_value.post.call_args[1]
    assert call_args["json"]["model"] == "test-model"
    assert call_args["json"]["temperature"] == 0.7
    assert call_args["json"]["max_tokens"] == 100
    assert len(call_args["json"]["messages"]) == 2
    assert call_args["json"]["messages"][0]["role"] == "system"
    assert call_args["json"]["messages"][0]["content"] == "Test system prompt"

async def test_llama_adapter_execute_http_status_error(mock_http_client):
    mock_http_client.return_value.post.return_value.raise_for_status.side_effect = (
        httpx.HTTPStatusError(
            "400 Bad Request",
            request=MagicMock(),
            response=MagicMock(
                status_code=400,
                json=MagicMock(return_value={"error": {"message": "Invalid request"}}),
            ),
        )
    )
    config = LlamaAdapterConfig(api_key="test_key")
    adapter = LlamaAdapter(config=config)
    result = await adapter.execute("Test prompt")
    assert result.error == "API Error (HTTP 400): Invalid request"
    assert result.text_response is None
    assert result.finish_reason == "error"

async def test_llama_adapter_execute_request_error(mock_http_client):
    mock_http_client.return_value.post.side_effect = httpx.RequestError("Connection error")
    config = LlamaAdapterConfig(api_key="test_key")
    adapter = LlamaAdapter(config=config)
    result = await adapter.execute("Test prompt")
    assert result.error == "HTTP Client Error: RequestError - Connection error"
    assert result.text_response is None
    assert result.finish_reason == "error"

async def test_llama_adapter_execute_prompt_blocked(mock_http_client):
    mock_http_client.return_value.post.return_value.raise_for_status.side_effect = (
        httpx.HTTPStatusError(
            "400 Bad Request",
            request=MagicMock(),
            response=MagicMock(
                status_code=400,
                json=MagicMock(
                    return_value={
                        "error": {
                            "message": "The prompt was blocked by the content filter."
                        }
                    }
                ),
            ),
        )
    )
    config = LlamaAdapterConfig(api_key="test_key")
    adapter = LlamaAdapter(config=config)
    result = await adapter.execute("Test prompt")
    assert result.error == "API Error (HTTP 400): The prompt was blocked by the content filter."
    assert result.text_response is None
    assert result.finish_reason == "error"
