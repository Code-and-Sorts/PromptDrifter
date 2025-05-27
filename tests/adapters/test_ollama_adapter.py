import json

import httpx
import pytest

from promptdrifter.adapters.models import (
    OllamaRawResponse,
    OllamaResponse,
)
from promptdrifter.adapters.ollama import OllamaAdapter, OllamaAdapterConfig
from promptdrifter.config.adapter_settings import (
    DEFAULT_OLLAMA_BASE_URL as config_OLLAMA_BASE_URL,
)
from promptdrifter.config.adapter_settings import (
    DEFAULT_OLLAMA_MODEL as config_DEFAULT_OLLAMA_MODEL,
)

pytestmark = pytest.mark.asyncio

@pytest.fixture
def mock_response_non_streaming_data():
    return {
        "response": "Test response from Ollama",
        "done": True,
        "prompt_eval_count": 3,
        "eval_count": 7
    }

@pytest.fixture
def base_config_data():
    return OllamaAdapterConfig(
        base_url=config_OLLAMA_BASE_URL,
        default_model=config_DEFAULT_OLLAMA_MODEL,
        max_tokens=1024,
    )

@pytest.fixture
def adapter(base_config_data):
    config = base_config_data
    adapter_instance = OllamaAdapter(config=config)
    return adapter_instance

async def test_ollama_adapter_init_with_config_object():
    config = OllamaAdapterConfig(base_url="config_url", default_model="config_model", max_tokens=256)
    adapter_instance = OllamaAdapter(config=config)
    assert adapter_instance.config is config

async def test_ollama_adapter_init_with_defaults():
    adapter_instance = OllamaAdapter()
    assert adapter_instance.config.base_url == config_OLLAMA_BASE_URL
    assert adapter_instance.config.default_model == config_DEFAULT_OLLAMA_MODEL
    assert adapter_instance.config.max_tokens == 1024

async def test_execute_successful_non_streaming(adapter, respx_mock, mock_response_non_streaming_data):
    prompt = "Hello, Ollama!"
    config_override = OllamaAdapterConfig(default_model="llama2-override", max_tokens=50)
    route = respx_mock.post(f"{adapter.config.base_url}/api/generate").mock(
        return_value=httpx.Response(200, json=mock_response_non_streaming_data)
    )
    result = await adapter.execute(prompt, config_override=config_override, stream=False)
    assert route.called
    assert isinstance(result, OllamaResponse)
    assert result.text_response == "Test response from Ollama"
    expected_raw_response_obj = OllamaRawResponse.model_validate(mock_response_non_streaming_data)
    assert result.raw_response == expected_raw_response_obj.model_dump()
    assert result.model_name == "llama2-override"
    assert result.finish_reason == "completed"
    assert result.usage == {"total_tokens": 10, "prompt_tokens": 3, "completion_tokens": 7}
    assert result.error is None

async def test_execute_successful_streaming(adapter, respx_mock):
    prompt = "Stream test"
    config_override = OllamaAdapterConfig(default_model="stream-model")
    async def stream_content_generator():
        yield json.dumps({"response": "Stream part 1", "done": False}).encode() + b"\n"
        yield json.dumps({"response": " part 2", "done": True, "prompt_eval_count": 10, "eval_count": 5}).encode() + b"\n"
    route = respx_mock.post(f"{adapter.config.base_url}/api/generate").mock(
        return_value=httpx.Response(200, content=stream_content_generator())
    )
    result = await adapter.execute(prompt, config_override=config_override, stream=True)
    assert route.called
    assert isinstance(result, OllamaResponse)
    assert result.text_response == "Stream part 1 part 2"
    assert result.model_name == "stream-model"
    assert result.finish_reason == "completed"
    assert result.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    assert result.error is None
    assert "parts" in result.raw_response
    assert len(result.raw_response["parts"]) == 2
    assert "final_context" in result.raw_response

async def test_execute_uses_config_default_max_tokens(adapter, respx_mock, mock_response_non_streaming_data):
    prompt = "Test prompt"
    expected_payload_dict = adapter.config.get_payload(prompt, None, stream=False)
    assert expected_payload_dict["options"]["num_predict"] == adapter.config.max_tokens
    assert expected_payload_dict["model"] == adapter.config.default_model
    custom_mock_response_data = mock_response_non_streaming_data.copy()
    custom_mock_response_data["eval_count"] = adapter.config.max_tokens
    route = respx_mock.post(f"{adapter.config.base_url}/api/generate").mock(
        return_value=httpx.Response(200, json=custom_mock_response_data)
    )
    result = await adapter.execute(prompt)
    assert route.called
    assert isinstance(result, OllamaResponse)
    assert result.usage["completion_tokens"] == adapter.config.max_tokens

async def test_execute_http_status_error(adapter, respx_mock):
    error_response_dict = {"error": "Model not found"}
    url = f"{adapter.config.base_url}/api/generate"
    respx_mock.post(url).mock(return_value=httpx.Response(404, json=error_response_dict))
    result = await adapter.execute("A prompt")
    assert isinstance(result, OllamaResponse)
    assert result.error is not None
    assert "HTTP error 404" in result.error
    assert "Model not found" in result.error
    assert result.raw_response == error_response_dict
    assert result.text_response is None
    assert result.finish_reason == "error"

async def test_execute_request_error(adapter, respx_mock):
    connect_error = httpx.ConnectError("Connection failed")
    url = f"{adapter.config.base_url}/api/generate"
    respx_mock.post(url).mock(side_effect=connect_error)
    result = await adapter.execute("A prompt")
    assert isinstance(result, OllamaResponse)
    assert result.error is not None
    assert "Connection failed" in result.error
    assert result.raw_response == {"error_detail": "Connection failed"}
    assert result.text_response is None
    assert result.finish_reason == "error"

async def test_execute_timeout_error(adapter, respx_mock):
    timeout_error = httpx.ReadTimeout("Read operation timed out")
    url = f"{adapter.config.base_url}/api/generate"
    respx_mock.post(url).mock(side_effect=timeout_error)
    result = await adapter.execute("A prompt")
    assert isinstance(result, OllamaResponse)
    assert result.error is not None
    assert "Read operation timed out" in result.error
    assert result.raw_response == {"error_detail": "Read operation timed out"}
    assert result.text_response is None
    assert result.finish_reason == "error"

async def test_execute_unexpected_error(adapter, respx_mock):
    unexpected_exception = Exception("Something totally unexpected")
    url = f"{adapter.config.base_url}/api/generate"
    respx_mock.post(url).mock(side_effect=unexpected_exception)
    result = await adapter.execute("A prompt")
    assert isinstance(result, OllamaResponse)
    assert result.error is not None
    assert "Something totally unexpected" in result.error
    assert result.raw_response == {"error_detail": "Something totally unexpected"}
    assert result.text_response is None
    assert result.finish_reason == "error"

async def test_adapter_close_called(adapter):
    await adapter.close()
