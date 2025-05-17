import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from promptdrifter.adapters.qwen import QwenAdapter
from promptdrifter.config.adapter_settings import (
    API_KEY_ENV_VAR_QWEN,
    DEFAULT_QWEN_MODEL,
    QWEN_API_BASE_URL,
)

TEST_API_KEY = "test_qwen_api_key"
TEST_PROMPT = "Hello, Qwen!"
TEST_MODEL = "qwen-test-model"
CUSTOM_BASE_URL = "http://localhost:8001"

SUCCESS_RESPONSE_PAYLOAD = {
    "request_id": "some-request-id",
    "output": {
        "text": None,
        "finish_reason": "stop",
        "message": {
            "role": "assistant",
            "content": "This is a test response from Qwen."
        }
    },
    "usage": {
        "output_tokens": 10,
        "input_tokens": 5
    }
}

API_ERROR_RESPONSE_PAYLOAD = {
    "request_id": "another-request-id",
    "code": "InvalidParameter",
    "message": "The input parameter 'model' is invalid."
}


@pytest.fixture
def mock_httpx_async_client_class(mocker):
    mock_class = mocker.patch("promptdrifter.adapters.qwen.httpx.AsyncClient", autospec=True)

    return mock_class


@pytest.fixture
def mock_os_getenv(mocker):
    return mocker.patch("os.getenv")

@pytest.mark.asyncio
async def test_qwen_adapter_init_with_direct_api_key(mock_httpx_async_client_class):
    adapter = QwenAdapter(api_key=TEST_API_KEY)
    assert adapter.config.api_key == TEST_API_KEY
    assert adapter.config.base_url == QWEN_API_BASE_URL
    assert adapter.config.default_model == DEFAULT_QWEN_MODEL
    mock_class = mock_httpx_async_client_class
    mock_class.assert_called_once_with(
        base_url=QWEN_API_BASE_URL,
        headers={
            "Authorization": f"Bearer {TEST_API_KEY}",
            "Content-Type": "application/json",
        }
    )
    mock_client_instance = mock_class.return_value
    await adapter.close()
    mock_client_instance.aclose.assert_called_once()

@pytest.mark.asyncio
async def test_qwen_adapter_init_with_qwen_env_var(mock_os_getenv, mock_httpx_async_client_class):
    mock_os_getenv.side_effect = lambda key, default=None: TEST_API_KEY if key == API_KEY_ENV_VAR_QWEN else (os.environ.get(key, default) if key != "DASHSCOPE_API_KEY" else None)
    adapter = QwenAdapter()
    assert adapter.config.api_key == TEST_API_KEY
    mock_os_getenv.assert_any_call(API_KEY_ENV_VAR_QWEN)
    mock_class = mock_httpx_async_client_class
    mock_class.assert_called_once()
    mock_client_instance = mock_class.return_value
    await adapter.close()
    mock_client_instance.aclose.assert_called_once()

@pytest.mark.asyncio
async def test_qwen_adapter_init_with_dashscope_env_var(mock_os_getenv, mock_httpx_async_client_class):
    mock_os_getenv.side_effect = lambda key, default=None: TEST_API_KEY if key == "DASHSCOPE_API_KEY" else (os.environ.get(key, default) if key != API_KEY_ENV_VAR_QWEN else None)
    adapter = QwenAdapter()
    assert adapter.config.api_key == TEST_API_KEY
    mock_os_getenv.assert_any_call("DASHSCOPE_API_KEY")
    mock_class = mock_httpx_async_client_class
    mock_class.assert_called_once()
    mock_client_instance = mock_class.return_value
    await adapter.close()
    mock_client_instance.aclose.assert_called_once()

@pytest.mark.asyncio
async def test_qwen_adapter_init_api_key_priority(mock_os_getenv, mock_httpx_async_client_class):
    mock_os_getenv.return_value = "env_api_key"
    adapter = QwenAdapter(api_key="direct_api_key")
    assert adapter.config.api_key == "direct_api_key"
    mock_class = mock_httpx_async_client_class
    mock_class.assert_called_once()
    mock_client_instance1 = mock_class.return_value
    await adapter.close()
    mock_client_instance1.aclose.assert_called_once()

    mock_class.reset_mock()

    mock_os_getenv.side_effect = lambda key, default=None: "qwen_env_key" if key == API_KEY_ENV_VAR_QWEN else ("dash_env_key" if key == "DASHSCOPE_API_KEY" else os.environ.get(key, default))
    adapter_2 = QwenAdapter()
    assert adapter_2.config.api_key == "qwen_env_key"
    mock_class.assert_called_once()
    mock_client_instance2 = mock_class.return_value
    await adapter_2.close()
    mock_client_instance2.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_qwen_adapter_init_missing_api_key(mock_os_getenv):
    mock_os_getenv.return_value = None
    with pytest.raises(ValueError) as exc_info:
        QwenAdapter()
    assert API_KEY_ENV_VAR_QWEN in str(exc_info.value)
    assert "DASHSCOPE_API_KEY" in str(exc_info.value)

@pytest.mark.asyncio
async def test_qwen_adapter_init_custom_url_and_model(mock_httpx_async_client_class):
    adapter = QwenAdapter(api_key=TEST_API_KEY, base_url=CUSTOM_BASE_URL, default_model="custom-default")
    assert adapter.config.base_url == CUSTOM_BASE_URL
    assert adapter.config.default_model == "custom-default"
    mock_class = mock_httpx_async_client_class
    mock_class.assert_called_once_with(
        base_url=CUSTOM_BASE_URL,
        headers={
            "Authorization": f"Bearer {TEST_API_KEY}",
            "Content-Type": "application/json",
        }
    )
    mock_client_instance = mock_class.return_value
    await adapter.close()
    mock_client_instance.aclose.assert_called_once()

@pytest.mark.asyncio
async def test_qwen_adapter_execute_successful(mock_httpx_async_client_class):
    adapter = QwenAdapter(api_key=TEST_API_KEY)
    mock_client_instance = mock_httpx_async_client_class.return_value

    mock_response = AsyncMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json = AsyncMock(return_value=SUCCESS_RESPONSE_PAYLOAD)
    mock_client_instance.post.return_value = mock_response

    result = await adapter.execute(prompt=TEST_PROMPT)

    expected_payload = {
        "model": DEFAULT_QWEN_MODEL,
        "input": {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": TEST_PROMPT}
            ]
        },
        "parameters": {"result_format": "message"}
    }
    mock_client_instance.post.assert_called_once_with(
        "/api/v1/services/aigc/text-generation/generation",
        json=expected_payload,
        timeout=180.0
    )
    assert result["text_response"] == SUCCESS_RESPONSE_PAYLOAD["output"]["message"]["content"]
    assert result["raw_response"] == SUCCESS_RESPONSE_PAYLOAD
    assert result["model_used"] == DEFAULT_QWEN_MODEL
    assert result["finish_reason"] == "stop"
    assert result["usage"] == SUCCESS_RESPONSE_PAYLOAD["usage"]
    assert "error" not in result
    await adapter.close()
    mock_client_instance.aclose.assert_called_once()

@pytest.mark.asyncio
async def test_qwen_adapter_execute_with_model_and_temp_and_kwargs(mock_httpx_async_client_class):
    adapter = QwenAdapter(api_key=TEST_API_KEY)
    mock_client_instance = mock_httpx_async_client_class.return_value

    mock_response = AsyncMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json = AsyncMock(return_value=SUCCESS_RESPONSE_PAYLOAD)
    mock_client_instance.post.return_value = mock_response

    custom_system_prompt = "You are a Qwen test bot."
    result = await adapter.execute(
        prompt=TEST_PROMPT,
        model=TEST_MODEL,
        temperature=0.5,
        system_prompt=custom_system_prompt,
        top_p=0.9,
        custom_param="value"
    )

    expected_payload = {
        "model": TEST_MODEL,
        "input": {
            "messages": [
                {"role": "system", "content": custom_system_prompt},
                {"role": "user", "content": TEST_PROMPT}
            ]
        },
        "parameters": {
            "result_format": "message",
            "temperature": 0.5,
            "top_p": 0.9,
            "custom_param": "value"
        }
    }
    mock_client_instance.post.assert_called_once_with(
        "/api/v1/services/aigc/text-generation/generation",
        json=expected_payload,
        timeout=180.0
    )
    assert result["model_used"] == TEST_MODEL
    await adapter.close()
    mock_client_instance.aclose.assert_called_once()

@pytest.mark.asyncio
async def test_qwen_adapter_execute_no_system_prompt(mock_httpx_async_client_class):
    adapter = QwenAdapter(api_key=TEST_API_KEY)
    mock_client_instance = mock_httpx_async_client_class.return_value

    mock_response = AsyncMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json = AsyncMock(return_value=SUCCESS_RESPONSE_PAYLOAD)
    mock_client_instance.post.return_value = mock_response

    await adapter.execute(prompt=TEST_PROMPT, system_prompt=None)
    called_json = mock_client_instance.post.call_args.kwargs["json"]
    assert len(called_json["input"]["messages"]) == 1
    assert called_json["input"]["messages"][0]["role"] == "user"
    await adapter.close()
    mock_client_instance.aclose.assert_called_once()

@pytest.mark.asyncio
async def test_qwen_adapter_execute_empty_kwargs_parameters_not_sent(mock_httpx_async_client_class):
    adapter = QwenAdapter(api_key=TEST_API_KEY)
    mock_client_instance = mock_httpx_async_client_class.return_value

    mock_response = AsyncMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json = AsyncMock(return_value=SUCCESS_RESPONSE_PAYLOAD)
    mock_client_instance.post.return_value = mock_response

    await adapter.execute(prompt=TEST_PROMPT)

    args, kwargs = mock_client_instance.post.call_args
    sent_payload = kwargs['json']
    assert "parameters" in sent_payload
    assert sent_payload["parameters"] == {"result_format": "message"}
    await adapter.close()
    mock_client_instance.aclose.assert_called_once()

@pytest.mark.asyncio
async def test_qwen_adapter_execute_api_error_in_json_response(mock_httpx_async_client_class):
    adapter = QwenAdapter(api_key=TEST_API_KEY)
    mock_client_instance = mock_httpx_async_client_class.return_value

    mock_response = AsyncMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json = AsyncMock(return_value=API_ERROR_RESPONSE_PAYLOAD)
    mock_response.raise_for_status = MagicMock()
    mock_client_instance.post.return_value = mock_response

    result = await adapter.execute(prompt=TEST_PROMPT)

    assert "error" in result
    assert API_ERROR_RESPONSE_PAYLOAD["code"] in result["error"]
    assert API_ERROR_RESPONSE_PAYLOAD["message"] in result["error"]
    assert result["raw_response"] == API_ERROR_RESPONSE_PAYLOAD
    assert result["text_response"] is None
    await adapter.close()
    mock_client_instance.aclose.assert_called_once()

@pytest.mark.asyncio
async def test_qwen_adapter_execute_http_status_error_json_body(mock_httpx_async_client_class):
    adapter = QwenAdapter(api_key=TEST_API_KEY)
    mock_client_instance = mock_httpx_async_client_class.return_value

    mock_response = AsyncMock(spec=httpx.Response)
    mock_response.status_code = 400
    mock_response.json = AsyncMock(return_value={"error_detail": "Bad request from JSON"})
    mock_response.text = "Fallback text error"
    mock_response.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError(
        "Error", request=AsyncMock(spec=httpx.Request), response=mock_response
    ))
    mock_client_instance.post.return_value = mock_response

    result = await adapter.execute(prompt=TEST_PROMPT)
    assert "error" in result
    assert "HTTP error 400" in result["error"]
    assert result["raw_response_error"] == {"error_detail": "Bad request from JSON"}
    await adapter.close()
    mock_client_instance.aclose.assert_called_once()

@pytest.mark.asyncio
async def test_qwen_adapter_execute_http_status_error_text_body(mock_httpx_async_client_class):
    adapter = QwenAdapter(api_key=TEST_API_KEY)
    mock_client_instance = mock_httpx_async_client_class.return_value

    mock_response = AsyncMock(spec=httpx.Response)
    mock_response.status_code = 500
    mock_response.json = AsyncMock(side_effect=Exception("JSON decode error"))
    mock_response.text = "Internal Server Error Text"
    mock_response.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError(
        "Error", request=AsyncMock(spec=httpx.Request), response=mock_response
    ))
    mock_client_instance.post.return_value = mock_response

    result = await adapter.execute(prompt=TEST_PROMPT)
    assert "error" in result
    assert "Failed to decode JSON response" in result["error"]
    assert f"(status {mock_response.status_code})" in result["error"]
    assert mock_response.text in result["error"]
    assert result["raw_response"] == "Internal Server Error Text"
    await adapter.close()
    mock_client_instance.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_qwen_adapter_execute_request_error(mock_httpx_async_client_class):
    adapter = QwenAdapter(api_key=TEST_API_KEY)
    mock_client_instance = mock_httpx_async_client_class.return_value
    mock_client_instance.post.side_effect = httpx.RequestError("Connection failed", request=None)

    result = await adapter.execute(prompt=TEST_PROMPT)
    assert "error" in result
    assert "Request error connecting to Qwen API: Connection failed" in result["error"]
    assert result["raw_response"] is None
    assert result["text_response"] is None
    await adapter.close()
    mock_client_instance.aclose.assert_called_once()

@pytest.mark.asyncio
async def test_qwen_adapter_execute_non_json_success_response(mock_httpx_async_client_class):
    adapter = QwenAdapter(api_key=TEST_API_KEY)
    mock_client_instance = mock_httpx_async_client_class.return_value

    mock_response = AsyncMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json = AsyncMock(side_effect=Exception("Cannot parse JSON"))
    mock_response.text = "This is not JSON but was a 200 OK"
    mock_client_instance.post.return_value = mock_response

    result = await adapter.execute(prompt=TEST_PROMPT)
    assert "error" in result
    assert "Failed to decode JSON response" in result["error"]
    assert "This is not JSON" in result["error"]
    assert result["raw_response"] == "This is not JSON but was a 200 OK"
    await adapter.close()
    mock_client_instance.aclose.assert_called_once()

@pytest.mark.asyncio
async def test_qwen_adapter_execute_unexpected_exception(mock_httpx_async_client_class):
    adapter = QwenAdapter(api_key=TEST_API_KEY)
    mock_client_instance = mock_httpx_async_client_class.return_value
    mock_client_instance.post.side_effect = Exception("Something totally unexpected")

    result = await adapter.execute(prompt=TEST_PROMPT)
    assert "error" in result
    assert "An unexpected error occurred with QwenAdapter: Something totally unexpected" in result["error"]
    await adapter.close()
    mock_client_instance.aclose.assert_called_once()

@pytest.mark.asyncio
async def test_qwen_adapter_close(mock_httpx_async_client_class):
    adapter = QwenAdapter(api_key=TEST_API_KEY)
    mock_client_instance = mock_httpx_async_client_class.return_value
    await adapter.close()
    mock_client_instance.aclose.assert_called_once()

@pytest.mark.asyncio
async def test_qwen_adapter_close_no_client():
    with patch.object(os, 'getenv', return_value=TEST_API_KEY):
        adapter = QwenAdapter(api_key=TEST_API_KEY)

    del adapter.client

    try:
        await adapter.close()
    except Exception as e:
        pytest.fail(f"adapter.close() raised an exception unexpectedly: {e}")

@pytest.mark.asyncio
async def test_qwen_output_parsing_variations(mock_httpx_async_client_class):
    adapter = QwenAdapter(api_key=TEST_API_KEY)
    mock_client_instance = mock_httpx_async_client_class.return_value

    response_missing_output = {"request_id": "r1", "usage": {"input_tokens": 1, "output_tokens": 0}}
    mock_response1 = AsyncMock(spec=httpx.Response)
    mock_response1.status_code = 200
    mock_response1.json = AsyncMock(return_value=response_missing_output)
    mock_client_instance.post.return_value = mock_response1
    result1 = await adapter.execute(prompt="test")
    assert result1["text_response"] is None
    assert result1["finish_reason"] is None

    response_missing_message = {"request_id": "r2", "output": {"finish_reason": "stop"}, "usage": {"input_tokens": 1, "output_tokens": 0}}
    mock_response2 = AsyncMock(spec=httpx.Response)
    mock_response2.status_code = 200
    mock_response2.json = AsyncMock(return_value=response_missing_message)
    mock_client_instance.post.return_value = mock_response2
    result2 = await adapter.execute(prompt="test")
    assert result2["text_response"] is None
    assert result2["finish_reason"] == "stop"

    response_missing_content = {"request_id": "r3", "output": {"message": {"role": "assistant"}, "finish_reason": "length"}, "usage": {"input_tokens": 1, "output_tokens": 0}}
    mock_response3 = AsyncMock(spec=httpx.Response)
    mock_response3.status_code = 200
    mock_response3.json = AsyncMock(return_value=response_missing_content)
    mock_client_instance.post.return_value = mock_response3
    result3 = await adapter.execute(prompt="test")
    assert result3["text_response"] is None
    assert result3["finish_reason"] == "length"

    await adapter.close()
    mock_client_instance.aclose.assert_called_once()
