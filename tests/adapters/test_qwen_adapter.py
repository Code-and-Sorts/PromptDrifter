import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from promptdrifter.adapters.qwen import QwenAdapter, QwenAdapterConfig
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


@pytest.fixture
def mock_httpx_async_client_class(mocker):
    mock_class = mocker.patch(
        "promptdrifter.adapters.qwen.httpx.AsyncClient", autospec=True
    )

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
        },
    )
    mock_client_instance = mock_class.return_value
    await adapter.close()
    mock_client_instance.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_qwen_adapter_init_with_qwen_env_var(
    mock_os_getenv, mock_httpx_async_client_class
):
    mock_os_getenv.side_effect = (
        lambda key, default=None: TEST_API_KEY
        if key == API_KEY_ENV_VAR_QWEN
        else (None if key == "DASHSCOPE_API_KEY" else os.environ.get(key, default))
    )
    adapter = QwenAdapter()
    assert adapter.config.api_key == TEST_API_KEY
    mock_os_getenv.assert_any_call(API_KEY_ENV_VAR_QWEN)
    mock_class = mock_httpx_async_client_class
    mock_class.assert_called_once_with(
        base_url=QWEN_API_BASE_URL,
        headers={
            "Authorization": f"Bearer {TEST_API_KEY}",
            "Content-Type": "application/json",
        },
    )
    mock_client_instance = mock_class.return_value
    await adapter.close()
    mock_client_instance.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_qwen_adapter_init_with_dashscope_env_var(
    mock_os_getenv, mock_httpx_async_client_class
):
    mock_os_getenv.side_effect = (
        lambda key, default=None: TEST_API_KEY
        if key == "DASHSCOPE_API_KEY"
        else (None if key == API_KEY_ENV_VAR_QWEN else os.environ.get(key, default))
    )
    adapter = QwenAdapter()
    assert adapter.config.api_key == TEST_API_KEY
    mock_os_getenv.assert_any_call("DASHSCOPE_API_KEY")
    mock_class = mock_httpx_async_client_class
    mock_class.assert_called_once_with(
        base_url=QWEN_API_BASE_URL,
        headers={
            "Authorization": f"Bearer {TEST_API_KEY}",
            "Content-Type": "application/json",
        },
    )
    mock_client_instance = mock_class.return_value
    await adapter.close()
    mock_client_instance.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_qwen_adapter_init_api_key_priority(
    mock_os_getenv, mock_httpx_async_client_class
):
    mock_os_getenv.return_value = "env_api_key"
    adapter = QwenAdapter(api_key="direct_api_key")
    assert adapter.config.api_key == "direct_api_key"
    mock_class = mock_httpx_async_client_class
    mock_class.assert_called_once_with(
        base_url=QWEN_API_BASE_URL,
        headers={
            "Authorization": "Bearer direct_api_key",
            "Content-Type": "application/json",
        },
    )
    mock_client_instance1 = mock_class.return_value
    await adapter.close()
    mock_client_instance1.aclose.assert_called_once()

    mock_class.reset_mock()

    mock_os_getenv.side_effect = (
        lambda key, default=None: "qwen_env_key"
        if key == API_KEY_ENV_VAR_QWEN
        else (
            "dash_env_key"
            if key == "DASHSCOPE_API_KEY"
            else os.environ.get(key, default)
        )
    )
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
    adapter = QwenAdapter(
        api_key=TEST_API_KEY, base_url=CUSTOM_BASE_URL, default_model="custom-default"
    )
    assert adapter.config.base_url == CUSTOM_BASE_URL
    assert adapter.config.default_model == "custom-default"
    mock_class = mock_httpx_async_client_class
    mock_class.assert_called_once_with(
        base_url=CUSTOM_BASE_URL,
        headers={
            "Authorization": f"Bearer {TEST_API_KEY}",
            "Content-Type": "application/json",
        },
    )
    mock_client_instance = mock_class.return_value
    await adapter.close()
    mock_client_instance.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_qwen_adapter_init_with_config_object(mock_httpx_async_client_class):
    config = QwenAdapterConfig(
        api_key="cfg_key", base_url="cfg_url", default_model="cfg_model"
    )
    adapter = QwenAdapter(config=config)
    assert adapter.config is config
    assert adapter.config.api_key == "cfg_key"
    assert adapter.config.base_url == "cfg_url"
    assert adapter.config.default_model == "cfg_model"
    mock_class = mock_httpx_async_client_class
    mock_class.assert_called_once_with(
        base_url="cfg_url",
        headers={"Authorization": "Bearer cfg_key", "Content-Type": "application/json"},
    )
    mock_client_instance = mock_class.return_value
    await adapter.close()
    mock_client_instance.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_qwen_adapter_execute_successful(mock_httpx_async_client_class):
    adapter = QwenAdapter(api_key=TEST_API_KEY)
    mock_client_instance = mock_httpx_async_client_class.return_value

    current_success_payload = SUCCESS_RESPONSE_PAYLOAD.copy()
    current_success_payload["model"] = adapter.config.default_model

    mock_response = AsyncMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json = MagicMock(return_value=current_success_payload)
    mock_response.raise_for_status = MagicMock()
    mock_client_instance.post.return_value = mock_response

    result = await adapter.execute(prompt=TEST_PROMPT)

    expected_payload = {
        "model": adapter.config.default_model,
        "messages": [
            {"role": "user", "content": TEST_PROMPT},
        ],
    }
    mock_client_instance.post.assert_called_once_with(
        "/chat/completions",
        json=expected_payload,
        timeout=180.0,
    )
    assert (
        result["text_response"]
        == current_success_payload["choices"][0]["message"]["content"]
    )
    assert result["raw_response"] == current_success_payload
    assert result["model_used"] == adapter.config.default_model
    assert (
        result["finish_reason"]
        == current_success_payload["choices"][0]["finish_reason"]
    )
    assert result["usage"] == current_success_payload["usage"]
    assert "error" not in result
    await adapter.close()
    mock_client_instance.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_qwen_adapter_execute_with_model_and_temp_and_kwargs(
    mock_httpx_async_client_class,
):
    adapter = QwenAdapter(api_key=TEST_API_KEY)
    mock_client_instance = mock_httpx_async_client_class.return_value

    current_success_payload = SUCCESS_RESPONSE_PAYLOAD.copy()
    current_success_payload["model"] = TEST_MODEL

    mock_response = AsyncMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json = MagicMock(return_value=current_success_payload)
    mock_response.raise_for_status = MagicMock()
    mock_client_instance.post.return_value = mock_response

    custom_system_prompt = "You are a Qwen test bot."
    test_max_tokens = 100

    result = await adapter.execute(
        prompt=TEST_PROMPT,
        model=TEST_MODEL,
        temperature=0.5,
        max_tokens=test_max_tokens,
        system_prompt=custom_system_prompt,
        top_p=0.9,
        custom_param="value",
    )

    expected_payload = {
        "model": TEST_MODEL,
        "messages": [
            {"role": "system", "content": custom_system_prompt},
            {"role": "user", "content": TEST_PROMPT},
        ],
        "temperature": 0.5,
        "max_tokens": test_max_tokens,
        "top_p": 0.9,
        "custom_param": "value",
    }
    mock_client_instance.post.assert_called_once_with(
        "/chat/completions",
        json=expected_payload,
        timeout=180.0,
    )
    assert result["model_used"] == TEST_MODEL
    assert (
        result["text_response"]
        == current_success_payload["choices"][0]["message"]["content"]
    )
    assert (
        result["finish_reason"]
        == current_success_payload["choices"][0]["finish_reason"]
    )
    await adapter.close()
    mock_client_instance.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_qwen_adapter_execute_no_system_prompt(mock_httpx_async_client_class):
    adapter = QwenAdapter(api_key=TEST_API_KEY)
    mock_client_instance = mock_httpx_async_client_class.return_value

    current_success_payload = SUCCESS_RESPONSE_PAYLOAD.copy()
    current_success_payload["model"] = adapter.config.default_model

    mock_response = AsyncMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json = MagicMock(return_value=current_success_payload)
    mock_response.raise_for_status = MagicMock()
    mock_client_instance.post.return_value = mock_response

    await adapter.execute(prompt=TEST_PROMPT, system_prompt=None)

    mock_client_instance.post.assert_called_once()
    args, kwargs_call = mock_client_instance.post.call_args
    assert args[0] == "/chat/completions"
    sent_payload = kwargs_call["json"]

    assert len(sent_payload["messages"]) == 1
    assert sent_payload["messages"][0]["role"] == "user"
    assert sent_payload["messages"][0]["content"] == TEST_PROMPT
    assert "temperature" not in sent_payload
    assert "max_tokens" not in sent_payload

    await adapter.close()
    mock_client_instance.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_qwen_adapter_execute_minimal_call(
    mock_httpx_async_client_class,
):
    adapter = QwenAdapter(api_key=TEST_API_KEY)
    mock_client_instance = mock_httpx_async_client_class.return_value

    current_success_payload = SUCCESS_RESPONSE_PAYLOAD.copy()
    current_success_payload["model"] = adapter.config.default_model

    mock_response = AsyncMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json = MagicMock(return_value=current_success_payload)
    mock_response.raise_for_status = MagicMock()
    mock_client_instance.post.return_value = mock_response

    await adapter.execute(prompt=TEST_PROMPT)

    mock_client_instance.post.assert_called_once()
    args, kwargs_call = mock_client_instance.post.call_args
    assert args[0] == "/chat/completions"
    sent_payload = kwargs_call["json"]

    expected_keys = {"model", "messages"}
    assert set(sent_payload.keys()) == expected_keys
    assert sent_payload["model"] == adapter.config.default_model
    assert len(sent_payload["messages"]) == 1
    assert sent_payload["messages"][0]["role"] == "user"
    assert sent_payload["messages"][0]["content"] == TEST_PROMPT

    await adapter.close()
    mock_client_instance.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_qwen_adapter_execute_api_error_in_json_response(
    mock_httpx_async_client_class,
):
    adapter = QwenAdapter(api_key=TEST_API_KEY)
    mock_client_instance = mock_httpx_async_client_class.return_value

    mock_response = AsyncMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json = MagicMock(return_value=API_ERROR_RESPONSE_PAYLOAD)
    mock_response.raise_for_status = MagicMock()
    mock_client_instance.post.return_value = mock_response

    result = await adapter.execute(prompt=TEST_PROMPT)

    assert "error" in result
    error_details = API_ERROR_RESPONSE_PAYLOAD["error"]
    assert error_details["type"] in result["error"]
    assert error_details["code"] in result["error"]
    assert error_details["message"] in result["error"]
    assert result["raw_response"] == API_ERROR_RESPONSE_PAYLOAD
    assert result["text_response"] is None
    await adapter.close()
    mock_client_instance.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_qwen_adapter_execute_http_status_error_json_body(
    mock_httpx_async_client_class,
):
    adapter = QwenAdapter(api_key=TEST_API_KEY)
    mock_client_instance = mock_httpx_async_client_class.return_value

    mock_error_response = AsyncMock(spec=httpx.Response)
    mock_error_response.status_code = 400
    mock_error_response.json = MagicMock(return_value=API_ERROR_RESPONSE_PAYLOAD)
    mock_error_response.text = (
        "This should be ignored if JSON parsing succeeds for error"
    )

    http_error = httpx.HTTPStatusError(
        message="Bad Request",
        request=AsyncMock(spec=httpx.Request),
        response=mock_error_response,
    )
    mock_client_instance.post.return_value = (
        mock_error_response
    )
    mock_error_response.raise_for_status = MagicMock(
        side_effect=http_error
    )

    result = await adapter.execute(prompt=TEST_PROMPT)
    assert "error" in result
    assert "HTTP error 400" in result["error"]

    error_details = API_ERROR_RESPONSE_PAYLOAD["error"]
    assert error_details["type"] in result["error"]
    assert error_details["code"] in result["error"]
    assert error_details["message"] in result["error"]
    assert result["raw_response_error"] == API_ERROR_RESPONSE_PAYLOAD
    await adapter.close()
    mock_client_instance.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_qwen_adapter_execute_http_status_error_text_body(
    mock_httpx_async_client_class,
):
    adapter = QwenAdapter(api_key=TEST_API_KEY)
    mock_client_instance = mock_httpx_async_client_class.return_value

    mock_error_response = AsyncMock(spec=httpx.Response)
    mock_error_response.status_code = 500
    mock_error_response.json = MagicMock(
        side_effect=json.JSONDecodeError("err", "doc", 0)
    )
    mock_error_response.text = "Server error text"

    http_error = httpx.HTTPStatusError(
        message="Server Error",
        request=AsyncMock(spec=httpx.Request),
        response=mock_error_response,
    )
    mock_client_instance.post.return_value = mock_error_response
    mock_error_response.raise_for_status = MagicMock(side_effect=http_error)

    result = await adapter.execute(prompt=TEST_PROMPT)

    mock_client_instance.post.assert_called_once()
    args, _ = mock_client_instance.post.call_args
    assert args[0] == "/chat/completions"

    assert "error" in result
    assert "HTTP error 500 from Qwen API: Server error text" == result["error"]
    assert result["raw_response_error"] == "Server error text"
    await adapter.close()
    mock_client_instance.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_qwen_adapter_execute_non_json_success_response(
    mock_httpx_async_client_class,
):
    adapter = QwenAdapter(api_key=TEST_API_KEY)
    mock_client_instance = mock_httpx_async_client_class.return_value

    mock_response = AsyncMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json = MagicMock(side_effect=json.JSONDecodeError("err", "doc", 0))
    mock_response.text = "This is not JSON but was a 200 OK"
    mock_response.raise_for_status = MagicMock()
    mock_client_instance.post.return_value = mock_response

    result = await adapter.execute(prompt=TEST_PROMPT)

    mock_client_instance.post.assert_called_once()
    args, _ = mock_client_instance.post.call_args
    assert args[0] == "/chat/completions"

    assert "error" in result
    assert "Failed to decode JSON response" in result["error"]
    assert f"(status {mock_response.status_code})" in result["error"]
    assert mock_response.text in result["error"]
    assert result["raw_response"] == "This is not JSON but was a 200 OK"
    assert result["text_response"] is None
    await adapter.close()
    mock_client_instance.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_qwen_adapter_execute_request_error(mock_httpx_async_client_class):
    adapter = QwenAdapter(api_key=TEST_API_KEY)
    mock_client_instance = mock_httpx_async_client_class.return_value
    mock_client_instance.post.side_effect = httpx.RequestError(
        "Connection failed", request=AsyncMock(spec=httpx.Request)
    )

    result = await adapter.execute(prompt=TEST_PROMPT)
    assert "error" in result
    assert "Request error connecting to Qwen API: Connection failed" in result["error"]
    assert result["raw_response"] is None
    assert result["text_response"] is None
    await adapter.close()
    mock_client_instance.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_qwen_adapter_execute_unexpected_exception(mock_httpx_async_client_class):
    adapter = QwenAdapter(api_key=TEST_API_KEY)
    mock_client_instance = mock_httpx_async_client_class.return_value
    mock_client_instance.post.side_effect = Exception("Something totally unexpected")

    result = await adapter.execute(prompt=TEST_PROMPT)
    assert "error" in result
    assert (
        "An unexpected error occurred with QwenAdapter: Something totally unexpected"
        in result["error"]
    )
    await adapter.close()
    mock_client_instance.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_qwen_adapter_close(mock_httpx_async_client_class):
    adapter = QwenAdapter(api_key=TEST_API_KEY)
    mock_client_instance = mock_httpx_async_client_class.return_value
    await adapter.close()
    mock_client_instance.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_qwen_adapter_close_no_client_graceful(mock_os_getenv):
    mock_os_getenv.return_value = None
    with pytest.raises(ValueError):
        QwenAdapter()

    with patch.object(os, "getenv", return_value=TEST_API_KEY):
        adapter_for_close_test = QwenAdapter()

    if hasattr(adapter_for_close_test, "client"):
        del adapter_for_close_test.client

    try:
        await adapter_for_close_test.close()
    except Exception as e:
        pytest.fail(f"adapter.close() raised an exception unexpectedly: {e}")


@pytest.mark.asyncio
async def test_qwen_output_parsing_variations(mock_httpx_async_client_class):
    adapter = QwenAdapter(api_key=TEST_API_KEY)
    mock_client_instance = mock_httpx_async_client_class.return_value

    response_empty_choices = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 123,
        "model": adapter.config.default_model,
        "choices": [],
        "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
    }
    mock_response1 = AsyncMock(spec=httpx.Response)
    mock_response1.status_code = 200
    mock_response1.json = MagicMock(return_value=response_empty_choices)
    mock_response1.raise_for_status = MagicMock()
    mock_client_instance.post.return_value = mock_response1
    result1 = await adapter.execute(prompt="test1")
    assert result1["text_response"] is None
    assert result1["finish_reason"] is None
    assert result1["raw_response"] == response_empty_choices

    response_choice_no_message = {
        "id": "chatcmpl-2",
        "object": "chat.completion",
        "created": 124,
        "model": adapter.config.default_model,
        "choices": [{"index": 0, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
    }
    mock_response2 = AsyncMock(spec=httpx.Response)
    mock_response2.status_code = 200
    mock_response2.json = MagicMock(return_value=response_choice_no_message)
    mock_response2.raise_for_status = MagicMock()
    mock_client_instance.post.return_value = mock_response2
    result2 = await adapter.execute(prompt="test2")
    assert result2["text_response"] is None
    assert result2["finish_reason"] == "stop"
    assert result2["raw_response"] == response_choice_no_message

    response_message_no_content = {
        "id": "chatcmpl-3",
        "object": "chat.completion",
        "created": 125,
        "model": adapter.config.default_model,
        "choices": [
            {"index": 0, "message": {"role": "assistant"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
    }
    mock_response3 = AsyncMock(spec=httpx.Response)
    mock_response3.status_code = 200
    mock_response3.json = MagicMock(return_value=response_message_no_content)
    mock_response3.raise_for_status = MagicMock()
    mock_client_instance.post.return_value = mock_response3
    result3 = await adapter.execute(prompt="test3")
    assert result3["text_response"] is None
    assert result3["finish_reason"] == "stop"
    assert result3["raw_response"] == response_message_no_content

    response_no_usage = {
        "id": "chatcmpl-4",
        "object": "chat.completion",
        "created": 126,
        "model": adapter.config.default_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "No usage info"},
                "finish_reason": "length",
            }
        ],
    }
    mock_response4 = AsyncMock(spec=httpx.Response)
    mock_response4.status_code = 200
    mock_response4.json = MagicMock(return_value=response_no_usage)
    mock_response4.raise_for_status = MagicMock()
    mock_client_instance.post.return_value = mock_response4
    result4 = await adapter.execute(prompt="test4")
    assert result4["text_response"] == "No usage info"
    assert result4["finish_reason"] == "length"
    assert result4["usage"] is None
    assert result4["raw_response"] == response_no_usage

    response_non_list_choices = {
        "id": "chatcmpl-5",
        "object": "chat.completion",
        "created": 127,
        "model": adapter.config.default_model,
        "choices": {"invalid": "data"},
        "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
    }
    mock_response5 = AsyncMock(spec=httpx.Response)
    mock_response5.status_code = 200
    mock_response5.json = MagicMock(return_value=response_non_list_choices)
    mock_response5.raise_for_status = MagicMock()
    mock_client_instance.post.return_value = mock_response5
    result5 = await adapter.execute(prompt="test5")
    assert result5["text_response"] is None
    assert result5["finish_reason"] is None
    assert result5["raw_response"] == response_non_list_choices

    response_non_dict_choice_message = {
        "id": "chatcmpl-6",
        "object": "chat.completion",
        "created": 128,
        "model": adapter.config.default_model,
        "choices": [{"index": 0, "message": "not_a_dict", "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
    }
    mock_response6 = AsyncMock(spec=httpx.Response)
    mock_response6.status_code = 200
    mock_response6.json = MagicMock(return_value=response_non_dict_choice_message)
    mock_response6.raise_for_status = MagicMock()
    mock_client_instance.post.return_value = mock_response6
    result6 = await adapter.execute(prompt="test6")
    assert result6["text_response"] is None
    assert result6["finish_reason"] == "stop"
    assert result6["raw_response"] == response_non_dict_choice_message

    await adapter.close()
    mock_client_instance.aclose.assert_called_once()
