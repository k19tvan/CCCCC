"""Tests for FakeLLM, the OpenAI-compatible client (urlopen mocked) and JSON parsing.

No real network calls are made.
"""

from __future__ import annotations

import io
import json
import socket
import urllib.error
from typing import Any

import pytest

from agent_system.core.llm import client as client_module
from agent_system.core.llm.client import (
    FakeLLM,
    LLMAPIError,
    LLMConfigError,
    LLMConnectionError,
    LLMError,
    LLMMessage,
    LLMTimeoutError,
    OpenAICompatibleClient,
    create_llm_client,
    parse_json_object,
)

MESSAGES = [LLMMessage("system", "sys"), LLMMessage("user", "hi")]


class TestFakeLLM:
    def test_returns_scripted_replies_in_order(self):
        llm = FakeLLM(["one", {"k": 2}])

        assert llm.complete(MESSAGES).text == "one"
        assert json.loads(llm.complete(MESSAGES).text) == {"k": 2}
        assert llm.call_count == 2

    def test_repeats_last_reply_when_script_runs_out(self):
        llm = FakeLLM(["a", "b"])

        assert [llm.complete(MESSAGES).text for _ in range(4)] == ["a", "b", "b", "b"]

    def test_raises_scripted_exception(self):
        llm = FakeLLM([LLMTimeoutError("slow")])

        with pytest.raises(LLMTimeoutError):
            llm.complete(MESSAGES)
        assert llm.call_count == 1

    def test_callable_reply_receives_messages(self):
        llm = FakeLLM([lambda messages: {"n": len(messages)}])

        assert json.loads(llm.complete(MESSAGES).text) == {"n": 2}

    def test_records_messages(self):
        llm = FakeLLM(["x"])
        llm.complete(MESSAGES)

        assert llm.calls == [MESSAGES]

    def test_empty_script_rejected(self):
        with pytest.raises(ValueError):
            FakeLLM([])


class FakeHTTPResponse(io.BytesIO):
    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def chat_body(content: str, **extra: Any) -> bytes:
    data = {
        "model": "served-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "prompt_tokens_details": None},
    }
    data.update(extra)
    return json.dumps(data).encode()


@pytest.fixture
def urlopen(monkeypatch):
    """Replace urlopen; `urlopen.reply` is the bytes to return or an exception to raise."""

    class Stub:
        reply: Any = chat_body('{"ok": true}')
        requests: list[Any] = []
        timeouts: list[float] = []

        def __call__(self, request, timeout):
            self.requests.append(request)
            self.timeouts.append(timeout)
            if isinstance(self.reply, BaseException):
                raise self.reply
            return FakeHTTPResponse(self.reply)

    stub = Stub()
    stub.requests, stub.timeouts = [], []
    monkeypatch.setattr(client_module.urllib.request, "urlopen", stub)
    return stub


def make_client(**kwargs: Any) -> OpenAICompatibleClient:
    params = {"model": "gemma", "api_key": "sk-test", "base_url": "http://llm.local/v1/", "timeout": 7}
    params.update(kwargs)
    return OpenAICompatibleClient(**params)


class TestOpenAICompatibleClient:
    def test_success_returns_text_model_and_int_usage(self, urlopen):
        response = make_client().complete(MESSAGES)

        assert response.text == '{"ok": true}'
        assert response.model == "served-model"
        assert response.usage == {"prompt_tokens": 10, "completion_tokens": 5}

    def test_request_url_headers_and_payload(self, urlopen):
        make_client().complete(MESSAGES, response_schema={"type": "object"})

        request = urlopen.requests[0]
        assert request.full_url == "http://llm.local/v1/chat/completions"
        assert request.get_header("Authorization") == "Bearer sk-test"
        payload = json.loads(request.data)
        assert payload["model"] == "gemma"
        assert payload["messages"] == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        assert payload["temperature"] == 0.0
        assert payload["response_format"] == {"type": "json_object"}
        assert urlopen.timeouts == [7]

    def test_no_json_mode_without_schema(self, urlopen):
        make_client().complete(MESSAGES)

        assert "response_format" not in json.loads(urlopen.requests[0].data)

    def test_custom_base_url_without_key_sends_no_auth(self, urlopen):
        make_client(api_key="").complete(MESSAGES)

        assert urlopen.requests[0].get_header("Authorization") is None

    def test_missing_key_for_openai_is_config_error(self):
        with pytest.raises(LLMConfigError):
            OpenAICompatibleClient(model="gpt-4o-mini", api_key="")

    def test_http_error_is_api_error_with_status(self, urlopen):
        urlopen.reply = urllib.error.HTTPError(
            "http://llm.local", 429, "Too Many Requests", {}, io.BytesIO(b"rate limited")
        )

        with pytest.raises(LLMAPIError) as info:
            make_client().complete(MESSAGES)
        assert info.value.status_code == 429
        assert "rate limited" in str(info.value)
        assert info.value.kind == "api_error"

    @pytest.mark.parametrize(
        "error",
        [
            TimeoutError("timed out"),
            socket.timeout("timed out"),
            urllib.error.URLError(socket.timeout("timed out")),
        ],
    )
    def test_timeouts_are_timeout_errors(self, urlopen, error):
        urlopen.reply = error

        with pytest.raises(LLMTimeoutError) as info:
            make_client().complete(MESSAGES)
        assert info.value.kind == "timeout"

    def test_url_error_is_connection_error(self, urlopen):
        urlopen.reply = urllib.error.URLError(ConnectionRefusedError("refused"))

        with pytest.raises(LLMConnectionError):
            make_client().complete(MESSAGES)

    @pytest.mark.parametrize("body", [b"not json", b'{"choices": []}', b'{"error": "x"}'])
    def test_malformed_body_is_api_error(self, urlopen, body):
        urlopen.reply = body

        with pytest.raises(LLMAPIError):
            make_client().complete(MESSAGES)

    def test_null_content_becomes_empty_text(self, urlopen):
        urlopen.reply = json.dumps(
            {"choices": [{"message": {"role": "assistant", "content": None}}]}
        ).encode()

        assert make_client().complete(MESSAGES).text == ""

    def test_all_errors_share_base_class(self):
        for cls in (LLMAPIError, LLMConfigError, LLMConnectionError, LLMTimeoutError):
            assert issubclass(cls, LLMError)


class TestCreateLLMClient:
    def test_unknown_provider_is_config_error(self):
        with pytest.raises(LLMConfigError, match="not supported"):
            create_llm_client(provider="anthropic-xyz")

    def test_fake_provider_is_not_built_here(self):
        with pytest.raises(LLMConfigError):
            create_llm_client(provider="fake")

    def test_reads_base_url_and_timeout_from_env(self, monkeypatch, urlopen):
        monkeypatch.setenv("LLM_BASE_URL", "http://env.local/v1")
        monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "12.5")

        llm = create_llm_client(provider="OpenAI_Compatible", model="m", api_key="")
        llm.complete(MESSAGES)

        assert urlopen.requests[0].full_url == "http://env.local/v1/chat/completions"
        assert urlopen.timeouts == [12.5]
        assert llm.model == "m"

    def test_invalid_timeout_env_is_config_error(self, monkeypatch):
        monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "fast")

        with pytest.raises(LLMConfigError):
            create_llm_client(provider="openai", api_key="sk")


class TestParseJsonObject:
    @pytest.mark.parametrize(
        "text",
        [
            '{"a": 1}',
            '  {"a": 1}\n',
            '```json\n{"a": 1}\n```',
            '```\n{"a": 1}\n```',
            'Here is the result: {"a": 1} done.',
        ],
    )
    def test_extracts_object(self, text):
        assert parse_json_object(text) == {"a": 1}

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("", "empty output"),
            ("no json here", "no JSON object found"),
            ('{"a": 1,}', "invalid JSON syntax"),
            ("[1, 2]", "no JSON object found"),
        ],
    )
    def test_invalid_raises_value_error(self, text, message):
        with pytest.raises(ValueError, match=message):
            parse_json_object(text)
