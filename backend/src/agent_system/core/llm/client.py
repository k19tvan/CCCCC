"""Provider-independent LLM adapter and a deterministic FakeLLM for tests.

- The agent loop depends only on `LLMClient.complete(messages, response_schema)`.
- Infrastructure failures (network, API, timeout, config) raise `LLMError` or a subclass. These are
  distinct from schema-invalid output, which the agent loop handles with retries.
- No provider SDK is required: the real client calls an OpenAI-compatible chat completions API with
  the standard library; provider/model/base URL are set through environment variables.
"""

from __future__ import annotations

import json
import os
import re
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol, Sequence, Union

from agent_system.core.config import config

MessageRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class LLMMessage:
    """One message in a conversation sent to the LLM."""

    role: MessageRole
    content: str


@dataclass
class LLMResponse:
    """Raw LLM response; the agent loop parses and validates it."""

    text: str
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)


class LLMError(Exception):
    """Error while calling the LLM (not a schema error). `kind` is used in traces and messages."""

    kind = "llm_error"


class LLMTimeoutError(LLMError):
    """The LLM did not respond within the allowed time."""

    kind = "timeout"


class LLMConnectionError(LLMError):
    """Could not connect to the LLM API (DNS, network, connection refused, ...)."""

    kind = "connection_error"


class LLMAPIError(LLMError):
    """The LLM API returned an HTTP error or a malformed response."""

    kind = "api_error"

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMConfigError(LLMError):
    """Invalid LLM configuration (missing API key, unsupported provider, ...)."""

    kind = "config_error"


class LLMClient(Protocol):
    """LLM interface used by the agent loop."""

    name: str
    model: str

    def complete(
        self,
        messages: Sequence[LLMMessage],
        response_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """
        Send the conversation and return the text response.

        `response_schema` is the expected JSON schema of the output, if any. The client uses it to
        enable the provider's JSON mode; the agent loop still validates the output itself.

        Raises:
            LLMError: Network, API, timeout or configuration failure.
        """
        ...


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


class OpenAICompatibleClient:
    """
    Client for an OpenAI-style `/chat/completions` endpoint.

    Works with OpenAI and with providers/servers exposing a compatible API (set `base_url`).
    """

    name = "openai_compatible"

    def __init__(
        self,
        model: str,
        api_key: str = "",
        base_url: str = DEFAULT_OPENAI_BASE_URL,
        timeout: float = 60.0,
        temperature: float = 0.0,
    ) -> None:
        base_url = (base_url or DEFAULT_OPENAI_BASE_URL).rstrip("/")
        if not api_key and base_url == DEFAULT_OPENAI_BASE_URL:
            raise LLMConfigError(
                "LLM_API_KEY is missing for the OpenAI provider. "
                "Set the LLM_API_KEY environment variable or use LLM_PROVIDER=fake to run offline."
            )
        self.model = model
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._temperature = temperature

    def complete(
        self,
        messages: Sequence[LLMMessage],
        response_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self._temperature,
        }
        if response_schema is not None:
            # json_object is more widely supported than strict json_schema; the schema is already in the prompt.
            payload["response_format"] = {"type": "json_object"}

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        request = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise LLMAPIError(
                f"LLM API returned HTTP {exc.code}: {detail}", status_code=exc.code
            ) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise LLMTimeoutError(
                f"LLM did not respond within {self._timeout:g}s"
            ) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise LLMTimeoutError(
                    f"LLM did not respond within {self._timeout:g}s"
                ) from exc
            raise LLMConnectionError(f"Could not connect to the LLM API: {exc.reason}") from exc

        try:
            data = json.loads(body)
            text = data["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMAPIError(
                "LLM API response is not in chat completions format"
            ) from exc

        usage = {
            k: v for k, v in (data.get("usage") or {}).items() if isinstance(v, int)
        }
        return LLMResponse(text=text, model=str(data.get("model") or self.model), usage=usage)


FakeReply = Union[
    str,
    dict[str, Any],
    BaseException,
    Callable[[Sequence[LLMMessage]], Union[str, dict[str, Any]]],
]


class FakeLLM:
    """
    Fake LLM that returns scripted responses in order, for unit/integration tests and mechanism evals.

    Each item in `responses` can be:
    - `str`: returned verbatim (to simulate invalid JSON/schema output);
    - `dict`: returned as JSON;
    - `BaseException`: raised (e.g. `LLMTimeoutError(...)` to simulate a timeout);
    - a callable taking the messages and returning `str`/`dict`.

    Once the script is exhausted the last item is repeated, which suits "always schema-invalid"
    scenarios that exercise running out of retries.
    """

    name = "fake"

    def __init__(self, responses: Sequence[FakeReply], model: str = "fake-scripted") -> None:
        if not responses:
            raise ValueError("FakeLLM needs at least one scripted response")
        self.model = model
        self._responses = list(responses)
        self.calls: list[list[LLMMessage]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def complete(
        self,
        messages: Sequence[LLMMessage],
        response_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        index = min(len(self.calls), len(self._responses)) - 1
        reply = self._responses[index]

        if isinstance(reply, BaseException):
            raise reply
        if callable(reply):
            reply = reply(messages)
        text = reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
        return LLMResponse(text=text, model=self.model)


_CODE_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


def parse_json_object(text: str) -> dict[str, Any]:
    """
    Extract a JSON object from LLM output (a ```json code fence is accepted).

    Raises:
        ValueError: If there is no valid JSON object.
    """
    stripped = text.strip()
    fenced = _CODE_FENCE.match(stripped)
    if fenced:
        stripped = fenced.group(1).strip()
    if not stripped:
        raise ValueError("empty output")

    if not stripped.startswith("{"):
        start, end = stripped.find("{"), stripped.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object found in output")
        stripped = stripped[start : end + 1]

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON syntax ({exc.msg}, line {exc.lineno} column {exc.colno})") from exc
    if not isinstance(data, dict):
        raise ValueError(f"JSON must be an object, got {type(data).__name__}")
    return data


_OPENAI_COMPATIBLE_PROVIDERS = {"openai", "openai_compatible"}


def create_llm_client(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
) -> LLMClient:
    """
    Create a real LLM client from arguments or environment variables.

    Environment variables: LLM_PROVIDER, LLM_MODEL, LLM_API_KEY, LLM_BASE_URL, LLM_TIMEOUT_SECONDS.
    The offline `fake` provider is handled by the orchestrator, not created here.

    Raises:
        LLMConfigError: Unsupported provider or missing required configuration.
    """
    provider = (provider or config.LLM_PROVIDER).strip().lower()
    if provider not in _OPENAI_COMPATIBLE_PROVIDERS:
        raise LLMConfigError(
            f"LLM_PROVIDER '{provider}' is not supported. "
            f"Use one of: fake, {', '.join(sorted(_OPENAI_COMPATIBLE_PROVIDERS))}."
        )

    if timeout is None:
        try:
            timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
        except ValueError as exc:
            raise LLMConfigError("LLM_TIMEOUT_SECONDS must be a number") from exc

    return OpenAICompatibleClient(
        model=model or config.LLM_MODEL,
        api_key=config.LLM_API_KEY if api_key is None else api_key,
        base_url=base_url or os.getenv("LLM_BASE_URL", DEFAULT_OPENAI_BASE_URL),
        timeout=timeout,
    )
