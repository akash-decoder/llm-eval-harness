"""Model clients.

The harness only needs one thing from a model: text in, text out. Anything with a
`complete(system, user) -> str` method works, so adding a provider is a dozen lines.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from .io import load_callable


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


def client_name(client: object) -> str:
    return str(getattr(client, "name", type(client).__name__))


class FunctionClient:
    """Wrap a plain function. Handy for tests and quick experiments."""

    def __init__(self, fn: Callable[[str, str], str], name: str = "function") -> None:
        self._fn = fn
        self.name = name

    def complete(self, system: str, user: str) -> str:
        return self._fn(system, user)


class AnthropicClient:
    """Thin wrapper over the Anthropic Messages API (`pip install 'llm-eval-harness[anthropic]'`).

    Pass `client` to inject a pre-built SDK client (also how the tests fake it).
    `temperature` is left unset by default. Recent SDK versions do not take it as a keyword
    argument, so when you set it, it is sent through `extra_body`. Some models reject
    sampling parameters entirely; in that case leave it as None.
    """

    def __init__(
        self,
        model: str,
        *,
        temperature: float | None = None,
        max_tokens: int = 1024,
        client: Any = None,
    ) -> None:
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise ImportError("pip install 'llm-eval-harness[anthropic]'") from exc
            client = anthropic.Anthropic()
        self._client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.name = f"anthropic:{model}"

    def complete(self, system: str, user: str) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if self.temperature is not None:
            kwargs["extra_body"] = {"temperature": self.temperature}
        message = self._client.messages.create(**kwargs)
        return "".join(
            block.text for block in message.content if getattr(block, "type", None) == "text"
        )


class OpenAIClient:
    """Thin wrapper over OpenAI Chat Completions (`pip install 'llm-eval-harness[openai]'`)."""

    def __init__(
        self,
        model: str,
        *,
        temperature: float | None = None,
        client: Any = None,
    ) -> None:
        if client is None:
            try:
                import openai
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise ImportError("pip install 'llm-eval-harness[openai]'") from exc
            client = openai.OpenAI()
        self._client = client
        self.model = model
        self.temperature = temperature
        self.name = f"openai:{model}"

    def complete(self, system: str, user: str) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


def load_client(spec: str, model: str | None = None, temperature: float | None = None) -> LLMClient:
    """Build a client from a CLI-style spec.

    `anthropic` and `openai` need `model`. Anything else is read as `module:Factory`
    (or `file.py:Factory`), called with no arguments.
    """
    if spec in {"anthropic", "openai"}:
        if not model:
            raise ValueError(f"--model is required with --client {spec}")
        if spec == "anthropic":
            return AnthropicClient(model, temperature=temperature)
        return OpenAIClient(model, temperature=temperature)
    factory = load_callable(spec)
    client = factory()
    if not isinstance(client, LLMClient):
        raise TypeError(f"{spec}() did not return an object with complete(system, user)")
    return client
