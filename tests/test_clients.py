from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from conftest import EXAMPLES
from llm_eval_harness import AnthropicClient, OpenAIClient, load_client
from llm_eval_harness.io import load_callable


class FakeAnthropic:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", text="ignored"),
                SimpleNamespace(type="text", text="hello "),
                SimpleNamespace(type="text", text="world"),
            ]
        )


class FakeOpenAI:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])


def test_anthropic_client_sends_system_prompt_and_joins_text_blocks() -> None:
    fake = FakeAnthropic()
    client = AnthropicClient("some-model", client=fake)
    assert client.complete("be strict", "grade this") == "hello world"
    call = fake.calls[0]
    assert call["model"] == "some-model" and call["system"] == "be strict"
    assert call["messages"] == [{"role": "user", "content": "grade this"}]
    assert "temperature" not in call and "extra_body" not in call
    assert client.name == "anthropic:some-model"


def test_anthropic_temperature_goes_through_extra_body() -> None:
    # Recent SDK versions have no `temperature` keyword on messages.create.
    fake = FakeAnthropic()
    AnthropicClient("m", temperature=0.0, client=fake).complete("s", "u")
    assert fake.calls[0]["extra_body"] == {"temperature": 0.0}
    assert "temperature" not in fake.calls[0]


def test_openai_temperature_is_optional() -> None:
    fake = FakeOpenAI()
    OpenAIClient("m", client=fake).complete("s", "u")
    assert "temperature" not in fake.calls[0]
    OpenAIClient("m", temperature=0.2, client=fake).complete("s", "u")
    assert fake.calls[1]["temperature"] == 0.2


def test_openai_client_builds_chat_messages() -> None:
    fake = FakeOpenAI()
    assert OpenAIClient("gpt-x", client=fake).complete("sys", "usr") == "ok"
    assert fake.calls[0]["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]


def test_load_client_requires_a_model_for_hosted_providers() -> None:
    with pytest.raises(ValueError, match="--model"):
        load_client("anthropic")


def test_load_client_from_file_factory() -> None:
    client = load_client(f"{EXAMPLES / 'demo_judge.py'}:DemoJudgeClient")
    assert client.name == "heuristic-demo-judge"  # type: ignore[attr-defined]


def test_load_callable_errors_are_specific() -> None:
    with pytest.raises(ValueError, match="module:name"):
        load_callable("nocolon")
    with pytest.raises(FileNotFoundError):
        load_callable("missing.py:thing")
    with pytest.raises(AttributeError, match="nope"):
        load_callable(f"{EXAMPLES / 'toy_target.py'}:nope")
