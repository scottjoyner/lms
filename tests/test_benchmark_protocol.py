from __future__ import annotations

import json
from types import SimpleNamespace

from lms_agent_bench import benchmark_protocol


class FakeResponse:
    def __init__(
        self,
        *,
        status_code=200,
        headers=None,
        lines=None,
        payload=None,
        text="",
    ):
        self.status_code = status_code
        self.headers = headers or {}
        self._lines = lines or []
        self._payload = payload
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def iter_lines(self, decode_unicode=True):
        return iter(self._lines)

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def test_stream_requires_done_marker(monkeypatch):
    response = FakeResponse(
        headers={"content-type": "text/event-stream"},
        lines=[
            "data: "
            + json.dumps(
                {"choices": [{"delta": {"content": "READY"}}]}
            )
        ],
    )
    monkeypatch.setattr(
        benchmark_protocol.requests,
        "post",
        lambda *args, **kwargs: response,
    )
    metrics = benchmark_protocol.call_chat_completions_stream(
        "http://127.0.0.1:1234/v1",
        "model",
        [{"role": "user", "content": "test"}],
        8,
        0.0,
        1,
        None,
    )
    assert metrics.ok is False
    assert metrics.output_text == "READY"
    assert "without [DONE]" in str(metrics.error)


def test_stream_rejects_malformed_chunks(monkeypatch):
    response = FakeResponse(
        headers={"content-type": "text/event-stream; charset=utf-8"},
        lines=[
            "data: not-json",
            "data: "
            + json.dumps(
                {"choices": [{"delta": {"content": "READY"}}]}
            ),
            "data: [DONE]",
        ],
    )
    monkeypatch.setattr(
        benchmark_protocol.requests,
        "post",
        lambda *args, **kwargs: response,
    )
    metrics = benchmark_protocol.call_chat_completions_stream(
        "http://127.0.0.1:1234/v1",
        "model",
        [{"role": "user", "content": "test"}],
        8,
        0.0,
        1,
        None,
    )
    assert metrics.ok is False
    assert "malformed chunks" in str(metrics.error)


def test_stream_accepts_complete_valid_sse(monkeypatch):
    ticks = iter([0.0, 0.2, 1.2])
    monkeypatch.setattr(
        benchmark_protocol._legacy,
        "now_s",
        lambda: next(ticks),
    )
    response = FakeResponse(
        headers={"content-type": "text/event-stream"},
        lines=[
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {"content": "READY"},
                            "finish_reason": None,
                        }
                    ]
                }
            ),
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {"delta": {}, "finish_reason": "stop"}
                    ],
                    "usage": {"completion_tokens": 10},
                }
            ),
            "data: [DONE]",
        ],
    )
    monkeypatch.setattr(
        benchmark_protocol.requests,
        "post",
        lambda *args, **kwargs: response,
    )
    metrics = benchmark_protocol.call_chat_completions_stream(
        "http://127.0.0.1:1234/v1",
        "model",
        [{"role": "user", "content": "test"}],
        8,
        0.0,
        1,
        None,
    )
    assert metrics.ok is True
    assert metrics.output_text == "READY"
    assert metrics.ttft_s == 0.2
    assert metrics.wall_s == 1.2
    assert metrics.tokens_per_sec == 10.0
    assert metrics.finish_reason == "stop"


def test_stream_rejects_wrong_content_type(monkeypatch):
    response = FakeResponse(
        headers={"content-type": "application/json"},
        lines=[],
    )
    monkeypatch.setattr(
        benchmark_protocol.requests,
        "post",
        lambda *args, **kwargs: response,
    )
    metrics = benchmark_protocol.call_chat_completions_stream(
        "http://127.0.0.1:1234/v1",
        "model",
        [{"role": "user", "content": "test"}],
        8,
        0.0,
        1,
        None,
    )
    assert metrics.ok is False
    assert "content-type" in str(metrics.error)


def test_non_streaming_requires_text_choice(monkeypatch):
    response = FakeResponse(
        headers={"content-type": "application/json"},
        payload={"choices": [{"message": {"content": ""}}]},
    )
    monkeypatch.setattr(
        benchmark_protocol.requests,
        "post",
        lambda *args, **kwargs: response,
    )
    metrics = benchmark_protocol.call_chat_completions_once(
        "http://127.0.0.1:1234/v1",
        "model",
        [{"role": "user", "content": "test"}],
        8,
        0.0,
        1,
        None,
    )
    assert metrics.ok is False
    assert "no text content" in str(metrics.error)


def test_non_streaming_accepts_valid_completion(monkeypatch):
    response = FakeResponse(
        payload={
            "choices": [
                {
                    "message": {"content": "READY"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "total_tokens": 3,
            },
        }
    )
    monkeypatch.setattr(
        benchmark_protocol.requests,
        "post",
        lambda *args, **kwargs: response,
    )
    monkeypatch.setattr(
        benchmark_protocol._legacy,
        "now_s",
        SimpleNamespace(
            __call__=lambda self: 0.0
        ),
    )
    metrics = benchmark_protocol.call_chat_completions_once(
        "http://127.0.0.1:1234/v1",
        "model",
        [{"role": "user", "content": "test"}],
        8,
        0.0,
        1,
        None,
    )
    assert metrics.ok is True
    assert metrics.output_text == "READY"
    assert metrics.completion_tokens == 1
    assert metrics.finish_reason == "stop"
