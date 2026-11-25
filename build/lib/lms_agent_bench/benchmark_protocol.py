"""Strict OpenAI-compatible protocol measurements for benchmark trials."""
from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional

import requests
from requests.exceptions import RequestException

from lms_agent_bench import lms_eval as _lms_eval

sys.modules.setdefault("lms_eval", _lms_eval)
from lms_agent_bench import benchmark_lmstudio_cross_machine_models as _legacy


def _headers(api_key: Optional[str], accept: str) -> Dict[str, str]:
    headers = {"Accept": accept}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def call_chat_completions_stream(
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout_s: float,
    api_key: Optional[str],
) -> _legacy.CompletionMetrics:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    started = _legacy.now_s()
    ttft: Optional[float] = None
    output_parts: List[str] = []
    last_chunk: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    prompt_tokens = completion_tokens = total_tokens = None
    done_seen = False
    malformed_chunks = 0
    http_status: Optional[int] = None
    try:
        with requests.post(
            url,
            headers=_headers(api_key, "text/event-stream"),
            json=payload,
            stream=True,
            timeout=timeout_s,
        ) as response:
            http_status = response.status_code
            if response.status_code >= 400:
                body = response.text[:2000] if response.text else ""
                return _legacy.CompletionMetrics(
                    False,
                    response.status_code,
                    f"HTTP {response.status_code}: {body}",
                    "",
                    _legacy.now_s() - started,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            content_type = str(response.headers.get("content-type") or "").lower()
            if "text/event-stream" not in content_type:
                return _legacy.CompletionMetrics(
                    False,
                    response.status_code,
                    f"unexpected streaming content-type: {content_type or '<missing>'}",
                    "",
                    _legacy.now_s() - started,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                if not raw_line.startswith("data:"):
                    if raw_line.startswith(":"):
                        continue
                    malformed_chunks += 1
                    continue
                data = raw_line[5:].strip()
                if data == "[DONE]":
                    done_seen = True
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    malformed_chunks += 1
                    continue
                if not isinstance(chunk, dict):
                    malformed_chunks += 1
                    continue
                last_chunk = chunk
                choices = chunk.get("choices")
                if not isinstance(choices, list) or not choices:
                    if isinstance(chunk.get("usage"), dict):
                        usage = chunk["usage"]
                        prompt_tokens = usage.get("prompt_tokens")
                        completion_tokens = usage.get("completion_tokens")
                        total_tokens = usage.get("total_tokens")
                    continue
                choice = choices[0] if isinstance(choices[0], dict) else {}
                delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
                content = delta.get("content")
                if content is not None:
                    if not isinstance(content, str):
                        malformed_chunks += 1
                    elif content:
                        if ttft is None:
                            ttft = _legacy.now_s() - started
                        output_parts.append(content)
                if choice.get("finish_reason") is not None:
                    finish_reason = str(choice.get("finish_reason"))
                if isinstance(chunk.get("usage"), dict):
                    usage = chunk["usage"]
                    prompt_tokens = usage.get("prompt_tokens")
                    completion_tokens = usage.get("completion_tokens")
                    total_tokens = usage.get("total_tokens")
    except RequestException as exc:
        return _legacy.CompletionMetrics(
            False,
            http_status,
            str(exc),
            "".join(output_parts),
            _legacy.now_s() - started,
            ttft,
            None,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            None,
            finish_reason,
            last_chunk,
        )

    output = "".join(output_parts)
    wall = _legacy.now_s() - started
    if completion_tokens is None and output:
        completion_tokens = _legacy.approx_tokens_from_text(output)
    generation_s = wall - ttft if ttft is not None else None
    tokens_per_sec = (
        float(completion_tokens) / generation_s
        if completion_tokens and generation_s and generation_s > 0
        else None
    )
    protocol_errors = []
    if not done_seen:
        protocol_errors.append("stream ended without [DONE]")
    if not output:
        protocol_errors.append("stream produced no content")
    if malformed_chunks:
        protocol_errors.append(f"stream contained {malformed_chunks} malformed chunks")
    return _legacy.CompletionMetrics(
        not protocol_errors,
        http_status,
        "; ".join(protocol_errors) if protocol_errors else None,
        output,
        wall,
        ttft,
        None,
        prompt_tokens,
        completion_tokens,
        total_tokens,
        tokens_per_sec,
        finish_reason,
        last_chunk,
    )


def call_chat_completions_once(
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout_s: float,
    api_key: Optional[str],
) -> _legacy.CompletionMetrics:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    started = _legacy.now_s()
    try:
        response = requests.post(
            url,
            headers=_headers(api_key, "application/json"),
            json=payload,
            timeout=timeout_s,
        )
        wall = _legacy.now_s() - started
        if response.status_code >= 400:
            body = response.text[:2000] if response.text else ""
            return _legacy.CompletionMetrics(
                False,
                response.status_code,
                f"HTTP {response.status_code}: {body}",
                "",
                wall,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        try:
            data = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            return _legacy.CompletionMetrics(
                False,
                response.status_code,
                f"invalid JSON response: {exc}",
                "",
                wall,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return _legacy.CompletionMetrics(
                False,
                response.status_code,
                "response contains no valid choices",
                "",
                wall,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                data if isinstance(data, dict) else None,
            )
        choice = choices[0]
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        text = message.get("content")
        if not isinstance(text, str) or not text:
            return _legacy.CompletionMetrics(
                False,
                response.status_code,
                "response choice contains no text content",
                "" if text is None else str(text),
                wall,
                None,
                None,
                None,
                None,
                None,
                None,
                choice.get("finish_reason"),
                data,
            )
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens") or _legacy.approx_tokens_from_text(text)
        total_tokens = usage.get("total_tokens")
        tps = float(completion_tokens) / wall if completion_tokens and wall > 0 else None
        return _legacy.CompletionMetrics(
            True,
            response.status_code,
            None,
            text,
            wall,
            None,
            None,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            tps,
            choice.get("finish_reason"),
            data,
        )
    except RequestException as exc:
        return _legacy.CompletionMetrics(
            False,
            None,
            str(exc),
            "",
            _legacy.now_s() - started,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
