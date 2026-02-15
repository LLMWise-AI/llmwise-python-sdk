from __future__ import annotations

import json
from typing import Any, AsyncIterator, Iterator

import httpx

from .errors import LLMWiseError


def _raise_for_status(resp: httpx.Response) -> None:
    if resp.status_code < 400:
        return
    payload: Any | None = None
    try:
        payload = resp.json()
    except Exception:
        payload = resp.text
    message = "Request failed"
    if isinstance(payload, dict) and payload.get("error"):
        message = str(payload["error"])
    raise LLMWiseError(status_code=resp.status_code, message=message, payload=payload)


def iter_sse_json(resp: httpx.Response) -> Iterator[dict[str, Any]]:
    """
    Parse FastAPI EventSourceResponse output:
    lines like: `data: {...}` and sentinel `data: [DONE]`.
    """
    _raise_for_status(resp)

    for line in resp.iter_lines():
        if not line:
            continue
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data:
            continue
        if data == "[DONE]":
            return
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            # Ignore malformed chunks; caller can still get subsequent valid events.
            continue


async def aiter_sse_json(resp: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    _raise_for_status(resp)

    async for line in resp.aiter_lines():
        if not line:
            continue
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data:
            continue
        if data == "[DONE]":
            return
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            continue

