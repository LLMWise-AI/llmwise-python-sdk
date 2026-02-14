from __future__ import annotations

from typing import Any, AsyncIterator, Iterator

import httpx

from ._util import bearer_headers, env, normalize_api_base
from .errors import LLMWiseError
from .sse import aiter_sse_json, iter_sse_json, _raise_for_status
from .types import JsonDict, Message, RoutingConfig


def _user_agent() -> str:
    # Keep this simple and stable so it shows up cleanly in provider logs.
    return "llmwise-python-sdk/0.1.0"


class LLMWise:
    """
    Synchronous client.

    Base URL defaults to `https://llmwise.ai/api/v1`.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
    ):
        self.api_key = api_key or env("LLMWISE_API_KEY")
        resolved_base = normalize_api_base(base_url or env("LLMWISE_BASE_URL") or "")

        if client is not None:
            self._client = client
            return

        headers = {"User-Agent": _user_agent(), **bearer_headers(self.api_key)}
        self._client = httpx.Client(
            base_url=resolved_base,
            timeout=timeout,
            headers=headers,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "LLMWise":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _post_json(self, path: str, payload: dict[str, Any]) -> JsonDict:
        resp = self._client.post(path, json=payload)
        _raise_for_status(resp)
        data = resp.json()
        if not isinstance(data, dict):
            raise LLMWiseError(status_code=502, message="Unexpected response shape", payload=data)
        return data

    def _get_json(self, path: str) -> Any:
        resp = self._client.get(path)
        _raise_for_status(resp)
        return resp.json()

    def chat(
        self,
        *,
        model: str,
        messages: list[Message],
        stream: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cost_saver: bool | None = None,
        optimization_goal: str | None = None,
        semantic_memory: bool | None = None,
        semantic_top_k: int | None = None,
        semantic_min_score: float | None = None,
        routing: RoutingConfig | None = None,
        conversation_id: str | None = None,
    ) -> JsonDict:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": bool(stream),
        }
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        if cost_saver is not None:
            payload["cost_saver"] = bool(cost_saver)
        if optimization_goal is not None:
            payload["optimization_goal"] = str(optimization_goal)
        if semantic_memory is not None:
            payload["semantic_memory"] = bool(semantic_memory)
        if semantic_top_k is not None:
            payload["semantic_top_k"] = int(semantic_top_k)
        if semantic_min_score is not None:
            payload["semantic_min_score"] = float(semantic_min_score)
        if routing is not None:
            payload["routing"] = routing
        if conversation_id is not None:
            payload["conversation_id"] = str(conversation_id)

        if stream:
            # Use `chat_stream()` for streaming.
            raise ValueError("Use chat_stream(...), not chat(stream=True)")

        return self._post_json("/chat", payload)

    def chat_stream(
        self,
        *,
        model: str,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
        cost_saver: bool | None = None,
        optimization_goal: str | None = None,
        semantic_memory: bool | None = None,
        semantic_top_k: int | None = None,
        semantic_min_score: float | None = None,
        routing: RoutingConfig | None = None,
        conversation_id: str | None = None,
    ) -> Iterator[JsonDict]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        if cost_saver is not None:
            payload["cost_saver"] = bool(cost_saver)
        if optimization_goal is not None:
            payload["optimization_goal"] = str(optimization_goal)
        if semantic_memory is not None:
            payload["semantic_memory"] = bool(semantic_memory)
        if semantic_top_k is not None:
            payload["semantic_top_k"] = int(semantic_top_k)
        if semantic_min_score is not None:
            payload["semantic_min_score"] = float(semantic_min_score)
        if routing is not None:
            payload["routing"] = routing
        if conversation_id is not None:
            payload["conversation_id"] = str(conversation_id)

        headers = {"Accept": "text/event-stream"}
        with self._client.stream("POST", "/chat", json=payload, headers=headers) as resp:
            yield from iter_sse_json(resp)

    def compare(
        self,
        *,
        models: list[str],
        messages: list[Message],
        stream: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        semantic_memory: bool | None = None,
        semantic_top_k: int | None = None,
        semantic_min_score: float | None = None,
        conversation_id: str | None = None,
    ) -> JsonDict:
        payload: dict[str, Any] = {
            "models": models,
            "messages": messages,
            "stream": bool(stream),
        }
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        if semantic_memory is not None:
            payload["semantic_memory"] = bool(semantic_memory)
        if semantic_top_k is not None:
            payload["semantic_top_k"] = int(semantic_top_k)
        if semantic_min_score is not None:
            payload["semantic_min_score"] = float(semantic_min_score)
        if conversation_id is not None:
            payload["conversation_id"] = str(conversation_id)

        if stream:
            raise ValueError("Use compare_stream(...), not compare(stream=True)")
        return self._post_json("/compare", payload)

    def compare_stream(
        self,
        *,
        models: list[str],
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
        semantic_memory: bool | None = None,
        semantic_top_k: int | None = None,
        semantic_min_score: float | None = None,
        conversation_id: str | None = None,
    ) -> Iterator[JsonDict]:
        payload: dict[str, Any] = {
            "models": models,
            "messages": messages,
            "stream": True,
        }
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        if semantic_memory is not None:
            payload["semantic_memory"] = bool(semantic_memory)
        if semantic_top_k is not None:
            payload["semantic_top_k"] = int(semantic_top_k)
        if semantic_min_score is not None:
            payload["semantic_min_score"] = float(semantic_min_score)
        if conversation_id is not None:
            payload["conversation_id"] = str(conversation_id)

        headers = {"Accept": "text/event-stream"}
        with self._client.stream("POST", "/compare", json=payload, headers=headers) as resp:
            yield from iter_sse_json(resp)

    def blend(
        self,
        *,
        models: list[str],
        messages: list[Message],
        stream: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        semantic_memory: bool | None = None,
        semantic_top_k: int | None = None,
        semantic_min_score: float | None = None,
        strategy: str | None = None,
        synthesizer: str | None = None,
        layers: int | None = None,
        samples: int | None = None,
        conversation_id: str | None = None,
    ) -> JsonDict:
        payload: dict[str, Any] = {
            "models": models,
            "messages": messages,
            "stream": bool(stream),
        }
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        if semantic_memory is not None:
            payload["semantic_memory"] = bool(semantic_memory)
        if semantic_top_k is not None:
            payload["semantic_top_k"] = int(semantic_top_k)
        if semantic_min_score is not None:
            payload["semantic_min_score"] = float(semantic_min_score)
        if strategy is not None:
            payload["strategy"] = str(strategy)
        if synthesizer is not None:
            payload["synthesizer"] = str(synthesizer)
        if layers is not None:
            payload["layers"] = int(layers)
        if samples is not None:
            payload["samples"] = int(samples)
        if conversation_id is not None:
            payload["conversation_id"] = str(conversation_id)

        if stream:
            raise ValueError("Use blend_stream(...), not blend(stream=True)")
        return self._post_json("/blend", payload)

    def blend_stream(
        self,
        *,
        models: list[str],
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
        semantic_memory: bool | None = None,
        semantic_top_k: int | None = None,
        semantic_min_score: float | None = None,
        strategy: str | None = None,
        synthesizer: str | None = None,
        layers: int | None = None,
        samples: int | None = None,
        conversation_id: str | None = None,
    ) -> Iterator[JsonDict]:
        payload: dict[str, Any] = {
            "models": models,
            "messages": messages,
            "stream": True,
        }
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        if semantic_memory is not None:
            payload["semantic_memory"] = bool(semantic_memory)
        if semantic_top_k is not None:
            payload["semantic_top_k"] = int(semantic_top_k)
        if semantic_min_score is not None:
            payload["semantic_min_score"] = float(semantic_min_score)
        if strategy is not None:
            payload["strategy"] = str(strategy)
        if synthesizer is not None:
            payload["synthesizer"] = str(synthesizer)
        if layers is not None:
            payload["layers"] = int(layers)
        if samples is not None:
            payload["samples"] = int(samples)
        if conversation_id is not None:
            payload["conversation_id"] = str(conversation_id)

        headers = {"Accept": "text/event-stream"}
        with self._client.stream("POST", "/blend", json=payload, headers=headers) as resp:
            yield from iter_sse_json(resp)

    def judge(
        self,
        *,
        contestants: list[str],
        judge: str,
        messages: list[Message],
        stream: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        semantic_memory: bool | None = None,
        semantic_top_k: int | None = None,
        semantic_min_score: float | None = None,
        criteria: list[str] | None = None,
        conversation_id: str | None = None,
    ) -> JsonDict:
        payload: dict[str, Any] = {
            "contestants": contestants,
            "judge": judge,
            "messages": messages,
            "stream": bool(stream),
        }
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        if semantic_memory is not None:
            payload["semantic_memory"] = bool(semantic_memory)
        if semantic_top_k is not None:
            payload["semantic_top_k"] = int(semantic_top_k)
        if semantic_min_score is not None:
            payload["semantic_min_score"] = float(semantic_min_score)
        if criteria is not None:
            payload["criteria"] = list(criteria)
        if conversation_id is not None:
            payload["conversation_id"] = str(conversation_id)

        if stream:
            raise ValueError("Use judge_stream(...), not judge(stream=True)")
        return self._post_json("/judge", payload)

    def judge_stream(
        self,
        *,
        contestants: list[str],
        judge: str,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
        semantic_memory: bool | None = None,
        semantic_top_k: int | None = None,
        semantic_min_score: float | None = None,
        criteria: list[str] | None = None,
        conversation_id: str | None = None,
    ) -> Iterator[JsonDict]:
        payload: dict[str, Any] = {
            "contestants": contestants,
            "judge": judge,
            "messages": messages,
            "stream": True,
        }
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        if semantic_memory is not None:
            payload["semantic_memory"] = bool(semantic_memory)
        if semantic_top_k is not None:
            payload["semantic_top_k"] = int(semantic_top_k)
        if semantic_min_score is not None:
            payload["semantic_min_score"] = float(semantic_min_score)
        if criteria is not None:
            payload["criteria"] = list(criteria)
        if conversation_id is not None:
            payload["conversation_id"] = str(conversation_id)

        headers = {"Accept": "text/event-stream"}
        with self._client.stream("POST", "/judge", json=payload, headers=headers) as resp:
            yield from iter_sse_json(resp)

    def models(self) -> list[dict[str, Any]]:
        data = self._get_json("/models")
        if not isinstance(data, list):
            raise LLMWiseError(status_code=502, message="Unexpected response shape", payload=data)
        return data

    def credits_balance(self) -> JsonDict:
        data = self._get_json("/credits/balance")
        if not isinstance(data, dict):
            raise LLMWiseError(status_code=502, message="Unexpected response shape", payload=data)
        return data

    def generate_api_key(self) -> JsonDict:
        # Requires Clerk JWT auth (dashboard), not an existing mm_sk_ key.
        return self._post_json("/keys/generate", {})


class AsyncLLMWise:
    """
    Asynchronous client.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ):
        self.api_key = api_key or env("LLMWISE_API_KEY")
        resolved_base = normalize_api_base(base_url or env("LLMWISE_BASE_URL") or "")

        if client is not None:
            self._client = client
            return

        headers = {"User-Agent": _user_agent(), **bearer_headers(self.api_key)}
        self._client = httpx.AsyncClient(
            base_url=resolved_base,
            timeout=timeout,
            headers=headers,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "AsyncLLMWise":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def _post_json(self, path: str, payload: dict[str, Any]) -> JsonDict:
        resp = await self._client.post(path, json=payload)
        _raise_for_status(resp)
        data = resp.json()
        if not isinstance(data, dict):
            raise LLMWiseError(status_code=502, message="Unexpected response shape", payload=data)
        return data

    async def _get_json(self, path: str) -> Any:
        resp = await self._client.get(path)
        _raise_for_status(resp)
        return resp.json()

    async def chat(
        self,
        *,
        model: str,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
        cost_saver: bool | None = None,
        optimization_goal: str | None = None,
        semantic_memory: bool | None = None,
        semantic_top_k: int | None = None,
        semantic_min_score: float | None = None,
        routing: RoutingConfig | None = None,
        conversation_id: str | None = None,
    ) -> JsonDict:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        if cost_saver is not None:
            payload["cost_saver"] = bool(cost_saver)
        if optimization_goal is not None:
            payload["optimization_goal"] = str(optimization_goal)
        if semantic_memory is not None:
            payload["semantic_memory"] = bool(semantic_memory)
        if semantic_top_k is not None:
            payload["semantic_top_k"] = int(semantic_top_k)
        if semantic_min_score is not None:
            payload["semantic_min_score"] = float(semantic_min_score)
        if routing is not None:
            payload["routing"] = routing
        if conversation_id is not None:
            payload["conversation_id"] = str(conversation_id)
        return await self._post_json("/chat", payload)

    async def chat_stream(
        self,
        *,
        model: str,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
        cost_saver: bool | None = None,
        optimization_goal: str | None = None,
        semantic_memory: bool | None = None,
        semantic_top_k: int | None = None,
        semantic_min_score: float | None = None,
        routing: RoutingConfig | None = None,
        conversation_id: str | None = None,
    ) -> AsyncIterator[JsonDict]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        if cost_saver is not None:
            payload["cost_saver"] = bool(cost_saver)
        if optimization_goal is not None:
            payload["optimization_goal"] = str(optimization_goal)
        if semantic_memory is not None:
            payload["semantic_memory"] = bool(semantic_memory)
        if semantic_top_k is not None:
            payload["semantic_top_k"] = int(semantic_top_k)
        if semantic_min_score is not None:
            payload["semantic_min_score"] = float(semantic_min_score)
        if routing is not None:
            payload["routing"] = routing
        if conversation_id is not None:
            payload["conversation_id"] = str(conversation_id)

        headers = {"Accept": "text/event-stream"}
        async with self._client.stream("POST", "/chat", json=payload, headers=headers) as resp:
            async for ev in aiter_sse_json(resp):
                yield ev

    async def models(self) -> list[dict[str, Any]]:
        data = await self._get_json("/models")
        if not isinstance(data, list):
            raise LLMWiseError(status_code=502, message="Unexpected response shape", payload=data)
        return data

