from __future__ import annotations

from typing import Any, AsyncIterator, Iterator

import httpx

from importlib.metadata import PackageNotFoundError, version

from ._util import bearer_headers, env, normalize_api_base
from .errors import LLMWiseError
from .sse import aiter_sse_json, iter_sse_json, _raise_for_status
from .types import JsonDict, Message, RoutingConfig


def _user_agent() -> str:
    # Keep this simple and stable so it shows up cleanly in logs.
    try:
        v = version("llmwise")
    except PackageNotFoundError:  # pragma: no cover
        v = "0.0.0"
    return f"llmwise-python-sdk/{v}"


class LLMWise:
    """
    Synchronous client.

    Base URL defaults to `https://llmwise.ai/api/v1`.
    """

    def __init__(
        self,
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

    def _get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        resp = self._client.get(path, params=params)
        _raise_for_status(resp)
        return resp.json()

    def _patch_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        resp = self._client.patch(path, params=params)
        _raise_for_status(resp)
        data = resp.json()
        if not isinstance(data, dict):
            raise LLMWiseError(status_code=502, message="Unexpected response shape", payload=data)
        return data

    def _put_json(self, path: str, payload: dict[str, Any] | None = None) -> JsonDict:
        resp = self._client.put(path, json=payload or {})
        _raise_for_status(resp)
        data = resp.json()
        if not isinstance(data, dict):
            raise LLMWiseError(status_code=502, message="Unexpected response shape", payload=data)
        return data

    def _get_text(self, path: str, params: dict[str, Any] | None = None) -> str:
        resp = self._client.get(path, params=params)
        _raise_for_status(resp)
        return resp.text

    def _delete_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        resp = self._client.delete(path, params=params)
        _raise_for_status(resp)
        data = resp.json()
        if not isinstance(data, dict):
            raise LLMWiseError(status_code=502, message="Unexpected response shape", payload=data)
        return data

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

    def conversations(self, *, limit: int = 20, offset: int = 0) -> JsonDict:
        return self._get_json(
            "/conversations",
            {
                "limit": int(limit),
                "offset": int(offset),
            },
        )

    def get_conversation(self, conversation_id: str) -> JsonDict:
        return self._get_json(f"/conversations/{conversation_id}")

    def create_conversation(self) -> JsonDict:
        return self._post_json("/conversations", {})

    def update_conversation(self, conversation_id: str, *, title: str | None = None) -> JsonDict:
        params: dict[str, Any] = {}
        if title is not None:
            params["title"] = str(title)
        return self._patch_json(f"/conversations/{conversation_id}", params=params)

    def delete_conversation(self, conversation_id: str) -> JsonDict:
        return self._delete_json(f"/conversations/{conversation_id}")

    def history(self, *, limit: int = 20, offset: int = 0, mode: str | None = None, search: str | None = None) -> JsonDict:
        params: dict[str, Any] = {
            "limit": int(limit),
            "offset": int(offset),
        }
        if mode is not None:
            params["mode"] = mode
        if search is not None:
            params["search"] = search
        return self._get_json("/history", params=params)

    def get_history_detail(self, request_id: str) -> JsonDict:
        return self._get_json(f"/history/{request_id}")

    def credits_wallet(self) -> JsonDict:
        return self._get_json("/credits/wallet")

    def credits_transactions(self, *, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        return self._get_json(
            "/credits/transactions",
            {"limit": int(limit), "offset": int(offset)},
        )

    def credits_packs(self) -> list[dict[str, Any]]:
        return self._get_json("/credits/packs")

    def credits_purchase(
        self,
        amount_usd: float | None = None,
        pack_id: str | None = None,
        **tracking: str,
    ) -> JsonDict:
        payload: dict[str, Any] = {}
        if amount_usd is not None:
            payload["amount_usd"] = float(amount_usd)
        if pack_id is not None:
            payload["pack_id"] = str(pack_id)
        for key, value in tracking.items():
            payload[key] = value
        return self._post_json("/credits/purchase", payload)

    def credits_confirm_checkout(self, *, session_id: str) -> JsonDict:
        return self._post_json("/credits/confirm-checkout", {"session_id": str(session_id)})

    def credits_update_auto_topup(
        self,
        *,
        enabled: bool,
        threshold_credits: int = 300,
        amount_usd: int = 10,
        monthly_cap_usd: int = 200,
    ) -> JsonDict:
        payload = {
            "enabled": bool(enabled),
            "threshold_credits": int(threshold_credits),
            "amount_usd": int(amount_usd),
            "monthly_cap_usd": int(monthly_cap_usd),
        }
        return self._put_json("/credits/auto-topup", payload)

    def usage_summary(self, *, days: int = 7) -> JsonDict:
        return self._get_json("/usage/summary", {"days": int(days)})

    def usage_recent(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        days: int = 30,
        mode: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "limit": int(limit),
            "offset": int(offset),
            "days": int(days),
        }
        if mode is not None:
            params["mode"] = mode
        return self._get_json("/usage/recent", params=params)

    def keys_info(self) -> JsonDict:
        return self._get_json("/keys/info")

    def revoke_api_key(self) -> JsonDict:
        return self._delete_json("/keys/revoke")

    def memory_list(self, *, limit: int = 20) -> JsonDict:
        return self._get_json("/memory", {"limit": int(limit)})

    def memory_search(
        self,
        *,
        q: str,
        top_k: int = 4,
        min_score: float | None = None,
    ) -> JsonDict:
        params = {
            "q": str(q),
            "top_k": int(top_k),
        }
        if min_score is not None:
            params["min_score"] = float(min_score)
        return self._get_json("/memory/search", params=params)

    def memory_delete(self, memory_id: str) -> JsonDict:
        return self._delete_json(f"/memory/{memory_id}")

    def memory_clear(self) -> JsonDict:
        return self._delete_json("/memory")

    def optimization_policy(self) -> JsonDict:
        return self._get_json("/optimization/policy")

    def optimization_update_policy(self, payload: dict[str, Any]) -> JsonDict:
        return self._put_json("/optimization/policy", payload)

    def optimization_report(
        self,
        *,
        goal: str = "balanced",
        days: int = 30,
        min_calls_per_model: int = 3,
        use_policy: bool = True,
        persist_snapshot: bool = False,
    ) -> JsonDict:
        return self._get_json(
            "/optimization/report",
            {
                "goal": goal,
                "days": int(days),
                "min_calls_per_model": int(min_calls_per_model),
                "use_policy": bool(use_policy),
                "persist_snapshot": bool(persist_snapshot),
            },
        )

    def optimization_evaluate(self) -> JsonDict:
        return self._post_json("/optimization/evaluate", {})

    def optimization_snapshots(self, *, goal: str | None = None, limit: int = 20) -> JsonDict:
        params: dict[str, Any] = {"limit": int(limit)}
        if goal is not None:
            params["goal"] = goal
        return self._get_json("/optimization/snapshots", params=params)

    def optimization_alerts(self, *, limit: int = 12) -> JsonDict:
        return self._get_json("/optimization/alerts", {"limit": int(limit)})

    def optimization_replay(self, *, days: int = 30, sample_size: int = 100) -> JsonDict:
        return self._post_json(
            "/optimization/replay",
            {
                "days": int(days),
                "sample_size": int(sample_size),
            },
        )

    def optimization_test_templates(self) -> JsonDict:
        return self._get_json("/optimization/test-templates")

    def optimization_test_suites(self, *, limit: int = 25) -> JsonDict:
        return self._get_json("/optimization/test-suites", {"limit": int(limit)})

    def optimization_create_test_suite(self, payload: dict[str, Any]) -> JsonDict:
        return self._post_json("/optimization/test-suites", payload)

    def optimization_update_test_suite(self, suite_id: str, payload: dict[str, Any]) -> JsonDict:
        return self._put_json(f"/optimization/test-suites/{suite_id}", payload)

    def optimization_run_test_suite(self, suite_id: str) -> JsonDict:
        return self._post_json(f"/optimization/test-suites/{suite_id}/run", {})

    def optimization_test_runs(self, *, limit: int = 20) -> JsonDict:
        return self._get_json("/optimization/test-runs", {"limit": int(limit)})

    def optimization_test_run_csv(self, run_id: str) -> str:
        return self._get_text(f"/optimization/test-runs/{run_id}/csv")

    def optimization_regression_schedules(self, *, limit: int = 30) -> JsonDict:
        return self._get_json("/optimization/regression-schedules", {"limit": int(limit)})

    def optimization_create_regression_schedule(self, payload: dict[str, Any]) -> JsonDict:
        return self._post_json("/optimization/regression-schedules", payload)

    def optimization_update_regression_schedule(self, schedule_id: str, payload: dict[str, Any]) -> JsonDict:
        return self._put_json(f"/optimization/regression-schedules/{schedule_id}", payload)

    def optimization_run_regression_schedule(self, schedule_id: str) -> JsonDict:
        return self._post_json(f"/optimization/regression-schedules/{schedule_id}/run", {})

    def credits_balance(self) -> JsonDict:
        data = self._get_json("/credits/balance")
        if not isinstance(data, dict):
            raise LLMWiseError(status_code=502, message="Unexpected response shape", payload=data)
        return data

    def generate_api_key(self) -> JsonDict:
        # Requires Clerk JWT auth (dashboard), not an existing mm_sk_ key.
        return self._post_json("/keys/generate", {})

    def settings_keys(self) -> JsonDict:
        return self._get_json("/settings/keys")

    def settings_save_keys(self, keys: dict[str, str]) -> JsonDict:
        return self._put_json("/settings/keys", {"keys": keys})

    def settings_delete_key(self, provider: str) -> JsonDict:
        return self._delete_json(f"/settings/keys/{provider}")

    def settings_privacy(self) -> JsonDict:
        return self._get_json("/settings/privacy")

    def settings_update_privacy(
        self,
        *,
        data_training_opt_in: bool | None = None,
        zero_retention_mode: bool | None = None,
        purge_existing_data: bool = True,
    ) -> JsonDict:
        payload: dict[str, Any] = {
            "purge_existing_data": bool(purge_existing_data),
        }
        if data_training_opt_in is not None:
            payload["data_training_opt_in"] = bool(data_training_opt_in)
        if zero_retention_mode is not None:
            payload["zero_retention_mode"] = bool(zero_retention_mode)
        return self._put_json("/settings/privacy", payload)

    def settings_copilot_state(self) -> JsonDict:
        return self._get_json("/settings/copilot")

    def settings_update_copilot(
        self,
        *,
        goal: str | None = None,
        onboarded: bool | None = None,
        checklist: dict[str, bool] | None = None,
    ) -> JsonDict:
        payload: dict[str, Any] = {}
        if goal is not None:
            payload["goal"] = str(goal)
        if onboarded is not None:
            payload["onboarded"] = bool(onboarded)
        if checklist is not None:
            payload["checklist"] = dict(checklist)
        return self._put_json("/settings/copilot", payload)

    def settings_ask_copilot(
        self,
        *,
        question: str,
        path: str | None = None,
        context: dict[str, str] | None = None,
    ) -> JsonDict:
        payload: dict[str, Any] = {
            "question": question,
        }
        if path is not None:
            payload["path"] = str(path)
        if context is not None:
            payload["context"] = dict(context)
        return self._post_json("/settings/copilot/ask", payload)


class AsyncLLMWise:
    """
    Asynchronous client.
    """

    def __init__(
        self,
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

    async def _get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        resp = await self._client.get(path, params=params)
        _raise_for_status(resp)
        return resp.json()

    async def _patch_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        resp = await self._client.patch(path, params=params)
        _raise_for_status(resp)
        data = resp.json()
        if not isinstance(data, dict):
            raise LLMWiseError(status_code=502, message="Unexpected response shape", payload=data)
        return data

    async def _delete_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        resp = await self._client.delete(path, params=params)
        _raise_for_status(resp)
        data = resp.json()
        if not isinstance(data, dict):
            raise LLMWiseError(status_code=502, message="Unexpected response shape", payload=data)
        return data

    async def _put_json(self, path: str, payload: dict[str, Any] | None = None) -> Any:
        resp = await self._client.put(path, json=payload or {})
        _raise_for_status(resp)
        data = resp.json()
        if not isinstance(data, dict):
            raise LLMWiseError(status_code=502, message="Unexpected response shape", payload=data)
        return data

    async def _get_text(self, path: str, params: dict[str, Any] | None = None) -> str:
        resp = await self._client.get(path, params=params)
        _raise_for_status(resp)
        return resp.text

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

    async def compare(
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
        stream: bool = False,
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
        return await self._post_json("/compare", payload)

    async def compare_stream(
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
    ) -> AsyncIterator[JsonDict]:
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
        async with self._client.stream("POST", "/compare", json=payload, headers=headers) as resp:
            async for ev in aiter_sse_json(resp):
                yield ev

    async def blend(
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
        stream: bool = False,
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
        return await self._post_json("/blend", payload)

    async def blend_stream(
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
    ) -> AsyncIterator[JsonDict]:
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
        async with self._client.stream("POST", "/blend", json=payload, headers=headers) as resp:
            async for ev in aiter_sse_json(resp):
                yield ev

    async def judge(
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
        stream: bool = False,
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
        return await self._post_json("/judge", payload)

    async def judge_stream(
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
    ) -> AsyncIterator[JsonDict]:
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
        async with self._client.stream("POST", "/judge", json=payload, headers=headers) as resp:
            async for ev in aiter_sse_json(resp):
                yield ev

    async def models(self) -> list[dict[str, Any]]:
        data = await self._get_json("/models")
        if not isinstance(data, list):
            raise LLMWiseError(status_code=502, message="Unexpected response shape", payload=data)
        return data

    async def conversations(self, *, limit: int = 20, offset: int = 0) -> JsonDict:
        return await self._get_json(
            "/conversations",
            {
                "limit": int(limit),
                "offset": int(offset),
            },
        )

    async def get_conversation(self, conversation_id: str) -> JsonDict:
        return await self._get_json(f"/conversations/{conversation_id}")

    async def create_conversation(self) -> JsonDict:
        return await self._post_json("/conversations", {})

    async def update_conversation(self, conversation_id: str, *, title: str | None = None) -> JsonDict:
        params: dict[str, Any] = {}
        if title is not None:
            params["title"] = str(title)
        return await self._patch_json(f"/conversations/{conversation_id}", params=params)

    async def delete_conversation(self, conversation_id: str) -> JsonDict:
        return await self._delete_json(f"/conversations/{conversation_id}")

    async def history(self, *, limit: int = 20, offset: int = 0, mode: str | None = None, search: str | None = None) -> JsonDict:
        params: dict[str, Any] = {
            "limit": int(limit),
            "offset": int(offset),
        }
        if mode is not None:
            params["mode"] = mode
        if search is not None:
            params["search"] = search
        return await self._get_json("/history", params=params)

    async def get_history_detail(self, request_id: str) -> JsonDict:
        return await self._get_json(f"/history/{request_id}")

    async def credits_balance(self) -> JsonDict:
        return await self._get_json("/credits/balance")

    async def credits_wallet(self) -> JsonDict:
        return await self._get_json("/credits/wallet")

    async def credits_transactions(self, *, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        return await self._get_json(
            "/credits/transactions",
            {"limit": int(limit), "offset": int(offset)},
        )

    async def credits_packs(self) -> list[dict[str, Any]]:
        return await self._get_json("/credits/packs")

    async def credits_purchase(
        self,
        amount_usd: float | None = None,
        pack_id: str | None = None,
        **tracking: str,
    ) -> JsonDict:
        payload: dict[str, Any] = {}
        if amount_usd is not None:
            payload["amount_usd"] = float(amount_usd)
        if pack_id is not None:
            payload["pack_id"] = str(pack_id)
        for key, value in tracking.items():
            payload[key] = value
        return await self._post_json("/credits/purchase", payload)

    async def credits_confirm_checkout(self, *, session_id: str) -> JsonDict:
        return await self._post_json("/credits/confirm-checkout", {"session_id": str(session_id)})

    async def credits_update_auto_topup(
        self,
        *,
        enabled: bool,
        threshold_credits: int = 300,
        amount_usd: int = 10,
        monthly_cap_usd: int = 200,
    ) -> JsonDict:
        payload = {
            "enabled": bool(enabled),
            "threshold_credits": int(threshold_credits),
            "amount_usd": int(amount_usd),
            "monthly_cap_usd": int(monthly_cap_usd),
        }
        return await self._put_json("/credits/auto-topup", payload)

    async def usage_summary(self, *, days: int = 7) -> JsonDict:
        return await self._get_json("/usage/summary", {"days": int(days)})

    async def usage_recent(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        days: int = 30,
        mode: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "limit": int(limit),
            "offset": int(offset),
            "days": int(days),
        }
        if mode is not None:
            params["mode"] = mode
        return await self._get_json("/usage/recent", params=params)

    async def keys_info(self) -> JsonDict:
        return await self._get_json("/keys/info")

    async def revoke_api_key(self) -> JsonDict:
        return await self._delete_json("/keys/revoke")

    async def memory_list(self, *, limit: int = 20) -> JsonDict:
        return await self._get_json("/memory", {"limit": int(limit)})

    async def memory_search(
        self,
        *,
        q: str,
        top_k: int = 4,
        min_score: float | None = None,
    ) -> JsonDict:
        params = {
            "q": str(q),
            "top_k": int(top_k),
        }
        if min_score is not None:
            params["min_score"] = float(min_score)
        return await self._get_json("/memory/search", params=params)

    async def memory_delete(self, memory_id: str) -> JsonDict:
        return await self._delete_json(f"/memory/{memory_id}")

    async def memory_clear(self) -> JsonDict:
        return await self._delete_json("/memory")

    async def optimization_policy(self) -> JsonDict:
        return await self._get_json("/optimization/policy")

    async def optimization_update_policy(self, payload: dict[str, Any]) -> JsonDict:
        return await self._put_json("/optimization/policy", payload)

    async def optimization_report(
        self,
        *,
        goal: str = "balanced",
        days: int = 30,
        min_calls_per_model: int = 3,
        use_policy: bool = True,
        persist_snapshot: bool = False,
    ) -> JsonDict:
        return await self._get_json(
            "/optimization/report",
            {
                "goal": goal,
                "days": int(days),
                "min_calls_per_model": int(min_calls_per_model),
                "use_policy": bool(use_policy),
                "persist_snapshot": bool(persist_snapshot),
            },
        )

    async def optimization_evaluate(self) -> JsonDict:
        return await self._post_json("/optimization/evaluate", {})

    async def optimization_snapshots(self, *, goal: str | None = None, limit: int = 20) -> JsonDict:
        params: dict[str, Any] = {"limit": int(limit)}
        if goal is not None:
            params["goal"] = goal
        return await self._get_json("/optimization/snapshots", params=params)

    async def optimization_alerts(self, *, limit: int = 12) -> JsonDict:
        return await self._get_json("/optimization/alerts", {"limit": int(limit)})

    async def optimization_replay(self, *, days: int = 30, sample_size: int = 100) -> JsonDict:
        return await self._post_json(
            "/optimization/replay",
            {
                "days": int(days),
                "sample_size": int(sample_size),
            },
        )

    async def optimization_test_templates(self) -> JsonDict:
        return await self._get_json("/optimization/test-templates")

    async def optimization_test_suites(self, *, limit: int = 25) -> JsonDict:
        return await self._get_json("/optimization/test-suites", {"limit": int(limit)})

    async def optimization_create_test_suite(self, payload: dict[str, Any]) -> JsonDict:
        return await self._post_json("/optimization/test-suites", payload)

    async def optimization_update_test_suite(self, suite_id: str, payload: dict[str, Any]) -> JsonDict:
        return await self._put_json(f"/optimization/test-suites/{suite_id}", payload)

    async def optimization_run_test_suite(self, suite_id: str) -> JsonDict:
        return await self._post_json(f"/optimization/test-suites/{suite_id}/run", {})

    async def optimization_test_runs(self, *, limit: int = 20) -> JsonDict:
        return await self._get_json("/optimization/test-runs", {"limit": int(limit)})

    async def optimization_test_run_csv(self, run_id: str) -> str:
        return await self._get_text(f"/optimization/test-runs/{run_id}/csv")

    async def optimization_regression_schedules(self, *, limit: int = 30) -> JsonDict:
        return await self._get_json("/optimization/regression-schedules", {"limit": int(limit)})

    async def optimization_create_regression_schedule(self, payload: dict[str, Any]) -> JsonDict:
        return await self._post_json("/optimization/regression-schedules", payload)

    async def optimization_update_regression_schedule(self, schedule_id: str, payload: dict[str, Any]) -> JsonDict:
        return await self._put_json(f"/optimization/regression-schedules/{schedule_id}", payload)

    async def optimization_run_regression_schedule(self, schedule_id: str) -> JsonDict:
        return await self._post_json(f"/optimization/regression-schedules/{schedule_id}/run", {})

    async def generate_api_key(self) -> JsonDict:
        return await self._post_json("/keys/generate", {})

    async def settings_keys(self) -> JsonDict:
        return await self._get_json("/settings/keys")

    async def settings_save_keys(self, keys: dict[str, str]) -> JsonDict:
        return await self._put_json("/settings/keys", {"keys": keys})

    async def settings_delete_key(self, provider: str) -> JsonDict:
        return await self._delete_json(f"/settings/keys/{provider}")

    async def settings_privacy(self) -> JsonDict:
        return await self._get_json("/settings/privacy")

    async def settings_update_privacy(
        self,
        *,
        data_training_opt_in: bool | None = None,
        zero_retention_mode: bool | None = None,
        purge_existing_data: bool = True,
    ) -> JsonDict:
        payload: dict[str, Any] = {
            "purge_existing_data": bool(purge_existing_data),
        }
        if data_training_opt_in is not None:
            payload["data_training_opt_in"] = bool(data_training_opt_in)
        if zero_retention_mode is not None:
            payload["zero_retention_mode"] = bool(zero_retention_mode)
        return await self._put_json("/settings/privacy", payload)

    async def settings_copilot_state(self) -> JsonDict:
        return await self._get_json("/settings/copilot")

    async def settings_update_copilot(
        self,
        *,
        goal: str | None = None,
        onboarded: bool | None = None,
        checklist: dict[str, bool] | None = None,
    ) -> JsonDict:
        payload: dict[str, Any] = {}
        if goal is not None:
            payload["goal"] = str(goal)
        if onboarded is not None:
            payload["onboarded"] = bool(onboarded)
        if checklist is not None:
            payload["checklist"] = dict(checklist)
        return await self._put_json("/settings/copilot", payload)

    async def settings_ask_copilot(
        self,
        *,
        question: str,
        path: str | None = None,
        context: dict[str, str] | None = None,
    ) -> JsonDict:
        payload: dict[str, Any] = {
            "question": question,
        }
        if path is not None:
            payload["path"] = str(path)
        if context is not None:
            payload["context"] = dict(context)
        return await self._post_json("/settings/copilot/ask", payload)
