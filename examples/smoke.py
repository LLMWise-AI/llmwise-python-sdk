"""Basic smoke checks for the LLMWise Python SDK.

These checks are read-only and safe:
- models
- credits wallet/balance
- usage summary/recent
- conversations/history listing
- keys/settings
- memory/search helpers
- optimization report/policy

Set LLMWISE_API_KEY or pass mm_sk_... directly in your own script.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import asyncio
import os
from typing import Any

from llmwise import AsyncLLMWise, LLMWise


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _short_repr(value: Any) -> str:
    if isinstance(value, dict):
        keys = list(value.keys())
        return f"dict(keys={keys[:4]}...)"
    if isinstance(value, list):
        return f"list(len={len(value)})"
    return str(type(value).__name__)


def run_sync_smoke(client: LLMWise) -> list[CheckResult]:
    def check(name: str, fn: Callable[[], Any]) -> CheckResult:
        try:
            value = fn()
            return CheckResult(name=name, ok=True, detail=_short_repr(value))
        except Exception as exc:  # pragma: no cover - external dependency
            return CheckResult(name=name, ok=False, detail=str(exc))

    checks = [
        ("GET /models", lambda: client.models()),
        ("GET /credits/balance", lambda: client.credits_balance()),
        ("GET /credits/wallet", lambda: client.credits_wallet()),
        ("GET /usage/summary", lambda: client.usage_summary()),
        ("GET /usage/recent", lambda: client.usage_recent(days=1, limit=3)),
        ("GET /conversations", lambda: client.conversations(limit=3, offset=0)),
        ("GET /history", lambda: client.history(limit=3, offset=0)),
        ("GET /keys/info", client.keys_info),
        ("GET /memory", lambda: client.memory_list(limit=3)),
        ("GET /memory/search", lambda: client.memory_search(q="platform", top_k=2)),
        ("GET /optimization/policy", client.optimization_policy),
        ("GET /optimization/report", lambda: client.optimization_report(days=1, goal="balanced")),
        ("GET /settings/keys", client.settings_keys),
        ("GET /settings/privacy", client.settings_privacy),
        ("GET /settings/copilot", client.settings_copilot_state),
    ]

    return [check(name, fn) for name, fn in checks]


async def run_async_smoke(client: AsyncLLMWise) -> list[CheckResult]:
    async def check(name: str, fn: Callable[[], Any]) -> CheckResult:
        try:
            value = await fn()
            return CheckResult(name=name, ok=True, detail=_short_repr(value))
        except Exception as exc:  # pragma: no cover - external dependency
            return CheckResult(name=name, ok=False, detail=str(exc))

    checks = [
        ("GET /models", lambda: client.models()),
        ("GET /credits/balance", client.credits_balance),
        ("GET /credits/wallet", client.credits_wallet),
        ("GET /usage/summary", client.usage_summary),
        ("GET /usage/recent", lambda: client.usage_recent(days=1, limit=3)),
        ("GET /conversations", lambda: client.conversations(limit=3, offset=0)),
        ("GET /history", lambda: client.history(limit=3, offset=0)),
        ("GET /keys/info", client.keys_info),
        ("GET /memory", lambda: client.memory_list(limit=3)),
        ("GET /memory/search", lambda: client.memory_search(q="platform", top_k=2)),
        ("GET /optimization/policy", client.optimization_policy),
        ("GET /optimization/report", lambda: client.optimization_report(days=1, goal="balanced")),
        ("GET /settings/keys", client.settings_keys),
        ("GET /settings/privacy", client.settings_privacy),
        ("GET /settings/copilot", client.settings_copilot_state),
    ]

    return [await check(name, fn) for name, fn in checks]


def print_results(title: str, results: list[CheckResult]) -> int:
    print(f"\n[{title}]")
    failed = 0
    for item in results:
        symbol = "✅" if item.ok else "❌"
        if not item.ok:
            failed += 1
        print(f"{symbol} {item.name}: {item.detail}")
    return failed


def main() -> None:
    if not os.getenv("LLMWISE_API_KEY"):
        print("Set LLMWISE_API_KEY before running this smoke script.")
        raise SystemExit(1)

    sync_client = LLMWise()
    sync_failed = print_results("sync", run_sync_smoke(sync_client))

    async def _run_async() -> int:
        async_client = AsyncLLMWise()
        try:
            async_results = await run_async_smoke(async_client)
            return print_results("async", async_results)
        finally:
            await async_client.aclose()

    async_failed = asyncio.run(_run_async())

    total_failed = sync_failed + async_failed
    print(f"\nSmoke checks complete. failed={total_failed}")
    if total_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
