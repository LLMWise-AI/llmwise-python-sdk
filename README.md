# LLMWise Python SDK

Lightweight Python client for the LLMWise multi-model API:

- Chat (single model + Auto routing)
- Failover routing (primary + fallback chain)
- Compare (run 2+ models in parallel)
- Blend (synthesize answers from multiple models, supports MoA / Self-MoA)
- Judge (model-vs-model evaluation)

This SDK is intentionally small: it depends only on `httpx`.

## Install

```bash
pip install llmwise-sdk
```

## Quickstart (Chat)

```python
from llmwise_sdk import LLMWise

client = LLMWise("mm_sk_...")

resp = client.chat(
    model="auto",
    messages=[{"role": "user", "content": "Write a haiku about rate limits."}],
    stream=False,
)

print(resp["content"])
```

## Streaming (Chat)

```python
from llmwise_sdk import LLMWise

client = LLMWise("mm_sk_...")

for event in client.chat_stream(
    model="claude-sonnet-4.5",
    messages=[{"role": "user", "content": "Explain recursion to a 10-year-old."}],
):
    if event.get("event") == "done":
        print(f"\n\ncharged={event.get('credits_charged')} remaining={event.get('credits_remaining')}")
        break
    delta = event.get("delta")
    if delta:
        print(delta, end="", flush=True)
```

## Failover (Chat + Routing)

```python
from llmwise_sdk import LLMWise

client = LLMWise("mm_sk_...")

for event in client.chat_stream(
    model="claude-sonnet-4.5",
    routing={"fallback": ["gpt-5.2", "gemini-3-flash"], "strategy": "rate-limit"},
    messages=[{"role": "user", "content": "Summarize this in 3 bullets: ..."}],
):
    # route/trace events are emitted when failover triggers
    if event.get("event") in {"route", "trace"}:
        continue
    if event.get("event") == "done":
        break
    if event.get("delta"):
        print(event["delta"], end="", flush=True)
```

## Compare (2+ models)

```python
from llmwise_sdk import LLMWise

client = LLMWise("mm_sk_...")

events = client.compare_stream(
    models=["gpt-5.2", "claude-sonnet-4.5", "gemini-3-flash"],
    messages=[{"role": "user", "content": "Draft a short product description for a credit-based LLM gateway."}],
)

by_model = {}
for ev in events:
    if ev.get("event") == "summary":
        print("summary:", ev)
    if ev.get("event") == "done":
        break
    if ev.get("delta"):
        by_model.setdefault(ev["model"], "")
        by_model[ev["model"]] += ev["delta"]
```

## Configure

Environment variables:

- `LLMWISE_API_KEY`
- `LLMWISE_BASE_URL` (default: `https://llmwise.ai/api/v1`)

## Notes

- Authentication uses `Authorization: Bearer <mm_sk_...>` (or a Clerk JWT).
- Streaming uses Server-Sent Events (SSE) and yields parsed JSON dictionaries.
