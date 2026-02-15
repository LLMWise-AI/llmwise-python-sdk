# LLMWise Python SDK

Lightweight Python client for the LLMWise multi-model API:

- Chat (single model + Auto routing)
- Failover routing (primary + fallback chain)
- Compare (run 2+ models in parallel)
- Blend (synthesize answers from multiple models, supports MoA / Self-MoA)
- Judge (model-vs-model evaluation)
- Full API coverage for conversations, history, credits, usage, keys, memory,
  optimization, and settings.

This SDK is intentionally small: it depends only on `httpx`.

## Install

```bash
pip install llmwise
```

## Quickstart (Chat)

```python
from llmwise import LLMWise

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
from llmwise import LLMWise

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
from llmwise import LLMWise

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
from llmwise import LLMWise

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

## Additional API Helpers

- `conversations()`, `create_conversation()`, `update_conversation()`, `delete_conversation()`
- `history()`, `get_history_detail()`
- `credits_wallet()`, `credits_transactions()`, `credits_packs()`, `credits_purchase()`,
  `credits_confirm_checkout()`, `credits_update_auto_topup()`, `credits_balance()`
- `usage_summary()`, `usage_recent()`
- `keys_info()`, `generate_api_key()`, `revoke_api_key()`
- `memory_list()`, `memory_search()`, `memory_delete()`, `memory_clear()`
- `optimization_*` helpers, including policy/report/evaluate/replay/test suites/regression schedules
- `settings_*` helpers, including provider keys, privacy, and copilot state/ask endpoints

## Additional API Helper Examples

### Dashboard/API Health Check (Read-only)

```python
from llmwise import LLMWise

client = LLMWise()

print("models:", len(client.models()))
print("balance:", client.credits_balance())
print("usage:", client.usage_summary(days=7))
print("conversations:", client.conversations(limit=5))
```

### Conversation Lifecycle

```python
from llmwise import LLMWise

client = LLMWise()

convo = client.create_conversation()
convo_id = convo["id"]
client.update_conversation(convo_id, title="API smoke room")
history = client.history(limit=10)
client.delete_conversation(convo_id)

print("history entries:", history.get("total", 0))
```

### Credits & Cost

```python
from llmwise import LLMWise

client = LLMWise()

print("wallet:", client.credits_wallet())
print("recent usage:", client.usage_recent(days=7, limit=5))
print("packs:", client.credits_packs())
```

### Settings and Optimization

```python
from llmwise import LLMWise

client = LLMWise()

print("copilot:", client.settings_copilot_state())
print("privacy:", client.settings_privacy())
print("policy:", client.optimization_policy())
print("report:", client.optimization_report(days=1))
```

## Smoke Script

Run all read-only SDK health checks:

```bash
LLMWISE_API_KEY=mm_sk_... python examples/smoke.py
```
