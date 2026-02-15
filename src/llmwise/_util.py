from __future__ import annotations

import os


def env(key: str, default: str | None = None) -> str | None:
    value = os.environ.get(key)
    if value is None:
        return default
    value = value.strip()
    return value or default


def normalize_api_base(base_url: str) -> str:
    url = (base_url or "").strip().rstrip("/")
    if not url:
        return "https://llmwise.ai/api/v1"
    # Allow passing either https://llmwise.ai or https://llmwise.ai/api/v1
    if url.endswith("/api/v1"):
        return url
    return f"{url}/api/v1"


def bearer_headers(api_key: str | None) -> dict[str, str]:
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}

