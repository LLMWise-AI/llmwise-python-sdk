from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LLMWiseError(Exception):
    status_code: int
    message: str
    payload: Any | None = None

    def __str__(self) -> str:  # pragma: no cover
        base = f"LLMWiseError(status_code={self.status_code}, message={self.message})"
        return base if self.payload is None else f"{base} payload={self.payload!r}"

