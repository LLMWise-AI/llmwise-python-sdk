from __future__ import annotations

from typing import Any, Literal, TypedDict

Role = Literal["system", "user", "assistant"]


class TextBlock(TypedDict):
    type: Literal["text"]
    text: str


class ImageUrl(TypedDict):
    url: str


class ImageBlock(TypedDict):
    type: Literal["image_url"]
    image_url: ImageUrl


ContentBlock = TextBlock | ImageBlock
MessageContent = str | list[ContentBlock]


class Message(TypedDict):
    role: Role
    content: MessageContent


class RoutingConfig(TypedDict, total=False):
    fallback: list[str]
    strategy: str


JsonDict = dict[str, Any]

