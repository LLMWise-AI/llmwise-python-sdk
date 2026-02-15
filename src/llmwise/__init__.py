from importlib.metadata import PackageNotFoundError, version

from .client import AsyncLLMWise, LLMWise
from .errors import LLMWiseError

try:
    __version__ = version("llmwise")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["AsyncLLMWise", "LLMWise", "LLMWiseError", "__version__"]
