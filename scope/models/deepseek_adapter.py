"""
DeepSeek API adapter for SCOPE.

DeepSeek exposes an OpenAI-compatible Chat Completions API, so this adapter
reuses the OpenAI adapter implementation and only handles DeepSeek-specific
defaults (API key, base URL, and model names).

Supported models:
- ``deepseek-chat``: general-purpose chat model (recommended for SCOPE)
- ``deepseek-reasoner``: reasoning model (does not accept ``temperature``)
"""
import os
from typing import Any, Optional

from .openai_adapter import OpenAIAdapter

DEFAULT_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com"


class DeepSeekAdapter(OpenAIAdapter):
    """
    Adapter for the DeepSeek API (OpenAI-compatible).

    Example:
        ```python
        from openai import AsyncOpenAI
        from scope.models import DeepSeekAdapter

        client = AsyncOpenAI(
            api_key="sk-...",
            base_url="https://api.deepseek.com",
        )
        model = DeepSeekAdapter(client, model="deepseek-chat")
        ```
    """

    def __init__(
        self,
        client: Any,
        model: str = DEFAULT_MODEL,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the DeepSeek adapter.

        Args:
            client: OpenAI-compatible client instance (AsyncOpenAI)
            model: DeepSeek model name (default: deepseek-chat)
            temperature: Sampling temperature (None for API default).
                Not supported by ``deepseek-reasoner``.
            max_tokens: Maximum tokens in response (None for model default)
            **kwargs: Additional parameters passed to chat.completions.create()
        """
        super().__init__(
            client,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )


def create_deepseek_model(
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    **kwargs
) -> DeepSeekAdapter:
    """
    Convenience function to create a DeepSeek adapter.

    Args:
        model: DeepSeek model name (default: deepseek-chat)
        api_key: DeepSeek API key (uses DEEPSEEK_API_KEY env var if not provided)
        base_url: Custom base URL (uses DEEPSEEK_API_BASE, DEEPSEEK_BASE_URL,
            or https://api.deepseek.com if not provided)
        temperature: Sampling temperature (None for API default)
        max_tokens: Maximum tokens in response (None for model default)
        **kwargs: Additional parameters passed to chat.completions.create()

    Returns:
        Configured DeepSeekAdapter

    Example:
        ```python
        from scope.models import create_deepseek_model

        # Uses DEEPSEEK_API_KEY env var
        model = create_deepseek_model("deepseek-chat")

        # With explicit credentials
        model = create_deepseek_model(
            "deepseek-chat",
            api_key="sk-...",
            base_url="https://api.deepseek.com",
        )
        ```

    Note:
        Use ``deepseek-chat`` with SCOPE. ``deepseek-reasoner`` does not accept
        a ``temperature`` parameter, so leave it as None for that model.
    """
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise ImportError(
            "OpenAI package not installed. Install with: pip install openai"
        )

    effective_api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not effective_api_key:
        raise ValueError(
            "DeepSeek API key not provided. Set the DEEPSEEK_API_KEY "
            "environment variable or pass api_key=..."
        )

    effective_base_url = (
        base_url
        or os.getenv("DEEPSEEK_API_BASE")
        or os.getenv("DEEPSEEK_BASE_URL")
        or DEFAULT_BASE_URL
    )

    client = AsyncOpenAI(
        api_key=effective_api_key,
        base_url=effective_base_url,
    )
    return DeepSeekAdapter(
        client,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )
