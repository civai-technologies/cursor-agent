"""
Factory for creating different types of AI agents.

This module provides a unified API for creating various agents (OpenAI, Claude, etc.)
with consistent configuration.
"""

import os
from typing import Optional, Callable, Any

from cursor_agent_tools.base import BaseAgent
from cursor_agent_tools.claude_agent import ClaudeAgent
from cursor_agent_tools.logger import get_logger
from cursor_agent_tools.openai_agent import OpenAIAgent
from cursor_agent_tools.ollama_agent import OllamaAgent
from cursor_agent_tools.permissions import PermissionOptions, PermissionRequest, PermissionStatus

# Initialize logger
logger = get_logger(__name__)

# Dictionary mapping providers to their supported models
# This makes it easy to determine which provider to use based on a model name
MODEL_MAPPING = {
    "claude": [
        "claude-3-5-sonnet-latest",
        "claude-3-7-sonnet-latest",
        "claude-3-5-sonnet",
        "claude-3.5-sonnet",
        "claude-3-sonnet",
        "claude-3-haiku",
        "claude-3.5-haiku"
    ],
    "openai": [
        "gpt-4o",
        "gpt-4o-2024-05-13",
        "gpt-4o-2024-08-06",
        "gpt-4o-mini",
        "gpt-4",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "gpt-3.5"
    ],
    "ollama": [
        "llama3",
        "llama3.1",
        "llama3.2",
        "llama3.3",
        "deepseek-r1",
        "gemma3",
        "mistral",
        "phi4",
        "qwen2.5",
        "qwen2.5-coder"
    ]
}

# For model normalization (e.g., handling model aliases)
MODEL_NORMALIZATION = {
    "gpt-4o-2024-05-13": "gpt-4o",
    "gpt-4o-2024-08-06": "gpt-4o",
    "gpt-4-turbo": "gpt-4",
    "claude-3.5-sonnet": "claude-3-5-sonnet-latest"
}

# Models commonly served by an OpenAI-compatible Cursor gateway.
# The package does not embed any private gateway URL — callers must set
# CURSOR_API_BASE_URL or pass base_url=.
CURSOR_MODEL_IDS = {
    "auto",
    "composer-2.5",
    "composer-2",
    "gpt-5.2",
    "gpt-5.3-codex",
    "opus-4.6-thinking",
    "sonnet-4.5-thinking",
    "gemini-3-pro",
}


def _cursor_base_url(explicit: Optional[str] = None) -> str:
    """Resolve Cursor gateway base URL (public package: no hardcoded host)."""
    if explicit:
        return str(explicit).rstrip("/")
    override = (os.getenv("CURSOR_API_BASE_URL") or "").strip()
    if override:
        return override.rstrip("/")
    raise ValueError(
        "Cursor gateway base URL not provided. Pass base_url=... or set "
        "CURSOR_API_BASE_URL to your OpenAI-compatible Cursor endpoint."
    )


def _cursor_api_key(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    return (os.getenv("CURSOR_API_KEY") or "").strip()


def _normalize_cursor_model(model: str) -> str:
    m = model.strip()
    if m.startswith("cursor/") or m.startswith("cursor-"):
        m = m.split("/", 1)[-1] if "/" in m else m[len("cursor-") :]
    return m or "composer-2.5"


def create_agent(
    model: str,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    timeout: int = 180,
    permission_callback: Optional[Callable[[PermissionRequest], PermissionStatus]] = None,
    permissions: Optional[PermissionOptions] = None,
    default_tool_timeout: int = 300,
    **kwargs: Any
) -> BaseAgent:
    """
    Create an agent based on the specified model.

    Args:
        model: The name of the model to use (e.g., "gpt-4o", "claude-3-opus",
            "ollama-llama3", "composer-2.5" / "cursor/auto" for a Cursor gateway)
        api_key: The API key to use for the model provider
        temperature: The temperature to use for the model
        timeout: Hard wall-clock cap (seconds) on each LLM HTTP round-trip
        permission_callback: A callback function to handle permission requests
        permissions: Optional PermissionOptions object containing permission settings
        default_tool_timeout: Maximum execution time in seconds for tool calls (default: 300)
        **kwargs: Additional model-specific arguments. For Cursor models, pass
            ``base_url`` (or set ``CURSOR_API_BASE_URL``) to an OpenAI-compatible
            gateway — the package does not hardcode a private host.

    Returns:
        An agent instance configured with the specified parameters
    """
    raw_model = model
    model = model.lower()  # Normalize model name to lowercase
    logger.info(f"Creating agent with model: {model}")
    logger.debug(f"Agent parameters: temperature={temperature}, timeout={timeout}, default_tool_timeout={default_tool_timeout}")

    # Set up permission options if not provided
    if permissions is None:
        permissions = PermissionOptions()
        logger.debug("Using default permission options")
    else:
        logger.debug(f"Using custom permission options, yolo_mode={permissions.yolo_mode}")

    # Handle Ollama models (prefix detection)
    if model.startswith("ollama-"):
        logger.debug("Detected Ollama model")
        # Ollama doesn't require an API key but uses a local server
        host = kwargs.get("host") or os.getenv("OLLAMA_HOST") or "http://localhost:11434"
        logger.debug(f"Using Ollama host: {host}")

        logger.info(f"Creating OllamaAgent with model {model}")
        return OllamaAgent(
            model=model,
            temperature=temperature,
            timeout=timeout,
            permission_callback=permission_callback,
            permission_options=permissions,
            default_tool_timeout=default_tool_timeout,
            host=host,
            **kwargs
        )

    # Cursor via caller-configured OpenAI-compatible gateway
    cursor_model = _normalize_cursor_model(model)
    if (
        model.startswith("cursor/")
        or model.startswith("cursor-")
        or cursor_model in CURSOR_MODEL_IDS
        or kwargs.get("provider") == "cursor"
    ):
        logger.debug("Detected Cursor gateway model")
        key = _cursor_api_key(api_key)
        if not key:
            raise ValueError(
                "Cursor API key not provided (pass api_key=... or set CURSOR_API_KEY)"
            )
        explicit_base = kwargs.pop("base_url", None) or kwargs.pop("api_base", None)
        base_url = _cursor_base_url(explicit_base)
        kwargs.pop("provider", None)
        kwargs.pop("host", None)
        logger.info(
            "Creating OpenAIAgent for Cursor model %s via %s", cursor_model, base_url
        )
        return OpenAIAgent(
            model=cursor_model,
            api_key=key,
            temperature=temperature,
            timeout=timeout,
            permission_callback=permission_callback,
            permission_options=permissions,
            default_tool_timeout=default_tool_timeout,
            base_url=base_url,
            **kwargs,
        )

    # Handle OpenAI models
    elif any(name in model for name in ["gpt-", "openai"]):
        logger.debug("Detected OpenAI model")
        # Use environment variable if no API key is provided
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.error("OpenAI API key not provided and not found in environment")
                raise ValueError("OpenAI API key not provided and not found in environment")
            else:
                logger.debug("Using OpenAI API key from environment")

        logger.info(f"Creating OpenAIAgent with model {model}")
        return OpenAIAgent(
            model=model,
            api_key=api_key,
            temperature=temperature,
            timeout=timeout,
            permission_callback=permission_callback,
            permission_options=permissions,
            default_tool_timeout=default_tool_timeout,
            **kwargs
        )

    # Handle Anthropic/Claude models
    elif any(name in model for name in ["claude", "anthropic"]):
        logger.debug("Detected Anthropic/Claude model")
        # Use environment variable if no API key is provided
        if api_key is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                logger.error("Anthropic API key not provided and not found in environment")
                raise ValueError("Anthropic API key not provided and not found in environment")
            else:
                logger.debug("Using Anthropic API key from environment")

        logger.info(f"Creating ClaudeAgent with model {model}")
        return ClaudeAgent(
            model=model,
            api_key=api_key,
            temperature=temperature,
            timeout=timeout,
            permission_callback=permission_callback,
            permission_options=permissions,
            default_tool_timeout=default_tool_timeout,
            **kwargs
        )

    # Raise error for unsupported models
    else:
        logger.error(f"Unsupported model: {raw_model}")
        raise ValueError(f"Unsupported model: {raw_model}")
