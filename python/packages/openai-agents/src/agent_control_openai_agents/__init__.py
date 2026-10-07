"""Native OpenAI Agents SDK Python adapter for Agent Hooks and optional ACS."""

from .adapter import OpenAIAgentsAdapter
from .portable import create_backend

__all__ = ["OpenAIAgentsAdapter", "create_backend"]
