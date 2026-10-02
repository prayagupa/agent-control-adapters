"""Real CrewAI runs against an operator-managed local Ollama model."""

import os

# Set these before importing CrewAI, including when a module is run with -m.
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ["CREWAI_TELEMETRY_ENABLED"] = "false"
