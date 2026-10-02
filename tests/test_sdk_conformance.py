"""Unmodified SDK reference vectors: dependency evidence, NOT adapter conformance."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from agent_hooks.ctk.reference import ReferenceHarness
from agent_hooks.ctk.runner import load_vectors, run_vector

VECTORS = load_vectors()


@pytest.mark.parametrize("vector", VECTORS, ids=[vector["id"] for vector in VECTORS])
def test_pinned_sdk_reference_contract(vector: dict[str, Any]) -> None:
    """Leave vector expectations intact and name any reference-host capability skip."""
    result = asyncio.run(run_vector(ReferenceHarness(), vector))
    if result.status == "skip":
        pytest.skip(f"SDK reference-host capability exclusion: {result.detail}")
    assert result.status == "pass", "\n".join(result.failures)
