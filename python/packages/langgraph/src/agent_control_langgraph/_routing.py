"""Bounded, graph-owned correlation proof for distinct model and tool actors."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from agent_control_adapters.codecs import wire_equal
from agent_control_adapters.json import JSON
from agent_control_adapters.runtime import PolicySession


@dataclass(frozen=True)
class Route:
    """A trusted static model-to-tool route with actual destination ownership."""

    origin: object
    destination: object


@dataclass(frozen=True)
class PendingRoute:
    """Capture model origin and unchanged native call before the graph routes it."""

    route: Route
    call: dict[str, JSON]
    active: bool = False


class RoutingLedger:
    """Verify exact destination before consuming shared budget under the source actor."""

    def __init__(self, session: PolicySession, routes: tuple[Route, ...]) -> None:
        self._session = session
        self._routes = routes
        self._lock = threading.RLock()
        self._pending: dict[str, PendingRoute] = {}
        self._seen: set[str] = set()

    def reserve(self, origin: object, call: dict[str, JSON]) -> None:
        """Retain a bounded proof from an actual completed model output."""
        with self._lock:
            identifier = call["id"]
            route = next((item for item in self._routes if item.origin is origin), None)
            if (
                route is None
                or not isinstance(identifier, str)
                or identifier in self._seen
                or len(self._pending) >= 64
                or len(self._seen) >= 64
            ):
                raise ValueError("Unproven, duplicate or oversized native tool route")
            self._seen.add(identifier)
            self._pending[identifier] = PendingRoute(route, call)

    def before(self, destination: object, call: dict[str, JSON]) -> int:
        """Check real node identity and call before budget authority crosses actors."""
        with self._lock:
            identifier = call["id"]
            if not isinstance(identifier, str):
                raise ValueError("Native correlation must be text")
            proof = self._pending.get(identifier)
            if (
                proof is None
                or proof.active
                or proof.route.destination is not destination
                or not wire_equal(proof.call, call)
            ):
                raise ValueError("Wrong node, unknown origin or replayed native tool route")
            source = self._session.binding(proof.route.origin)
            self._session.budget.before_tool(source, identifier)
            self._pending[identifier] = PendingRoute(proof.route, proof.call, True)
            return source

    def after(self, destination: object, call: dict[str, JSON], source: int) -> None:
        """Complete only the proven actual tool effect, exactly once."""
        with self._lock:
            identifier = call["id"]
            if not isinstance(identifier, str):
                raise ValueError("Native correlation must be text")
            proof = self._pending.get(identifier)
            if (
                proof is None
                or not proof.active
                or proof.route.destination is not destination
                or self._session.binding(proof.route.origin) != source
                or not wire_equal(proof.call, call)
            ):
                raise ValueError("Wrong origin or duplicate native tool completion")
            self._session.budget.after_tool(source, identifier)
            del self._pending[identifier]

    @property
    def pending(self) -> int:
        """Expose bounded owner state for cleanup assertions."""
        with self._lock:
            return len(self._pending)

    def close(self) -> None:
        """Release all owner-scoped proof and replay state on every terminal path."""
        with self._lock:
            self._pending.clear()
            self._seen.clear()
