from __future__ import annotations

import dataclasses
import queue
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, Dict, Optional

from .util import new_id, utc_ts


@dataclass(frozen=True)
class Event:
    type: str
    ts: float
    payload: Dict[str, Any]
    id: str = dataclasses.field(default_factory=lambda: new_id("evt"))


class EventBus:
    """Thread-safe pub/sub queue."""

    def __init__(self):
        self._q: "queue.Queue[Event]" = queue.Queue()
        self._lock = Lock()
        self._listeners: list[Callable[[Event], None]] = []

    def add_listener(self, fn: Callable[[Event], None]) -> None:
        """Register a best-effort listener invoked on every publish()."""
        with self._lock:
            self._listeners.append(fn)

    def remove_listener(self, fn: Callable[[Event], None]) -> None:
        with self._lock:
            self._listeners = [f for f in self._listeners if f is not fn]

    def publish(self, type: str, payload: Dict[str, Any]) -> Event:
        evt = Event(type=type, ts=utc_ts(), payload=payload)
        self._q.put(evt)
        # Notify listeners without affecting queue consumers.
        with self._lock:
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(evt)
            except Exception:
                # Best-effort: never let a listener break publish.
                pass
        return evt

    def get(self, timeout: Optional[float] = None) -> Optional[Event]:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

