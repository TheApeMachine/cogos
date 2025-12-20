from __future__ import annotations

import dataclasses
import queue
from dataclasses import dataclass
from typing import Any, Dict, Optional

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

    def publish(self, type: str, payload: Dict[str, Any]) -> Event:
        evt = Event(type=type, ts=utc_ts(), payload=payload)
        self._q.put(evt)
        return evt

    def get(self, timeout: Optional[float] = None) -> Optional[Event]:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

