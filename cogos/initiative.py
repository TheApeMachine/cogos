from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from .memory import MemoryStore
from .pyd_compat import BaseModel, Field
from .util import clamp, utc_ts


class ProactiveCandidate(BaseModel):
    message: str
    evidence_ids: List[str] = Field(default_factory=list)
    expected_utility: float = 0.5
    confidence: float = 0.5
    actionability: float = 0.5
    interruption_cost: float = 0.3
    risk: float = 0.2


class InitiativeManager:
    def __init__(
        self,
        memory: MemoryStore,
        *,
        threshold: float = 0.62,
        cooldown_s: float = 15.0,
        max_per_hour: int = 10,
    ):
        self.memory = memory
        self.threshold = float(threshold)
        self.cooldown_s = float(cooldown_s)
        self.max_per_hour = int(max_per_hour)
        self._lock = threading.Lock()
        self._last_emit = 0.0
        self._recent: List[float] = []

    def _score(self, c: ProactiveCandidate) -> float:
        impulse = (c.expected_utility * c.confidence * c.actionability) - (c.interruption_cost + c.risk)
        return clamp(0.5 + impulse, 0.0, 1.0)

    def submit(self, c: ProactiveCandidate) -> Optional[str]:
        score = self._score(c)
        now = utc_ts()
        with self._lock:
            # cooldown
            if now - self._last_emit < self.cooldown_s:
                return None
            # rate limit
            cutoff = now - 3600.0
            self._recent = [t for t in self._recent if t >= cutoff]
            if len(self._recent) >= self.max_per_hour:
                return None

            if score >= self.threshold:
                pid = self.memory.add_proactive(c.message, score=score, evidence_ids=c.evidence_ids)
                self._last_emit = now
                self._recent.append(now)
                return pid
        return None

    def poll(self, limit: int = 3) -> List[Dict[str, Any]]:
        return self.memory.fetch_undelivered_proactive(limit=limit)

