from __future__ import annotations

from typing import Any, Dict, List, Optional

from .ir import Plan, VerifiedAnswer
from .logging_utils import log
from .memory import MemoryStore
from .pyd_compat import BaseModel, Field, _model_dump
from .tools import ToolOutcome
from .util import jdump, short, utc_ts


class NotaryReport(BaseModel):
    escalated: bool
    task_id: Optional[str] = None
    evidence_id: Optional[str] = None
    reason: str = ""


class Notary:
    """
    Minimal "hard-line cut" mechanism.

    The Notary is not user-interactive: it records an internal trace and (optionally)
    creates a high-priority task for human review when the system cannot produce a
    verified answer after its allowed steering attempts.
    """

    def __init__(self, memory: MemoryStore, *, priority: int = 10):
        self.memory = memory
        self.priority = int(priority)

    def escalate(
        self,
        *,
        user_text: str,
        plan: Plan,
        verified: VerifiedAnswer,
        tool_outcomes: List[ToolOutcome],
        reason: str,
    ) -> NotaryReport:
        normalized_reason = str(reason) if reason else "unspecified"
        payload: Dict[str, Any] = {
            "ts": utc_ts(),
            "reason": normalized_reason,
            "user_text": user_text,
            "plan": _model_dump(plan),
            "verified": _model_dump(verified),
            "tool_outcomes": [_model_dump(o) for o in (tool_outcomes or [])],
        }

        evid = self.memory.add_evidence(
            "notary_escalation",
            jdump(payload),
            metadata={"source_type": "notary", "trust_score": 1.0},
            dedupe=False,
        )

        desc = (
            "CogOS could not produce a verified answer.\n\n"
            f"reason: {normalized_reason}\n"
            f"evidence_id: {evid}\n"
            f"user_text_snip: {short(user_text, 400)}\n"
        )
        tid = self.memory.add_task(
            "Human review required (CogOS Notary)",
            desc,
            priority=self.priority,
            payload={"evidence_id": evid, "reason": normalized_reason, "ts": payload["ts"]},
        )

        log.warning(
            "Notary escalation created (task=%s evidence=%s reason=%s)",
            tid,
            evid,
            normalized_reason,
            extra={"extra": {"task_id": tid, "evidence_id": evid, "reason": normalized_reason}},
        )
        return NotaryReport(
            escalated=True,
            task_id=tid,
            evidence_id=evid,
            reason=normalized_reason,
        )


