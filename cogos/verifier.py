from __future__ import annotations

import re
from typing import Dict, List, Optional

from .ir import Claim, ProposedAnswer, VerifiedAnswer
from .memory import MemoryStore
from .util import toks


class Verifier:
    """
    Enforces "no unsupported claims":
    - Evidence IDs must exist.
    - Support spans must appear verbatim in evidence texts.
    - If claim text contains numbers, those numbers must appear in evidence.
    """

    def __init__(
        self,
        memory: MemoryStore,
        *,
        require_spans: bool = True,
        min_span_hits: float = 0.5,
        min_trust_score: float = 0.0,
    ):
        self.memory = memory
        self.require_spans = require_spans
        self.min_span_hits = float(min_span_hits)
        self.min_trust_score = float(min_trust_score)

    def _ev_text(self, evid: str) -> Optional[str]:
        ev = self.memory.get_evidence(evid)
        if not ev:
            return None
        try:
            trust = float((ev.get("metadata") or {}).get("trust_score", 1.0))
        except Exception:
            trust = 1.0
        if trust < self.min_trust_score:
            return None
        return ev["content"] or ""

    @staticmethod
    def _numbers(s: str) -> List[str]:
        return re.findall(r"-?\d+(?:\.\d+)?", s)

    def verify_claim(self, c: Claim) -> Claim:
        # Evidence existence
        ev_texts: Dict[str, str] = {}
        for evid in c.evidence_ids:
            t = self._ev_text(evid)
            if t is not None:
                ev_texts[evid] = t
        if not ev_texts:
            return (
                c.model_copy(update={"status": "rejected", "score": 0.0})
                if hasattr(c, "model_copy")
                else c.copy(update={"status": "rejected", "score": 0.0})
            )  # type: ignore

        # Support spans
        spans = list(c.support_spans or [])
        if self.require_spans and not spans:
            return (
                c.model_copy(update={"status": "rejected", "score": 0.0})
                if hasattr(c, "model_copy")
                else c.copy(update={"status": "rejected", "score": 0.0})
            )  # type: ignore

        # Critical: spans must also appear in the claim text itself.
        #
        # Otherwise a model can "launder" an unsupported claim by citing arbitrary
        # substrings that happen to exist in evidence (e.g. JSON keys), while the
        # claim text says something else entirely.
        claim_txt = str(c.text or "")
        if spans and any(sp not in claim_txt for sp in spans):
            return (
                c.model_copy(update={"status": "rejected", "score": 0.0})
                if hasattr(c, "model_copy")
                else c.copy(update={"status": "rejected", "score": 0.0})
            )  # type: ignore

        hit = 0
        for sp in spans:
            found = any(sp in txt for txt in ev_texts.values())
            if found:
                hit += 1
        span_hit_rate = hit / max(1, len(spans))

        # Numeric grounding
        nums = self._numbers(c.text)
        num_ok = True
        if nums:
            joined = "\n".join(ev_texts.values())
            for n in nums:
                if n not in joined:
                    num_ok = False
                    break

        # Token overlap (weak secondary signal)
        ct = set(toks(c.text))
        et = set(toks("\n".join(ev_texts.values())))
        j = len(ct & et) / (len(ct | et) or 1)

        # Score and decision
        score = 0.65 * span_hit_rate + 0.20 * (1.0 if num_ok else 0.0) + 0.15 * j
        ok = (span_hit_rate >= self.min_span_hits) and num_ok

        status = "verified" if ok else "rejected"
        updated = {"status": status, "score": float(score)}
        if hasattr(c, "model_copy"):
            return c.model_copy(update=updated)
        return c.copy(update=updated)  # type: ignore

    def verify(self, p: ProposedAnswer) -> VerifiedAnswer:
        verified: List[Claim] = []
        rejected = 0
        for c in p.claims:
            vc = self.verify_claim(c)
            if vc.status == "verified":
                verified.append(vc)
            else:
                rejected += 1

        warns: List[str] = []
        if rejected:
            warns.append(f"Rejected {rejected} unsupported claim(s).")
        if not verified:
            return VerifiedAnswer(ok=False, claims=[], response="", warnings=warns + ["No verified claims. Abstaining."])
        return VerifiedAnswer(ok=True, claims=verified, response="", warnings=warns)

