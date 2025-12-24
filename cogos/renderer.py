from __future__ import annotations

from typing import Dict, List

from .ir import VerifiedAnswer
from .memory import MemoryStore


class Renderer:
    def __init__(self, memory: MemoryStore):
        self.memory = memory

    def render(self, v: VerifiedAnswer) -> str:
        if not v.ok:
            suffix = ""
            if v.warnings:
                suffix = " (" + "; ".join(v.warnings) + ")"
            return "I don’t know — I can’t verify any claims from evidence." + suffix

        lines: List[str] = []
        if v.warnings:
            lines.append("⚠️ " + " ".join(v.warnings))
        lines.append("Here’s what I can support from evidence:")

        # De-noise: merge duplicate claim texts and aggregate evidence IDs.
        by_text: Dict[str, List[str]] = {}
        for c in v.claims:
            txt = str(c.text or "").strip()
            if not txt:
                continue
            by_text.setdefault(txt, [])
            for eid in c.evidence_ids:
                if eid and eid not in by_text[txt]:
                    by_text[txt].append(eid)

        for txt, eids in by_text.items():
            lines.append(f"- {txt}  [evidence: {', '.join(eids)}]")
        return "\n".join(lines)

