from __future__ import annotations

from typing import List

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
        for c in v.claims:
            lines.append(f"- {c.text}  [evidence: {', '.join(c.evidence_ids)}]")
        return "\n".join(lines)

