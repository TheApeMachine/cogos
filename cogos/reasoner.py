from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional

from .ir import Claim, Plan, ProposedAnswer
from .llm import ChatMessage, ChatModel
from .pyd_compat import BaseModel, Field
from .tools import ToolOutcome
from .util import toks


class Reasoner:
    def propose(
        self,
        user_text: str,
        *,
        plan: Plan,
        evidence_map: Dict[str, str],
        memory_hits: Dict[str, Any],
        tool_outcomes: List[ToolOutcome],
    ) -> ProposedAnswer:
        raise NotImplementedError


class ConservativeReasoner(Reasoner):
    """No LLM: produces only tool-derived and memory-search derived claims."""

    def propose(
        self,
        user_text: str,  # noqa: ARG002
        *,
        plan: Plan,  # noqa: ARG002
        evidence_map: Dict[str, str],  # noqa: ARG002
        memory_hits: Dict[str, Any],
        tool_outcomes: List[ToolOutcome],
    ) -> ProposedAnswer:
        claims: List[Claim] = []
        # calc results
        for out in tool_outcomes:
            if out.ok and out.evidence_id and "result" in out.output:
                claims.append(
                    Claim(
                        text=f"Computed result: {out.output['result']}",
                        evidence_ids=[out.evidence_id],
                        support_spans=[str(out.output["result"])],
                        kind="math",
                    )
                )

        # string counts
        for out in tool_outcomes:
            if not (out.ok and out.evidence_id and out.tool == "count_chars"):
                continue
            text = str(out.output.get("text", "")).strip()
            char = str(out.output.get("char", "")).strip()
            count = out.output.get("count")
            if not text or not char or count is None:
                continue
            claims.append(
                Claim(
                    text=f"The letter {char} appears {count} time(s) in the word {text}.",
                    evidence_ids=[out.evidence_id],
                    support_spans=[str(count), char, text],
                    kind="math",
                )
            )

        # surface top memory hits as "related items" (not asserting facts)
        mem_evid = next((o for o in tool_outcomes if o.ok and o.tool == "memory_search" and o.evidence_id), None)
        if mem_evid and mem_evid.evidence_id:
            notes = memory_hits.get("notes") or []
            if notes:
                line = ", ".join([f"{n.get('title')}({n.get('id')})" for n in notes[:3] if n.get("id")])
                claims.append(
                    Claim(
                        text=f"Related notes: {line}",
                        evidence_ids=[mem_evid.evidence_id],
                        support_spans=[notes[0].get("id", "")],
                        kind="fact",
                    )
                )
        return ProposedAnswer(claims=claims, draft="Here is what I can ground from tools/memory.", proactive=[])


class LLMReasoner(Reasoner):
    def __init__(self, model: ChatModel):
        self.model = model

    class _RawClaim(BaseModel):
        text: str
        evidence_ids: List[str]
        support_spans: List[str]
        kind: Literal["fact", "math", "inference"] = "fact"

    class _Schema(BaseModel):
        claims: List["LLMReasoner._RawClaim"] = Field(default_factory=list)
        draft: str = ""
        proactive: List[Dict[str, Any]] = Field(default_factory=list)

    def propose(
        self,
        user_text: str,
        *,
        plan: Plan,  # noqa: ARG002
        evidence_map: Dict[str, str],
        memory_hits: Dict[str, Any],  # noqa: ARG002
        tool_outcomes: List[ToolOutcome],  # noqa: ARG002
    ) -> ProposedAnswer:
        def _ev_excerpt(s: str, n: int) -> str:
            if not s:
                return ""
            return s if len(s) <= n else s[:n]

        blocks: List[str] = []
        for eid, txt in list(evidence_map.items())[:12]:
            blocks.append(f"[{eid}]\n{_ev_excerpt(txt, 4000)}")

        sys = (
            "You are a reasoning compiler. Output JSON only.\n"
            "Goal: answer the user using atomic claims grounded in evidence.\n\n"
            "Rules (hard):\n"
            "- Every claim MUST include evidence_ids (existing IDs) AND support_spans.\n"
            "- Each support_span MUST be an exact substring of the corresponding evidence text.\n"
            "- Copy/paste support_spans from the evidence verbatim (including punctuation/quotes).\n"
            "- Prefer short spans that are easy to match (numbers, IDs, exact JSON fragments).\n"
            "- Never include the leading [evidence_id] label in support_spans.\n"
            "- Do NOT introduce facts not supported by evidence.\n"
            "- If evidence is insufficient, return claims=[] and draft='I don't know'.\n\n"
            "Examples (illustrative only; do not copy placeholder IDs):\n"
            "Evidence: [<EVID>]\\n{\"result\": 2.5, \"normalized_expression\": \"10/4\"}\n"
            "Good claim: {\"text\":\"10/4 equals 2.5\",\"evidence_ids\":[\"<EVID>\"],\"support_spans\":[\"2.5\",\"10/4\"],\"kind\":\"math\"}\n"
            "Bad claim:  {\"text\":\"10/4 equals 2.5\",\"evidence_ids\":[\"<EVID>\"],\"support_spans\":[\"2.50\"],\"kind\":\"math\"}  (span not exact)\n\n"
            "Output JSON format:\n"
            "{claims:[{text,evidence_ids,support_spans,kind}], draft:str, proactive:list}\n"
        )
        user = (
            f"User question:\n{user_text}\n\n"
            "Evidence you may cite:\n" + "\n".join(blocks) + "\n\n"
            "Return JSON only."
        )
        msgs = [ChatMessage(role="system", content=sys), ChatMessage(role="user", content=user)]
        raw = self.model.generate_json(msgs, self._Schema, temperature=0.2, max_tokens=1200)

        claims: List[Claim] = []
        for rc in raw.claims:
            text = str(rc.text or "").strip()
            eids = list(rc.evidence_ids or [])
            spans = [str(s) for s in (rc.support_spans or []) if str(s).strip()]
            if text and eids and spans:
                claims.append(Claim(text=text, evidence_ids=eids, support_spans=spans, kind=rc.kind))

        return ProposedAnswer(claims=claims, draft=str(raw.draft or ""), proactive=list(raw.proactive or []))


class SearchReasoner(Reasoner):
    """
    Best-of-N reasoning: sample multiple candidate claim sets and choose the one
    that *appears most verifiable* against the provided evidence.

    This is a practical approximation of "explore multiple reasoning paths"
    (Tree-of-Thoughts style) without implementing a full tree search.
    """

    def __init__(self, base: LLMReasoner, samples: int = 4):
        self.base = base
        n = int(samples)
        if n < 1:
            raise ValueError("samples must be >= 1")
        self.samples = n

    @staticmethod
    def _looks_supported(claim: Claim, evidence_map: Dict[str, str]) -> float:
        # Evidence existence
        ev_texts = [evidence_map.get(eid, "") for eid in claim.evidence_ids if eid in evidence_map]
        if not ev_texts:
            return 0.0

        joined = "\n".join(ev_texts)

        # Span hits (exact substring)
        spans = list(claim.support_spans or [])
        if not spans:
            return 0.0
        hit = sum(1 for sp in spans if sp and (sp in joined))
        span_rate = hit / max(1, len(spans))

        # Numeric grounding
        nums = re.findall(r"-?\d+(?:\.\d+)?", claim.text or "")
        num_ok = 1.0
        if nums:
            num_ok = 1.0 if all(n in joined for n in nums) else 0.0

        # Token overlap (weak)
        ct = set(toks(claim.text or ""))
        et = set(toks(joined))
        j = len(ct & et) / (len(ct | et) or 1)

        return 0.70 * span_rate + 0.20 * num_ok + 0.10 * j

    def propose(
        self,
        user_text: str,
        *,
        plan: Plan,
        evidence_map: Dict[str, str],
        memory_hits: Dict[str, Any],
        tool_outcomes: List[ToolOutcome],
    ) -> ProposedAnswer:
        best: Optional[ProposedAnswer] = None
        best_score = -1.0

        for _ in range(self.samples):
            cand = self.base.propose(
                user_text,
                plan=plan,
                evidence_map=evidence_map,
                memory_hits=memory_hits,
                tool_outcomes=tool_outcomes,
            )

            # Score candidate by how supported its claims look.
            if not cand.claims:
                score = 0.0
            else:
                per = [self._looks_supported(c, evidence_map) for c in cand.claims]
                score = sum(per) / max(1, len(per))

            # Tie-breaker: more claims isn't always better, but it can be if supported.
            score = score + 0.02 * len(cand.claims)

            if score > best_score:
                best_score = score
                best = cand

        return best or ProposedAnswer(claims=[], draft="I don't know.", proactive=[])
