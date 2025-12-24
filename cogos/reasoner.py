from __future__ import annotations

import json
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

    @staticmethod
    def _build_span_menu(evidence_text: str, user_text: str, *, max_spans: int = 14) -> List[str]:
        """
        Build a small menu of candidate support spans extracted from `evidence_text`.

        The model will cite spans by index, and we will map indices back to the exact
        substring. This avoids brittle "copy/paste exact substring" behavior.
        """
        txt = str(evidence_text or "")
        if not txt:
            return []

        MAX_LEN = 120
        max_spans = max(1, int(max_spans))
        user_tokens = set(toks(user_text or ""))

        def score(seg: str) -> float:
            st = set(toks(seg))
            if not st:
                return 0.0
            if not user_tokens:
                return 0.0
            return len(st & user_tokens) / len(st)

        def add(menu: List[str], seen: set[str], seg: str) -> None:
            if not seg:
                return
            s = str(seg)
            if "\n" in s:
                return
            s = s.strip()
            if not s:
                return
            if len(s) > MAX_LEN:
                s = s[:MAX_LEN]
            if s in seen:
                return
            seen.add(s)
            menu.append(s)

        # Candidates (all extracted as exact substrings or slices thereof).
        kv_pat = re.compile(
            r'"[^"\n]{1,60}"\s*:\s*(?:"[^"\n]{0,60}"|-?\d+(?:\.\d+)?|true|false|null)',
            re.IGNORECASE,
        )
        kv = [m.group(0) for m in kv_pat.finditer(txt)]

        quote_pat = re.compile(r'"[^"\n]{1,80}"')
        quotes = [m.group(0) for m in quote_pat.finditer(txt)]

        nums = re.findall(r"-?\d+(?:\.\d+)?", txt)
        lines = [ln for ln in txt.splitlines() if ln and ln.strip()]

        # Relevance-sort the "semantic" candidates; keep numbers separately.
        kv_sorted = sorted(kv, key=lambda s: (-score(s), len(s)))
        quotes_sorted = sorted(quotes, key=lambda s: (-score(s), len(s)))
        lines_sorted = sorted(lines, key=lambda s: (-score(s), len(s)))

        menu: List[str] = []
        seen: set[str] = set()

        # Always include the first non-empty line slice (helpful fallback context).
        for ln in lines:
            add(menu, seen, ln)
            break

        # Prefer structured fragments (JSON-ish key/value pairs).
        for seg in kv_sorted[: max_spans * 2]:
            add(menu, seen, seg)
            if len(menu) >= max_spans:
                return menu

        # Include a handful of numeric atoms.
        for n in nums[: max_spans * 2]:
            add(menu, seen, n)
            if len(menu) >= max_spans:
                return menu

        # Fill remaining slots with relevant quoted strings and lines.
        for seg in quotes_sorted:
            add(menu, seen, seg)
            if len(menu) >= max_spans:
                return menu
        for seg in lines_sorted:
            add(menu, seen, seg)
            if len(menu) >= max_spans:
                return menu

        return menu[:max_spans]

    class _RawClaim(BaseModel):
        text: str
        evidence_ids: List[str]
        support_span_ids: List[int]
        kind: Literal["fact", "math", "inference"] = "fact"

    class _Schema(BaseModel):
        claims: List["LLMReasoner._RawClaim"]
        draft: str
        proactive: List[Dict[str, Any]]

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
        span_menus: Dict[str, List[str]] = {}
        for eid, txt in list(evidence_map.items())[:8]:
            menu = self._build_span_menu(txt, user_text, max_spans=14)
            span_menus[eid] = menu
            menu_lines = "\n".join([f"{i}: {json.dumps(sp, ensure_ascii=False)}" for i, sp in enumerate(menu)]) or "(empty)"
            blocks.append(f"[{eid}]\nEXCERPT:\n{_ev_excerpt(txt, 900)}\n\nSPAN_MENU (cite by index):\n{menu_lines}")

        sys = (
            "You are a reasoning compiler. Output JSON only.\n"
            "Goal: answer the user using atomic claims grounded in evidence.\n\n"
            "Rules (hard):\n"
            "- Output must be small: at most 3 claims.\n"
            "- Each claim MUST cite exactly 1 evidence_id.\n"
            "- Each claim MUST include 1 or 2 support_span_ids (integers).\n"
            "- Each support_span_id MUST be a valid index into the cited evidence's SPAN_MENU.\n"
            "- Keep draft under 200 characters.\n"
            "- Every claim MUST include evidence_ids (existing IDs) AND support_span_ids.\n"
            "- Output key order MUST be: claims, draft, proactive.\n"
            "- Claim key order MUST be: text, evidence_ids, support_span_ids, kind.\n"
            "- Do NOT introduce facts not supported by evidence.\n"
            "- If evidence is insufficient, return claims=[] and draft='I don't know'.\n\n"
            "- Set proactive=[] unless the user explicitly asked for suggestions.\n\n"
            "Return ONLY the JSON object. Do not include examples or extra text.\n"
        )
        user = (
            f"User question:\n{user_text}\n\n"
            "Evidence you may cite:\n" + "\n".join(blocks) + "\n\n"
            "Return JSON only."
        )
        msgs = [ChatMessage(role="system", content=sys), ChatMessage(role="user", content=user)]
        try:
            raw = self.model.generate_json(msgs, self._Schema, temperature=0.15, max_tokens=600)
        except Exception:
            # Hard fail-safe: never crash the agent loop due to model JSON issues.
            return ProposedAnswer(claims=[], draft="I don't know.", proactive=[])

        claims: List[Claim] = []
        for rc in raw.claims:
            text = str(rc.text or "").strip()
            eids = list(rc.evidence_ids or [])
            span_ids = list(rc.support_span_ids or [])
            if not (text and eids and span_ids):
                continue
            evid = str(eids[0])
            menu = span_menus.get(evid) or []
            spans: List[str] = []
            ok = True
            for sid in span_ids:
                try:
                    i = int(sid)
                except Exception:
                    ok = False
                    break
                if i < 0 or i >= len(menu):
                    ok = False
                    break
                sp = menu[i]
                # Safety: ensure it's actually in evidence (should always be true).
                if sp and (sp in (evidence_map.get(evid, "") or "")):
                    spans.append(sp)
                else:
                    ok = False
                    break
            if ok and spans:
                claims.append(Claim(text=text, evidence_ids=[evid], support_spans=spans, kind=rc.kind))

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
            try:
                cand = self.base.propose(
                    user_text,
                    plan=plan,
                    evidence_map=evidence_map,
                    memory_hits=memory_hits,
                    tool_outcomes=tool_outcomes,
                )
            except Exception:
                cand = ProposedAnswer(claims=[], draft="I don't know.", proactive=[])

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
