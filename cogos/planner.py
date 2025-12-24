from __future__ import annotations

import re
from typing import Any, Dict, List

from .ir import Plan, PlanStep, StepMemorySearch, StepRespond, StepToolCall
from .llm import ChatMessage, ChatModel
from .memory import MemoryStore
from .tools import ToolBus
from .util import jdump
from . import pyd_compat


class Planner:
    def plan(self, user_text: str, *, tools: ToolBus, memory: MemoryStore) -> Plan:
        raise NotImplementedError


class RulePlanner(Planner):
    _ARITH = re.compile(r"(?<!\w)(?:\d+\s*[\+\-\*/\^]\s*\d+|calculate|compute|eval|evaluate)(?!\w)", re.IGNORECASE)
    _COUNT_LETTER = re.compile(
        r"how many times\s+(?:is|does)?\s*(?:the\s+)?letter\s+['\"]?([a-z])['\"]?\s+(?:appear(?:s)?\s+)?(?:in|within)\s+(?:the\s+)?(?:word\s+)?['\"]?([a-z]+)['\"]?",
        re.IGNORECASE,
    )

    def plan(self, user_text: str, *, tools: ToolBus, memory: MemoryStore) -> Plan:  # noqa: ARG002
        # Deterministic string counting.
        m = self._COUNT_LETTER.search(user_text or "")
        if m:
            ch = (m.group(1) or "").strip()
            word = (m.group(2) or "").strip()
            steps: List[PlanStep] = [
                StepToolCall(tool="count_chars", arguments={"text": word, "char": ch, "case_sensitive": False})
            ]
            steps.append(StepRespond())
            return Plan(steps=steps)

        # Deterministic arithmetic.
        if self._ARITH.search(user_text):
            expr = user_text
            m = re.search(r"(?:calculate|compute|eval|evaluate)\s*[:\-]?\s*(.*)$", user_text, re.IGNORECASE)
            if m and m.group(1).strip():
                expr = m.group(1).strip()
            steps = [StepToolCall(tool="calc", arguments={"expression": expr})]
            steps.append(StepRespond())
            return Plan(steps=steps)

        steps = [StepMemorySearch(query=user_text, k=6)]
        steps.append(StepRespond())
        return Plan(steps=steps)


class LLMPlanner(Planner):
    def __init__(self, model: ChatModel):
        self.model = model

    @staticmethod
    def _looks_like_arithmetic_request(user_text: str) -> bool:
        ut = user_text or ""
        if re.search(r"\d+\s*[\+\-\*/\^]\s*\d+", ut):
            return True
        if re.search(r"\b(calculate|compute|eval|evaluate)\b", ut, re.IGNORECASE):
            return True
        return False

    @staticmethod
    def _token_overlap(a: str, b: str) -> float:
        ta = set(re.findall(r"[a-z0-9_]+", (a or "").lower()))
        tb = set(re.findall(r"[a-z0-9_]+", (b or "").lower()))
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / (len(ta | tb) or 1)

    @staticmethod
    def _explicitly_asked_to_write_note(user_text: str) -> bool:
        ut = (user_text or "").lower()
        return bool(
            re.search(r"\b(write|save|store)\s+(a\s+)?note\b", ut)
            or re.search(r"\bremember\s+this\b", ut)
            or re.search(r"^note\s*:", ut.strip())
        )

    @staticmethod
    def _explicitly_asked_to_create_task(user_text: str) -> bool:
        ut = (user_text or "").lower()
        return bool(
            re.search(r"\b(create|add|make)\s+(a\s+)?task\b", ut)
            or re.search(r"\b(todo|to-do)\b", ut)
        )

    def _sanitize_plan(self, plan: Plan, *, user_text: str, tools: ToolBus) -> Plan:
        """
        Post-validate and sanitize an LLM-produced plan.

        The planner model is untrusted. We enforce:
        - no StepWriteNote/StepCreateTask unless the user explicitly asked
        - tool names must exist
        - no calc unless the user asked for arithmetic
        - memory_search query must be relevant (otherwise use the user_text)
        - always end with respond
        """
        # Import here to avoid cyclic import issues in some environments.
        from .ir import StepCreateTask, StepWriteNote  # noqa: WPS433

        allowed_tools = {str(t["name"]) for t in tools.list_tools() if t.get("name")}
        allow_note = self._explicitly_asked_to_write_note(user_text)
        allow_task = self._explicitly_asked_to_create_task(user_text)
        allow_calc = self._looks_like_arithmetic_request(user_text)

        out_steps: List[PlanStep] = []
        for st in list(plan.steps or []):
            if isinstance(st, StepRespond):
                continue

            if isinstance(st, StepWriteNote) and not allow_note:
                continue
            if isinstance(st, StepCreateTask) and not allow_task:
                continue

            if isinstance(st, StepToolCall):
                tool = str(st.tool or "").strip()
                if tool not in allowed_tools:
                    continue
                if tool == "calc" and not allow_calc:
                    continue
                out_steps.append(st)
                continue

            if isinstance(st, StepMemorySearch):
                q = str(st.query or "").strip()
                if not q:
                    q = user_text
                # If the model picks something unrelated (e.g. "hello"), override.
                if self._token_overlap(q, user_text) < 0.25:
                    q = user_text
                k = int(getattr(st, "k", 6) or 6)
                if k < 3:
                    k = 3
                # pydantic v1/v2 compatible update
                if hasattr(st, "model_copy"):
                    st2 = st.model_copy(update={"query": q, "k": k})
                else:
                    st2 = st.copy(update={"query": q, "k": k})  # type: ignore[attr-defined]
                out_steps.append(st2)
                continue

            # Unknown step type: drop it.
            continue

        # Safety cap: avoid runaway plans.
        out_steps = out_steps[:6]
        out_steps.append(StepRespond())
        return Plan(steps=out_steps)

    def plan(self, user_text: str, *, tools: ToolBus, memory: MemoryStore) -> Plan:  # noqa: ARG002
        compact_tools = []
        for t in tools.list_tools():
            props = (t["input_schema"].get("properties") or {})
            compact_tools.append(
                {
                    "name": t["name"],
                    "description": t["description"],
                    "side_effects": t["side_effects"],
                    "args": list(props.keys()),
                }
            )

        sys = (
            "You are a planning compiler. Output JSON only.\n"
            "Return a JSON object with this shape:\n"
            "{\"steps\":[ ... ]}\n\n"
            "Each step must be one of:\n"
            "- memory_search: {\"type\":\"memory_search\",\"query\":str,\"k\":int}\n"
            "- tool_call:    {\"type\":\"tool_call\",\"tool\":str,\"arguments\":object}\n"
            "- write_note:   {\"type\":\"write_note\",\"title\":str,\"content\":str,\"tags\":[str],\"confidence\":number}\n"
            "- create_task:  {\"type\":\"create_task\",\"title\":str,\"description\":str,\"priority\":int,\"payload\":object}\n"
            "- respond:      {\"type\":\"respond\",\"style\":str}\n\n"
            "Rules (hard):\n"
            "- steps must be non-empty.\n"
            "- The last step MUST be respond.\n"
            "- arguments MUST be an object ({}), not a list.\n"
            "- payload MUST be an object ({}), not a list.\n"
            "- Use the exact key order shown in each step format.\n"
            "- Do not add extra keys.\n"
            "- Do NOT create_task or write_note unless the user explicitly asked.\n"
            "- Avoid side-effect tools unless required.\n\n"
            "Guidance:\n"
            "- Use tool_call 'count_chars' for letter counting questions.\n\n"
            "Examples:\n"
            "User: hi\n"
            "{\"steps\":[{\"type\":\"respond\",\"style\":\"helpful\"}]}\n\n"
            "User: calculate 2+2\n"
            "{\"steps\":[{\"type\":\"tool_call\",\"tool\":\"calc\",\"arguments\":{\"expression\":\"2+2\"}},{\"type\":\"respond\",\"style\":\"helpful\"}]}\n"
            "\n"
            "User: How many times is the letter r in the word strawberry?\n"
            "{\"steps\":[{\"type\":\"tool_call\",\"tool\":\"count_chars\",\"arguments\":{\"text\":\"strawberry\",\"char\":\"r\",\"case_sensitive\":false}},{\"type\":\"respond\",\"style\":\"helpful\"}]}\n"
        )
        user = (
            "User request:\n"
            + user_text
            + "\n\nAvailable tools:\n"
            + jdump(compact_tools)
            + "\n\nReturn JSON only."
        )
        msgs = [ChatMessage(role="system", content=sys), ChatMessage(role="user", content=user)]
        plan = self.model.generate_json(msgs, Plan, temperature=0.1, max_tokens=700)
        if not isinstance(plan, Plan):
            raise TypeError(f"LLMPlanner expected Plan, got {type(plan).__name__}")
        if not plan.steps:
            raise ValueError("LLMPlanner returned an empty plan (no steps).")
        if not isinstance(plan.steps[-1], StepRespond):
            raise ValueError("LLMPlanner plan must end with a respond step.")
        # Sanitize/validate. Never trust the raw plan.
        try:
            return self._sanitize_plan(plan, user_text=user_text, tools=tools)
        except Exception:
            # Absolute fallback: deterministic plan.
            return Plan(steps=[StepMemorySearch(query=user_text, k=6), StepRespond()])
