from __future__ import annotations

import re
from typing import Any, Dict, List

from .ir import Plan, PlanStep, StepMemorySearch, StepRespond, StepToolCall
from .llm import ChatMessage, ChatModel
from .memory import MemoryStore
from .tools import ToolBus
from .util import jdump


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
        plan = self.model.generate_json(msgs, Plan, temperature=0.1, max_tokens=900)
        if not isinstance(plan, Plan):
            raise TypeError(f"LLMPlanner expected Plan, got {type(plan).__name__}")
        if not plan.steps:
            raise ValueError("LLMPlanner returned an empty plan (no steps).")
        if not isinstance(plan.steps[-1], StepRespond):
            raise ValueError("LLMPlanner plan must end with a respond step.")
        return plan
