from __future__ import annotations

import re
from typing import Any, Dict, List

from .ir import Plan, PlanStep, StepMemorySearch, StepRespond, StepToolCall
from .llm import ChatMessage, ChatModel
from .memory import MemoryStore
from .pyd_compat import BaseModel, Field
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

    class _Schema(BaseModel):
        steps: List[Dict[str, Any]] = Field(default_factory=list)

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
            "Create a plan as a list of steps. Allowed step types:\n"
            " - memory_search: {type:'memory_search', query, k}\n"
            " - tool_call: {type:'tool_call', tool, arguments}\n"
            " - write_note: {type:'write_note', title, content, tags, confidence}\n"
            " - create_task: {type:'create_task', title, description, priority, payload}\n"
            " - respond: {type:'respond', style}\n"
            "Rules:\n"
            "- Prefer memory_search first.\n"
            "- Use tool_call when useful.\n"
            "- Avoid side-effect tools unless required.\n"
            "- Always end with respond.\n"
        )
        user = (
            "User request:\n"
            + user_text
            + "\n\nAvailable tools:\n"
            + jdump(compact_tools)
            + "\n\nReturn {steps:[...]} only."
        )
        msgs = [ChatMessage(role="system", content=sys), ChatMessage(role="user", content=user)]
        raw = self.model.generate_json(msgs, self._Schema, temperature=0.1, max_tokens=900)
        return Plan(steps=raw.steps)  # pydantic validates union
