from __future__ import annotations

from typing import Any, Dict, List, Literal, Union

from .pyd_compat import BaseModel, Field
from .util import new_id


class StepBase(BaseModel):
    type: str


class StepMemorySearch(StepBase):
    type: Literal["memory_search"] = "memory_search"
    query: str
    k: int = 5


class StepToolCall(StepBase):
    type: Literal["tool_call"] = "tool_call"
    tool: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class StepWriteNote(StepBase):
    type: Literal["write_note"] = "write_note"
    title: str
    content: str
    tags: List[str] = Field(default_factory=list)
    confidence: float = 0.7


class StepCreateTask(StepBase):
    type: Literal["create_task"] = "create_task"
    title: str
    description: str
    priority: int = 0
    payload: Dict[str, Any] = Field(default_factory=dict)


class StepRespond(StepBase):
    type: Literal["respond"] = "respond"
    style: str = "helpful"


PlanStep = Union[StepMemorySearch, StepToolCall, StepWriteNote, StepCreateTask, StepRespond]


class Plan(BaseModel):
    steps: List[PlanStep] = Field(default_factory=list)


class Claim(BaseModel):
    id: str = Field(default_factory=lambda: new_id("clm"))
    text: str
    evidence_ids: List[str] = Field(default_factory=list)
    support_spans: List[str] = Field(default_factory=list)  # exact quotes expected to appear in evidence
    kind: Literal["fact", "math", "inference"] = "fact"
    status: Literal["proposed", "verified", "rejected"] = "proposed"
    score: float = 0.5


class ProposedAnswer(BaseModel):
    claims: List[Claim] = Field(default_factory=list)
    draft: str = ""
    proactive: List[Dict[str, Any]] = Field(default_factory=list)


class VerifiedAnswer(BaseModel):
    ok: bool
    claims: List[Claim] = Field(default_factory=list)
    response: str = ""
    warnings: List[str] = Field(default_factory=list)

