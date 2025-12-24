"""IR models for planning, reasoning, and verification.

This module defines the pydantic-style models that flow through CogOS:
- planner output (Plan/PlanStep)
- reasoner output (ProposedAnswer/Claim)
- verifier output (VerifiedAnswer)
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Union

from .pyd_compat import BaseModel, Field
from .util import new_id


class StepBase(BaseModel):
    """Base type for plan steps (discriminated by the `type` field on leaf classes)."""

    # Intentionally empty: step variants define a literal `type` field.
    #
    # basedpyright correctly flags overriding a mutable instance variable with a
    # narrower type (e.g. `str` -> `Literal["tool_call"]`) as incompatible.
    # Keeping `type` only on leaf classes preserves the discriminated-union shape
    # without fighting the type checker.
    # (No fields here on purpose.)


class StepMemorySearch(StepBase):
    """Plan step: retrieve related items from memory."""

    type: Literal["memory_search"] = "memory_search"
    query: str
    k: int = 5


class StepToolCall(StepBase):
    """Plan step: invoke a tool with validated arguments."""

    type: Literal["tool_call"] = "tool_call"
    tool: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class StepWriteNote(StepBase):
    """Plan step: write a Note into the memory store."""

    type: Literal["write_note"] = "write_note"
    title: str
    content: str
    tags: List[str] = Field(default_factory=list)
    confidence: float = 0.7


class StepCreateTask(StepBase):
    """Plan step: create a background Task."""

    type: Literal["create_task"] = "create_task"
    title: str
    description: str
    priority: int = 0
    payload: Dict[str, Any] = Field(default_factory=dict)


class StepRespond(StepBase):
    """Plan step: respond to the user."""

    type: Literal["respond"] = "respond"
    style: str = "helpful"


PlanStep = Union[StepMemorySearch, StepToolCall, StepWriteNote, StepCreateTask, StepRespond]


class Plan(BaseModel):
    """A planned sequence of steps (must end with a respond step)."""

    steps: List[PlanStep] = Field(default_factory=list)


class Claim(BaseModel):
    """Atomic, evidence-grounded statement proposed by the reasoner."""

    id: str = Field(default_factory=lambda: new_id("clm"))
    text: str
    evidence_ids: List[str] = Field(default_factory=list)
    # Exact quotes expected to appear in evidence (and, for safety, in the claim text).
    support_spans: List[str] = Field(default_factory=list)
    kind: Literal["fact", "math", "inference"] = "fact"
    status: Literal["proposed", "verified", "rejected"] = "proposed"
    score: float = 0.5


class ProposedAnswer(BaseModel):
    """Reasoner output prior to verification."""

    claims: List[Claim] = Field(default_factory=list)
    draft: str = ""
    proactive: List[Dict[str, Any]] = Field(default_factory=list)


class VerifiedAnswer(BaseModel):
    """Verifier output: only claims that survived evidence checks."""

    ok: bool
    claims: List[Claim] = Field(default_factory=list)
    response: str = ""
    warnings: List[str] = Field(default_factory=list)
