from __future__ import annotations

import ast
import datetime as dt
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .event_bus import EventBus
from .memory import MemoryStore
from .pyd_compat import BaseModel, Field, ValidationError, _model_dump, _model_json_schema
from .util import jdump, utc_ts


class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolOutcome(BaseModel):
    ok: bool
    output: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    evidence_id: Optional[str] = None
    tool: Optional[str] = None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: Callable[[BaseModel], BaseModel]
    side_effects: bool = False


class ToolBus:
    def __init__(self, memory: MemoryStore, event_bus: EventBus, *, allow_side_effects: bool = False):
        self.memory = memory
        self.bus = event_bus
        self.allow_side_effects = allow_side_effects
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def list_tools(self) -> List[Dict[str, Any]]:
        out = []
        for name in sorted(self._tools.keys()):
            spec = self._tools[name]
            out.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "side_effects": spec.side_effects,
                    "input_schema": _model_json_schema(spec.input_model),
                    "output_schema": _model_json_schema(spec.output_model),
                }
            )
        return out

    def execute(self, call: ToolCall) -> ToolOutcome:
        spec = self._tools.get(call.name)
        if not spec:
            return ToolOutcome(ok=False, error=f"Unknown tool: {call.name}", tool=call.name)
        if spec.side_effects and not self.allow_side_effects:
            return ToolOutcome(ok=False, error=f"Tool '{call.name}' is side-effectful and disabled.", tool=call.name)

        try:
            inp = spec.input_model(**call.arguments)
        except ValidationError as e:
            return ToolOutcome(ok=False, error=f"Invalid tool args: {e}", tool=call.name)

        try:
            out = spec.handler(inp)
            # Validate output
            out = spec.output_model(**_model_dump(out))
        except Exception as e:
            return ToolOutcome(ok=False, error=f"Tool error: {e}", tool=call.name)

        evid_id = self.memory.add_evidence(
            kind=f"tool:{call.name}",
            content=jdump(_model_dump(out)),
            metadata={"tool": call.name, "args": call.arguments},
        )
        self.bus.publish("tool_executed", {"tool": call.name, "call": _model_dump(call), "evidence_id": evid_id})
        return ToolOutcome(ok=True, output=_model_dump(out), evidence_id=evid_id, tool=call.name)


# ---- Built-in tools ----

_ALLOWED_FUNCS: Dict[str, Any] = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pi": math.pi,
    "e": math.e,
    "abs": abs,
    "round": round,
}


class CalcIn(BaseModel):
    expression: str


class CalcOut(BaseModel):
    result: float
    normalized_expression: str


class _SafeEval(ast.NodeVisitor):
    def visit(self, node):  # type: ignore[override]
        if isinstance(node, ast.Expression):
            return self.visit(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("Only numeric constants allowed.")
        if isinstance(node, ast.BinOp):
            a = self.visit(node.left)
            b = self.visit(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Div):
                return a / b
            if isinstance(node.op, ast.FloorDiv):
                return a // b
            if isinstance(node.op, ast.Mod):
                return a % b
            if isinstance(node.op, ast.Pow):
                return a**b
            raise ValueError("Operator not allowed.")
        if isinstance(node, ast.UnaryOp):
            v = self.visit(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +v
            if isinstance(node.op, ast.USub):
                return -v
            raise ValueError("Unary op not allowed.")
        if isinstance(node, ast.Name):
            if node.id in _ALLOWED_FUNCS and isinstance(_ALLOWED_FUNCS[node.id], (int, float)):
                return float(_ALLOWED_FUNCS[node.id])
            raise ValueError(f"Name not allowed: {node.id}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls allowed.")
            fn = node.func.id
            if fn not in _ALLOWED_FUNCS or not callable(_ALLOWED_FUNCS[fn]):
                raise ValueError(f"Function not allowed: {fn}")
            args = [self.visit(a) for a in node.args]
            return float(_ALLOWED_FUNCS[fn](*args))
        raise ValueError(f"Expression element not allowed: {type(node).__name__}")


def calc_handler(inp: CalcIn) -> CalcOut:
    expr = inp.expression.strip().replace("×", "*").replace("÷", "/").replace("^", "**")
    tree = ast.parse(expr, mode="eval")
    val = _SafeEval().visit(tree)
    return CalcOut(result=float(val), normalized_expression=expr)


class NowIn(BaseModel):
    pass


class NowOut(BaseModel):
    iso: str
    unix: float


def now_handler(_: NowIn) -> NowOut:
    now = dt.datetime.now().isoformat(timespec="seconds")
    return NowOut(iso=now, unix=utc_ts())


class MemSearchIn(BaseModel):
    query: str
    k: int = 5


class MemSearchOut(BaseModel):
    notes: List[Dict[str, Any]] = Field(default_factory=list)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    skills: List[Dict[str, Any]] = Field(default_factory=list)


def make_mem_search_handler(mem: MemoryStore) -> Callable[[MemSearchIn], MemSearchOut]:
    def _h(inp: MemSearchIn) -> MemSearchOut:
        return MemSearchOut(
            notes=mem.search_notes(inp.query, k=inp.k),
            evidence=mem.search_evidence(inp.query, k=inp.k),
            skills=mem.search_skills(inp.query, k=inp.k),
        )

    return _h


class CountCharsIn(BaseModel):
    text: str
    char: str
    case_sensitive: bool = True


class CountCharsOut(BaseModel):
    text: str
    char: str
    count: int
    case_sensitive: bool


def count_chars_handler(inp: CountCharsIn) -> CountCharsOut:
    text = inp.text or ""
    char = (inp.char or "")
    if len(char) != 1:
        raise ValueError("char must be a single character.")

    haystack = text
    needle = char
    if not inp.case_sensitive:
        haystack = haystack.lower()
        needle = needle.lower()

    return CountCharsOut(text=text, char=char, count=int(haystack.count(needle)), case_sensitive=bool(inp.case_sensitive))


def _resolve_under_roots(path: str, roots: Sequence[str]) -> Path:
    p = Path(path).expanduser().resolve()
    for r in roots:
        root = Path(r).expanduser().resolve()
        try:
            p.relative_to(root)
            return p
        except Exception:
            continue
    raise PermissionError(f"Path '{p}' is not under allowed roots: {list(roots)}")


class ReadFileIn(BaseModel):
    path: str
    max_bytes: int = 250_000


class ReadFileOut(BaseModel):
    path: str
    content: str
    truncated: bool


def make_read_file_handler(roots: Sequence[str]) -> Callable[[ReadFileIn], ReadFileOut]:
    roots = list(roots)

    def _h(inp: ReadFileIn) -> ReadFileOut:
        p = _resolve_under_roots(inp.path, roots)
        data = p.read_bytes()
        truncated = False
        if len(data) > inp.max_bytes:
            data = data[: inp.max_bytes]
            truncated = True
        return ReadFileOut(path=str(p), content=data.decode("utf-8", errors="replace"), truncated=truncated)

    return _h


class WriteFileIn(BaseModel):
    path: str
    content: str
    overwrite: bool = False


class WriteFileOut(BaseModel):
    path: str
    bytes_written: int


def make_write_file_handler(roots: Sequence[str]) -> Callable[[WriteFileIn], WriteFileOut]:
    roots = list(roots)

    def _h(inp: WriteFileIn) -> WriteFileOut:
        p = _resolve_under_roots(inp.path, roots)
        if p.exists() and not inp.overwrite:
            raise FileExistsError(f"File exists: {p} (set overwrite=true)")
        p.parent.mkdir(parents=True, exist_ok=True)
        data = inp.content.encode("utf-8")
        p.write_bytes(data)
        return WriteFileOut(path=str(p), bytes_written=len(data))

    return _h
