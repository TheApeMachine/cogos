from __future__ import annotations

import ast
import datetime as dt
import html
import logging
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
import concurrent.futures as cf
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from threading import Lock
from typing import Any, Callable, ClassVar, Dict, List, Optional, Sequence

from .event_bus import EventBus
from .memory import MemoryStore
from .pyd_compat import BaseModel, Field, ValidationError, _model_dump, _model_json_schema
from .util import jdump, utc_ts

logger = logging.getLogger(__name__)

try:
    # pydantic v2
    from pydantic import field_validator as _field_validator  # type: ignore
except Exception:  # pragma: no cover
    _field_validator = None

try:
    # pydantic v1
    from pydantic import validator as _validator  # type: ignore
except Exception:  # pragma: no cover
    _validator = None


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
    """
    Specification for a tool exposed to the agent runtime.

    `default_trust_score` is written into evidence metadata for tool executions and must be a
    finite float in the closed interval [0.0, 1.0].
    """

    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: Callable[[BaseModel], BaseModel]
    side_effects: bool = False
    source_type: str = "tool_output"
    # Conservative default: tool outputs are not treated as perfectly trustworthy by default.
    default_trust_score: float = 0.8
    evidence_metadata_builder: Optional[Callable[[BaseModel, BaseModel], Dict[str, Any]]] = None

    def __post_init__(self) -> None:
        # Validate trust score early so invalid specs are rejected at construction time.
        try:
            ts = float(self.default_trust_score)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"ToolSpec.default_trust_score must be a number in [0.0, 1.0]; got {self.default_trust_score!r}"
            ) from e

        if not math.isfinite(ts) or ts < 0.0 or ts > 1.0:
            raise ValueError(
                f"ToolSpec.default_trust_score must be a finite number in [0.0, 1.0]; got {self.default_trust_score!r}"
            )


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

        evidence_metadata: Dict[str, Any] = {
            "tool": call.name,
            "args": call.arguments,
            "source_type": spec.source_type,
            "trust_score": float(spec.default_trust_score),
        }
        if spec.evidence_metadata_builder is not None:
            try:
                evidence_metadata.update(spec.evidence_metadata_builder(inp, out))
            except Exception:
                # Metadata is best-effort; never break tool execution.
                logger.exception("evidence_metadata_builder failed for tool %s", call.name)

        evid_id = self.memory.add_evidence(
            kind=f"tool:{call.name}",
            content=jdump(_model_dump(out)),
            metadata=evidence_metadata,
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


# ---- Web search (best-effort, optional) ----


class WebSearchIn(BaseModel):
    query: str
    k: int = 5
    allow_domains: List[str] = Field(default_factory=list)
    deny_domains: List[str] = Field(default_factory=list)
    timeout_s: float = 8.0

    # Public bounds (used both for validation and runtime clamping).
    K_MIN: ClassVar[int] = 1
    K_MAX: ClassVar[int] = 50
    TIMEOUT_MIN: ClassVar[float] = 0.1
    TIMEOUT_MAX: ClassVar[float] = 60.0

    if _field_validator is not None:

        @_field_validator("k", mode="before")
        @classmethod
        def _validate_k_v2(cls, v: Any) -> int:
            if isinstance(v, bool):
                raise ValueError(f"k must be between {cls.K_MIN} and {cls.K_MAX}")
            try:
                iv = int(v)
            except Exception as e:  # noqa: BLE001
                raise ValueError(f"k must be between {cls.K_MIN} and {cls.K_MAX}") from e
            if iv < cls.K_MIN or iv > cls.K_MAX:
                raise ValueError(f"k must be between {cls.K_MIN} and {cls.K_MAX}")
            return iv

        @_field_validator("timeout_s", mode="before")
        @classmethod
        def _validate_timeout_s_v2(cls, v: Any) -> float:
            try:
                fv = float(v)
            except Exception as e:  # noqa: BLE001
                raise ValueError(
                    f"timeout_s must be between {cls.TIMEOUT_MIN:g} and {cls.TIMEOUT_MAX:g}"
                ) from e
            if not math.isfinite(fv) or fv < cls.TIMEOUT_MIN or fv > cls.TIMEOUT_MAX:
                raise ValueError(f"timeout_s must be between {cls.TIMEOUT_MIN:g} and {cls.TIMEOUT_MAX:g}")
            return fv

    elif _validator is not None:

        @_validator("k", pre=True)
        def _validate_k_v1(cls, v: Any) -> int:
            if isinstance(v, bool):
                raise ValueError(f"k must be between {cls.K_MIN} and {cls.K_MAX}")
            try:
                iv = int(v)
            except Exception as e:  # noqa: BLE001
                raise ValueError(f"k must be between {cls.K_MIN} and {cls.K_MAX}") from e
            if iv < cls.K_MIN or iv > cls.K_MAX:
                raise ValueError(f"k must be between {cls.K_MIN} and {cls.K_MAX}")
            return iv

        @_validator("timeout_s", pre=True)
        def _validate_timeout_s_v1(cls, v: Any) -> float:
            try:
                fv = float(v)
            except Exception as e:  # noqa: BLE001
                raise ValueError(
                    f"timeout_s must be between {cls.TIMEOUT_MIN:g} and {cls.TIMEOUT_MAX:g}"
                ) from e
            if not math.isfinite(fv) or fv < cls.TIMEOUT_MIN or fv > cls.TIMEOUT_MAX:
                raise ValueError(f"timeout_s must be between {cls.TIMEOUT_MIN:g} and {cls.TIMEOUT_MAX:g}")
            return fv

    else:  # pragma: no cover
        # If pydantic validators aren't available (e.g., during lightweight/static analysis),
        # skip validation rather than failing at import time.
        pass


class WebSearchResult(BaseModel):
    title: str
    url: str
    snippet: str = ""
    domain: str = ""
    rank: int = 0
    trust_score: float = 0.35


class WebSearchOut(BaseModel):
    query: str
    provider: str
    results: List[WebSearchResult] = Field(default_factory=list)


def _strip_html(s: str) -> str:
    if not s:
        return ""

    class _TextOnlyHTMLParser(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self._chunks: List[str] = []

        def handle_data(self, data: str) -> None:  # pyright: ignore[reportImplicitOverride]
            if data:
                self._chunks.append(data)

        def text(self) -> str:
            # Join chunks with spaces to preserve word boundaries across tags,
            # then normalize whitespace downstream.
            return " ".join(self._chunks)

    try:
        p = _TextOnlyHTMLParser()
        p.feed(s)
        p.close()
        out = p.text()
    except Exception:  # noqa: BLE001
        # Best-effort fallback to previous regex behavior.
        out = re.sub(r"<[^>]+>", " ", s)

    out = html.unescape(out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _normalize_domain(d: str) -> str:
    d = (d or "").strip().lower()
    if d.startswith("www."):
        d = d[4:]
    return d


def _domain_matches(domain: str, suffix: str) -> bool:
    domain = _normalize_domain(domain)
    suffix = _normalize_domain(suffix)
    if not suffix:
        return False
    return domain == suffix or domain.endswith("." + suffix)


def _domain_allowed(domain: str, allow: Sequence[str], deny: Sequence[str]) -> bool:
    domain = _normalize_domain(domain)
    if any(_domain_matches(domain, d) for d in deny if d):
        return False
    allow_clean = [a for a in allow if a]
    if not allow_clean:
        return True
    return any(_domain_matches(domain, a) for a in allow_clean)


_DEFAULT_DOMAIN_TRUST: Dict[str, float] = {
    "wikipedia.org": 0.9,
    "arxiv.org": 0.85,
    "github.com": 0.75,
    "docs.python.org": 0.85,
    "developer.mozilla.org": 0.85,
}


def _trust_for_domain(domain: str, allow: Sequence[str]) -> float:
    d = _normalize_domain(domain)
    for suf, score in _DEFAULT_DOMAIN_TRUST.items():
        if _domain_matches(d, suf):
            return float(score)
    # Light heuristics
    if d.endswith(".gov") or d.endswith(".gov.uk") or d.endswith(".gov.au"):
        return 0.8
    if d.endswith(".edu"):
        return 0.75
    if any(_domain_matches(d, a) for a in allow if a):
        return 0.6
    return 0.35


def _ddg_unwrap_url(href: str) -> str:
    href = (href or "").strip()
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    if href.startswith("/l/?") or href.startswith("/l/"):
        parsed = urllib.parse.urlparse("https://duckduckgo.com" + href)
        qs = urllib.parse.parse_qs(parsed.query)
        if "uddg" in qs and qs["uddg"]:
            try:
                return urllib.parse.unquote(qs["uddg"][0])
            except Exception:
                return qs["uddg"][0]
    if href.startswith("/"):
        return "https://duckduckgo.com" + href
    return href


def _duckduckgo_lite_search(
    query: str,
    *,
    k: int,
    timeout_s: float,
    raise_on_error: bool = False,
) -> List[Dict[str, str]]:
    """
    Best-effort DuckDuckGo Lite HTML scraping.

    Note: This helper is intentionally simple; production safeguards (caching, backoff,
    circuit breaker, metrics) are implemented in `make_web_search_handler`.
    """
    q = (query or "").strip()
    if not q:
        return []
    url = "https://lite.duckduckgo.com/lite/?" + urllib.parse.urlencode({"q": q})
    req = urllib.request.Request(url, headers={"User-Agent": "CogOS/1.0 (+local)"})
    try:
        with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
            data = resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        if raise_on_error:
            raise
        logger.warning("DuckDuckGo search request failed: %s", e)
        return []
    except Exception as e:
        if raise_on_error:
            raise
        logger.error("Unexpected error in web search: %s", e)
        return []
    page = data.decode("utf-8", errors="replace")

    # Try to extract results from either lite or html layouts (best-effort).
    link_pats = [
        re.compile(r'<a[^>]+class="result-link"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.I | re.S),
        re.compile(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.I | re.S),
    ]
    snip_pats = [
        re.compile(r'<td[^>]+class="result-snippet"[^>]*>(.*?)</td>', re.I | re.S),
        re.compile(r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', re.I | re.S),
        re.compile(r'<div[^>]+class="result__snippet"[^>]*>(.*?)</div>', re.I | re.S),
    ]

    matches = []
    for p in link_pats:
        matches = list(p.finditer(page))
        if matches:
            break
    if not matches:
        return []

    out: List[Dict[str, str]] = []
    for i, m in enumerate(matches):
        href = _ddg_unwrap_url(m.group(1))
        title = _strip_html(m.group(2))
        if not href or not title:
            continue
        # Find a snippet near this link.
        end = matches[i + 1].start() if i + 1 < len(matches) else min(len(page), m.end() + 6000)
        window = page[m.end() : end]
        snippet = ""
        for sp in snip_pats:
            sm = sp.search(window)
            if sm:
                snippet = _strip_html(sm.group(1))
                break
        out.append({"title": title, "url": href, "snippet": snippet})
        if len(out) >= int(k):
            break
    return out


def make_web_search_handler(
    *,
    allow_domains: Sequence[str] = (),
    deny_domains: Sequence[str] = (),
    provider: str = "duckduckgo_lite",
    # Production safeguards (defaults are conservative and backwards-compatible)
    cache_ttl_s: float = 300.0,
    cache_max_entries: int = 256,
    enable_cache: bool = True,
    rate_limit_window_s: float = 60.0,
    rate_limit_max_calls: int = 60,
    enable_rate_limit: bool = True,
    max_retries: int = 2,
    backoff_base_s: float = 0.25,
    backoff_max_s: float = 2.0,
    enable_backoff: bool = True,
    circuit_breaker_failure_threshold: int = 5,
    circuit_breaker_open_s: float = 30.0,
    enable_circuit_breaker: bool = True,
) -> Callable[[WebSearchIn], WebSearchOut]:
    allow = [str(d).strip().lower() for d in allow_domains if str(d).strip()]
    deny = [str(d).strip().lower() for d in deny_domains if str(d).strip()]

    def _normalize_query(q: str) -> str:
        q = (q or "").strip()
        q = re.sub(r"\s+", " ", q)
        return q

    def _norm_domains(ds: Sequence[str]) -> tuple[str, ...]:
        return tuple(sorted({_normalize_domain(str(d)) for d in (ds or []) if str(d).strip()}))

    def _cache_key(*, q: str, allow_eff: Sequence[str], deny_eff: Sequence[str], k: int, provider_name: str) -> str:
        # Keyed by normalized query + allow/deny + provider (plus k to avoid returning more than asked).
        aq = _normalize_query(q).lower()
        a = ",".join(_norm_domains(allow_eff))
        d = ",".join(_norm_domains(deny_eff))
        return f"v1|p={provider_name}|k={int(k)}|q={aq}|allow={a}|deny={d}"

    class _TTLCache:
        ttl_s: float
        max_entries: int
        _data: OrderedDict[str, tuple[float, WebSearchOut]]
        _lock: Lock

        def __init__(self, *, ttl_s: float, max_entries: int) -> None:
            self.ttl_s = float(ttl_s)
            self.max_entries = int(max_entries)
            self._data = OrderedDict()
            self._lock = Lock()

        def get(self, key: str) -> WebSearchOut | None:
            if self.ttl_s <= 0 or self.max_entries <= 0:
                return None
            now = time.monotonic()
            with self._lock:
                item = self._data.get(key)
                if not item:
                    return None
                exp, val = item
                if exp <= now:
                    # Expired
                    _ = self._data.pop(key, None)
                    return None
                # LRU touch
                self._data.move_to_end(key, last=True)
                return val

        def set(self, key: str, val: WebSearchOut) -> None:
            if self.ttl_s <= 0 or self.max_entries <= 0:
                return
            now = time.monotonic()
            exp = now + self.ttl_s
            with self._lock:
                self._data[key] = (exp, val)
                self._data.move_to_end(key, last=True)
                # Trim
                while len(self._data) > self.max_entries:
                    _ = self._data.popitem(last=False)

    class _FixedWindowLimiter:
        window_s: float
        max_calls: int
        _lock: Lock
        _window_start: float
        _count: int

        def __init__(self, *, window_s: float, max_calls: int) -> None:
            self.window_s = max(0.0, float(window_s))
            self.max_calls = max(0, int(max_calls))
            self._lock = Lock()
            self._window_start = 0.0
            self._count = 0

        def acquire(self) -> float:
            """
            Returns wait seconds (0 = allowed immediately).
            """
            if self.window_s <= 0 or self.max_calls <= 0:
                return 0.0
            now = time.monotonic()
            with self._lock:
                if self._window_start <= 0.0 or (now - self._window_start) >= self.window_s:
                    self._window_start = now
                    self._count = 0
                if self._count < self.max_calls:
                    self._count += 1
                    return 0.0
                return max(0.0, self.window_s - (now - self._window_start))

    class _CircuitBreaker:
        failure_threshold: int
        open_s: float
        _lock: Lock
        _failures: int
        _open_until: float
        _half_open_in_flight: bool

        def __init__(self, *, failure_threshold: int, open_s: float) -> None:
            self.failure_threshold = max(1, int(failure_threshold))
            self.open_s = max(0.0, float(open_s))
            self._lock = Lock()
            self._failures = 0
            self._open_until = 0.0
            self._half_open_in_flight = False

        def allow(self) -> bool:
            if self.open_s <= 0:
                return True
            now = time.monotonic()
            with self._lock:
                if now < self._open_until:
                    return False
                # Half-open: allow a single probe at a time.
                if self._open_until > 0.0 and not self._half_open_in_flight:
                    self._half_open_in_flight = True
                    return True
                if self._open_until > 0.0 and self._half_open_in_flight:
                    return False
                return True

        def on_success(self) -> None:
            with self._lock:
                self._failures = 0
                self._open_until = 0.0
                self._half_open_in_flight = False

        def on_failure(self) -> None:
            if self.open_s <= 0:
                return
            now = time.monotonic()
            with self._lock:
                self._failures += 1
                self._half_open_in_flight = False
                if self._failures >= self.failure_threshold:
                    self._open_until = now + self.open_s

    @dataclass
    class _Metrics:
        calls: int = 0
        success: int = 0
        failure: int = 0
        cache_hit: int = 0
        rate_limited: int = 0
        circuit_open: int = 0
        # latency histogram buckets (seconds)
        hist_le_0_1: int = 0
        hist_le_0_3: int = 0
        hist_le_1: int = 0
        hist_le_3: int = 0
        hist_le_8: int = 0
        hist_le_20: int = 0
        hist_gt_20: int = 0

        lock: Lock = field(default_factory=Lock, repr=False)

        def observe_latency(self, s: float) -> None:
            with self.lock:
                if s <= 0.1:
                    self.hist_le_0_1 += 1
                elif s <= 0.3:
                    self.hist_le_0_3 += 1
                elif s <= 1.0:
                    self.hist_le_1 += 1
                elif s <= 3.0:
                    self.hist_le_3 += 1
                elif s <= 8.0:
                    self.hist_le_8 += 1
                elif s <= 20.0:
                    self.hist_le_20 += 1
                else:
                    self.hist_gt_20 += 1

    _cache = _TTLCache(ttl_s=float(cache_ttl_s), max_entries=int(cache_max_entries))
    _limiter = _FixedWindowLimiter(window_s=float(rate_limit_window_s), max_calls=int(rate_limit_max_calls))
    _breaker = _CircuitBreaker(
        failure_threshold=int(circuit_breaker_failure_threshold),
        open_s=float(circuit_breaker_open_s),
    )
    _metrics = _Metrics()
    # Dedicated, small pool so web fetches don't monopolize any shared executor.
    _net_pool = cf.ThreadPoolExecutor(max_workers=4, thread_name_prefix="cogos-websearch")

    def _fetch_raw_ddg(query: str, *, k: int, timeout_s: float) -> list[dict[str, str]]:
        # Run in worker thread; enforce an overall timeout so we can trip the circuit breaker on hangs.
        fut: cf.Future[list[dict[str, str]]] = _net_pool.submit(
            _duckduckgo_lite_search,
            query,
            k=k,
            timeout_s=timeout_s,
            raise_on_error=True,
        )
        return fut.result(timeout=float(timeout_s) + 2.0)

    def _is_retryable_http(e: BaseException) -> bool:
        if isinstance(e, urllib.error.HTTPError) and getattr(e, "code", None) == 429:
            return True
        # URLError is too broad; treat explicit timeout-y errors as retryable.
        if isinstance(e, (TimeoutError, cf.TimeoutError)):
            return True
        # socket.timeout is a TimeoutError subclass, but keep explicit check defensive.
        if e.__class__.__name__ == "timeout":
            return True
        return False

    def _backoff_sleep_s(attempt: int) -> float:
        base = max(0.0, float(backoff_base_s))
        cap = max(base, float(backoff_max_s))
        # Exponential backoff with a small deterministic jitter derived from attempt.
        s = min(cap, base * (2.0**attempt))
        jitter = min(0.05, 0.01 * (attempt + 1))
        return max(0.0, s + jitter)

    def _h(inp: WebSearchIn) -> WebSearchOut:
        k = max(WebSearchIn.K_MIN, min(WebSearchIn.K_MAX, int(inp.k)))
        timeout_s = max(WebSearchIn.TIMEOUT_MIN, min(WebSearchIn.TIMEOUT_MAX, float(inp.timeout_s)))
        # Enforce config allow/deny; tool args can only further restrict.
        allow_eff = allow
        if inp.allow_domains:
            tool_allow = [str(d).strip().lower() for d in (inp.allow_domains or []) if str(d).strip()]
            if allow_eff:
                allow_eff = [
                    d for d in allow_eff if any(_domain_matches(d, a) or _domain_matches(a, d) for a in tool_allow)
                ]
            else:
                # If config didn't specify an allowlist, allow the tool to restrict.
                allow_eff = tool_allow
        deny_eff = sorted({*deny, *[str(d).strip().lower() for d in (inp.deny_domains or []) if str(d).strip()]})

        provider_name = str(provider)
        q_norm = _normalize_query(inp.query)
        ck = _cache_key(q=q_norm, allow_eff=allow_eff, deny_eff=deny_eff, k=k, provider_name=provider_name)

        if enable_cache:
            cached = _cache.get(ck)
            if cached is not None:
                with _metrics.lock:
                    _metrics.cache_hit += 1
                logger.debug("web_search cache_hit provider=%s query=%r", provider_name, q_norm)
                return cached

        with _metrics.lock:
            _metrics.calls += 1
            calls_n = _metrics.calls

        if enable_circuit_breaker and not _breaker.allow():
            with _metrics.lock:
                _metrics.circuit_open += 1
                _metrics.failure += 1
            logger.debug("web_search circuit_open provider=%s query=%r", provider_name, q_norm)
            out = WebSearchOut(query=inp.query, provider=provider_name, results=[])
            if enable_cache:
                # Cache the "fast-fail" briefly to reduce stampede when provider is down.
                _cache.set(ck, out)
            return out

        if enable_rate_limit:
            wait_s = _limiter.acquire()
            if wait_s > 0:
                with _metrics.lock:
                    _metrics.rate_limited += 1
                logger.debug("web_search rate_limited provider=%s wait_s=%.3f", provider_name, wait_s)
                # Keep wait bounded; we prefer returning quickly to avoid tying up tool execution.
                time.sleep(min(wait_s, 0.5))

        t0 = time.perf_counter()
        raw: list[dict[str, str]] = []
        ok = False
        last_err: BaseException | None = None
        for attempt in range(0, max(0, int(max_retries)) + 1):
            try:
                if provider_name == "duckduckgo_lite":
                    raw = _fetch_raw_ddg(q_norm, k=k * 3, timeout_s=timeout_s)
                else:
                    # Fallback: preserve previous behavior for unknown providers.
                    raw = _duckduckgo_lite_search(q_norm, k=k * 3, timeout_s=timeout_s)
                ok = True
                last_err = None
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                if enable_backoff and attempt < int(max_retries) and _is_retryable_http(e):
                    time.sleep(_backoff_sleep_s(attempt))
                    continue
                break

        dt_s = time.perf_counter() - t0
        _metrics.observe_latency(dt_s)

        if ok:
            if enable_circuit_breaker:
                _breaker.on_success()
            with _metrics.lock:
                _metrics.success += 1
        else:
            if enable_circuit_breaker:
                _breaker.on_failure()
            with _metrics.lock:
                _metrics.failure += 1
            # Preserve prior behavior: errors just yield empty results.
            raw = []

        results: List[WebSearchResult] = []
        for r in raw:
            url = str(r.get("url") or "").strip()
            if not url:
                continue
            domain = _normalize_domain(urllib.parse.urlparse(url).netloc)
            if not _domain_allowed(domain, allow_eff, deny_eff):
                continue
            trust = _trust_for_domain(domain, allow_eff)
            results.append(
                WebSearchResult(
                    title=str(r.get("title") or "").strip(),
                    url=url,
                    snippet=str(r.get("snippet") or "").strip(),
                    domain=domain,
                    rank=len(results) + 1,
                    trust_score=float(trust),
                )
            )
            if len(results) >= k:
                break
        out = WebSearchOut(query=inp.query, provider=provider_name, results=results)
        if enable_cache:
            _cache.set(ck, out)
        logger.debug(
            "web_search call provider=%s ok=%s latency_s=%.3f results=%d",
            provider_name,
            ok,
            dt_s,
            len(out.results),
        )
        # Periodic snapshot for log-based monitoring (best-effort; avoids per-call spam at INFO).
        if calls_n % 50 == 0:
            with _metrics.lock:
                snap = {
                    "calls": _metrics.calls,
                    "success": _metrics.success,
                    "failure": _metrics.failure,
                    "cache_hit": _metrics.cache_hit,
                    "rate_limited": _metrics.rate_limited,
                    "circuit_open": _metrics.circuit_open,
                    "hist_le_0_1": _metrics.hist_le_0_1,
                    "hist_le_0_3": _metrics.hist_le_0_3,
                    "hist_le_1": _metrics.hist_le_1,
                    "hist_le_3": _metrics.hist_le_3,
                    "hist_le_8": _metrics.hist_le_8,
                    "hist_le_20": _metrics.hist_le_20,
                    "hist_gt_20": _metrics.hist_gt_20,
                }
            logger.info("web_search metrics provider=%s %s", provider_name, snap)

        # Emit a warning on hard failures for visibility.
        if (not ok) and last_err is not None:
            logger.warning(
                "web_search provider=%s failed query=%r err=%s latency_s=%.3f",
                provider_name,
                q_norm,
                repr(last_err),
                dt_s,
            )
        return out

    return _h
