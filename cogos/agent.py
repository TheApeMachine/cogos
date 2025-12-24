"""CogOS agent runtime.

This module wires together the memory store, tool bus, planner, reasoner, verifier,
and renderer into a single `CogOS` façade.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Literal, cast, final

from .daemons import (
    BackgroundRunner,
    ConnectionMiner,
    ConsistencyAuditor,
    Daemon,
    DaemonContext,
    PruningDaemon,
    ReflectionDaemon,
    TaskSolverDaemon,
)
from .embeddings import EmbeddingModel, HashEmbed, SentenceTransformerEmbed
from .event_bus import Event, EventBus
from .initiative import InitiativeManager
from .llm import ChatModel, LlamaCppChatModel, OllamaChatModel, StubChatModel
from .logging_utils import log, setup_logging
from .model import DEFAULT_HF_MODEL, HFModelSpec, resolve_llama_model_path
from .memory import MemoryStore
from .notary import Notary
from .planner import LLMPlanner, RulePlanner
from . import pyd_compat
from .reasoner import ConservativeReasoner, LLMReasoner, SearchReasoner
from .renderer import Renderer
from .tools import (
    CalcIn,
    CalcOut,
    CountCharsIn,
    CountCharsOut,
    MemSearchIn,
    MemSearchOut,
    NowIn,
    NowOut,
    ReadFileIn,
    ReadFileOut,
    ToolBus,
    ToolCall,
    ToolOutcome,
    ToolSpec,
    WebSearchIn,
    WebSearchOut,
    WriteFileIn,
    WriteFileOut,
    calc_handler,
    count_chars_handler,
    make_mem_search_handler,
    make_read_file_handler,
    make_web_search_handler,
    make_write_file_handler,
    now_handler,
)
from .util import jdump, toks
from .verifier import Verifier

from .ir import Plan, StepCreateTask, StepMemorySearch, StepToolCall, StepWriteNote


JsonValue = str | int | float | bool | None | dict[str, "JsonValue"] | list["JsonValue"]
JsonObject = dict[str, JsonValue]


@dataclass
class AgentConfig:
    """Configuration for `CogOS`."""

    db: str = "cogos.db"
    session_id: str = "default"
    embedder: Literal["hash", "st"] = "hash"
    st_model: str = "all-MiniLM-L6-v2"
    llm_backend: Literal["auto", "stub", "llama_cpp", "ollama"] = "auto"
    llama_model: str = ""
    llama_model_dir: str = "models"
    llama_auto_download: bool = False
    llama_hf_repo: str = DEFAULT_HF_MODEL.repo_id
    llama_hf_file: str = DEFAULT_HF_MODEL.filename
    llama_hf_rev: str = DEFAULT_HF_MODEL.revision
    llama_ctx: int = 4096
    llama_threads: int | None = None
    llama_gpu_layers: int = 0
    llama_verbose: bool = False
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = ""
    planner: Literal["rule", "llm"] = "llm"
    reasoner: Literal["conservative", "llm", "search"] = "search"
    search_samples: int = 4
    allow_side_effects: bool = False
    allow_web_search: bool = False
    auto_research: bool = False
    # Default: no allowlist (permit any domain), but downstream trust scoring + verifier thresholds
    # still gate what can be used as evidence.
    web_allow_domains: tuple[str, ...] = ()
    web_deny_domains: tuple[str, ...] = ()
    # Default: reject low-trust evidence such as conversation-derived notes.
    # Web allowlisted domains typically score >= 0.6; conversation fragments cap at ~0.35.
    min_evidence_trust: float = 0.5
    notary: bool = False
    notary_priority: int = 10
    read_root: tuple[str, ...] = (".",)
    write_root: tuple[str, ...] = (".",)
    prune_episodes: bool = False
    episode_keep_last: int = 200
    episode_prune_batch: int = 50
    episode_digest_chars: int = 280
    episode_digest_confidence: float = 0.55
    background: bool = True
    json_logs: bool = False
    log_level: str = "INFO"
    log_file: str = "cogos.log"
    log_file_level: str = "DEBUG"
    log_file_max_bytes: int = 10_000_000
    log_file_backup_count: int = 3
    initiative_threshold: float = 0.62


@final
class CogOS:
    """Main runtime: plan → execute → reason → verify → render."""

    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        setup_logging(
            cfg.log_level,
            json_logs=cfg.json_logs,
            log_file=cfg.log_file,
            log_file_level=cfg.log_file_level,
            log_file_max_bytes=cfg.log_file_max_bytes,
            log_file_backup_count=cfg.log_file_backup_count,
        )
        self.last_trace: dict[str, Any] | None = None

        # embedder
        if cfg.embedder == "st":
            embedder: EmbeddingModel = SentenceTransformerEmbed(cfg.st_model)
        else:
            embedder = HashEmbed(384)

        self.bus = EventBus()
        self._recent_bus_events: "deque[JsonObject]" = deque(maxlen=200)

        def _record_bus_event(evt: Event) -> None:
            try:
                # Store a compact record; payload can be inspected via /show events live if needed.
                self._recent_bus_events.append(
                    {"type": str(evt.type), "ts": float(evt.ts), "payload": cast(JsonObject, dict(evt.payload))}
                )
            except Exception:  # pylint: disable=broad-exception-caught
                return

        self.bus.add_listener(_record_bus_event)
        self.memory = MemoryStore(cfg.db, embedder=embedder)

        self.tools = ToolBus(self.memory, self.bus, allow_side_effects=cfg.allow_side_effects)
        self._register_tools()

        self.initiative = InitiativeManager(self.memory, threshold=cfg.initiative_threshold)
        self.verifier = Verifier(
            self.memory,
            require_spans=True,
            min_span_hits=0.5,
            min_trust_score=cfg.min_evidence_trust,
        )
        self.renderer = Renderer(self.memory)
        self.notary = Notary(self.memory, priority=cfg.notary_priority) if cfg.notary else None

        # LLM (auto resolves to a real backend; never silently "show up as AI" while using stub).
        self.llm: ChatModel
        llm_errs: list[str] = []

        def _try_llama_cpp() -> ChatModel:
            default_spec = HFModelSpec(
                repo_id=cfg.llama_hf_repo,
                filename=cfg.llama_hf_file,
                revision=cfg.llama_hf_rev,
            )
            model_path = resolve_llama_model_path(
                cfg.llama_model,
                auto_download=cfg.llama_auto_download,
                model_dir=cfg.llama_model_dir,
                default_spec=default_spec,
            )
            return LlamaCppChatModel(
                model_path,
                n_ctx=cfg.llama_ctx,
                n_threads=cfg.llama_threads,
                n_gpu_layers=cfg.llama_gpu_layers,
                verbose=cfg.llama_verbose,
            )

        def _try_ollama() -> ChatModel:
            return OllamaChatModel(host=cfg.ollama_host, model=cfg.ollama_model)

        if cfg.llm_backend == "stub":
            self.llm = StubChatModel()
        elif cfg.llm_backend == "ollama":
            self.llm = _try_ollama()
        elif cfg.llm_backend == "llama_cpp":
            self.llm = _try_llama_cpp()
        else:
            # auto: try llama.cpp first, then Ollama, otherwise fail with a helpful error.
            try:
                self.llm = _try_llama_cpp()
            except Exception as e1:  # pylint: disable=broad-exception-caught
                llm_errs.append(f"llama_cpp: {type(e1).__name__}: {e1}")
                try:
                    self.llm = _try_ollama()
                except Exception as e2:  # pylint: disable=broad-exception-caught
                    llm_errs.append(f"ollama: {type(e2).__name__}: {e2}")
                    raise RuntimeError(
                        "No usable LLM backend found.\n\n"
                        "Fix options:\n"
                        "- Install llama-cpp-python: pip install -r requirements-llama.txt\n"
                        "- OR run Ollama locally and install a model: `ollama pull llama3.2` (then use --llm-backend ollama)\n"
                        "- OR explicitly run in deterministic mode: --llm-backend stub\n\n"
                        "Details:\n- "
                        + "\n- ".join(llm_errs)
                    ) from e2

        # Configure planner grammar constraints (when supported): only allow runnable tools.
        if isinstance(self.llm, LlamaCppChatModel) and hasattr(self.llm, "set_plan_tool_names"):
            try:
                allowed_tool_names = [
                    t["name"]
                    for t in self.tools.list_tools()
                    if (self.tools.allow_side_effects or (not bool(t.get("side_effects"))))
                ]
                self.llm.set_plan_tool_names(allowed_tool_names)
            except (ValueError, TypeError, AttributeError):  # noqa: BLE001
                # Grammar configuration is a best-effort enhancement; never block startup.
                pass

        # planner
        if cfg.planner == "llm":
            if isinstance(self.llm, StubChatModel):
                raise ValueError("--planner llm requires a real LLM backend (try --llm-backend llama_cpp|ollama|auto)")
            self.planner = LLMPlanner(self.llm)
        else:
            self.planner = RulePlanner()

        # reasoner
        if cfg.reasoner == "llm":
            if isinstance(self.llm, StubChatModel):
                raise ValueError("--reasoner llm requires a real LLM backend (try --llm-backend llama_cpp|ollama|auto)")
            self.reasoner = LLMReasoner(self.llm)
        elif cfg.reasoner == "search":
            if isinstance(self.llm, StubChatModel):
                raise ValueError("--reasoner search requires a real LLM backend (try --llm-backend llama_cpp|ollama|auto)")
            self.reasoner = SearchReasoner(LLMReasoner(self.llm), samples=cfg.search_samples)
        else:
            self.reasoner = ConservativeReasoner()

        # background
        self.bg: BackgroundRunner | None = None
        if cfg.background:
            ctx = DaemonContext(
                memory=self.memory,
                tools=self.tools,
                initiative=self.initiative,
                bus=self.bus,
                llm=None if cfg.llm_backend == "stub" else self.llm,
                session_id=cfg.session_id,
            )
            daemons: list[Daemon] = [
                ReflectionDaemon(),
                ConnectionMiner(),
                ConsistencyAuditor(),
                TaskSolverDaemon(),
            ]
            if cfg.prune_episodes:
                daemons.insert(
                    1,
                    PruningDaemon(
                        keep_last=cfg.episode_keep_last,
                        batch=cfg.episode_prune_batch,
                        digest_chars=cfg.episode_digest_chars,
                        confidence=cfg.episode_digest_confidence,
                    ),
                )
            self.bg = BackgroundRunner(self.bus, ctx, daemons)
            self.bg.start()

        log.info(
            "CogOS started (db=%s, embedder=%s, llm=%s, planner=%s, reasoner=%s, fts=%s)",
            cfg.db,
            embedder.name,
            getattr(self.llm, "name", cfg.llm_backend),
            cfg.planner,
            cfg.reasoner,
            self.memory.fts_ok,
            extra={
                "extra": {
                    "db": cfg.db,
                    "embedder": embedder.name,
                    "llm": getattr(self.llm, "name", cfg.llm_backend),
                    "planner": cfg.planner,
                    "reasoner": cfg.reasoner,
                    "fts": self.memory.fts_ok,
                }
            },
        )

    def close(self) -> None:
        """Stop background daemons and close the underlying storage."""

        if self.bg:
            self.bg.stop()
        self.memory.close()
        log.info("CogOS stopped")

    def _register_tools(self) -> None:
        def _mem_search_metadata(_inp: object, out: object) -> JsonObject:
            o = cast(MemSearchOut, out)
            notes = cast(list[JsonObject], getattr(o, "notes", []) or [])
            evidence = cast(list[JsonObject], getattr(o, "evidence", []) or [])
            skills = cast(list[JsonObject], getattr(o, "skills", []) or [])
            all_items: list[JsonObject] = notes + evidence + skills

            trust_scores: list[float] = []
            for item in all_items:
                ts = item.get("trust_score", 0.0)
                if isinstance(ts, (int, float)):
                    trust_scores.append(float(ts))

            # Do not override the tool execution trust_score: `memory_search` is a retrieval
            # tool, not a source of truth. We record the max trust of returned items as
            # a separate metadata field for debugging/inspection.
            return {
                "result_count": int(len(all_items)),
                "max_item_trust_score": float(max(trust_scores or [0.0])),
            }

        def _web_search_metadata(_inp: object, out: object) -> JsonObject:
            o = cast(WebSearchOut, out)
            provider = getattr(o, "provider", "unknown")
            results = getattr(o, "results", []) or []

            urls: list[JsonValue] = []
            trust_scores: list[float] = []
            for r in results[:5]:
                url = getattr(r, "url", "")
                if isinstance(url, str) and url:
                    urls.append(url)
                ts = getattr(r, "trust_score", 0.0)
                if isinstance(ts, (int, float)):
                    trust_scores.append(float(ts))

            return {
                "provider": str(provider) if provider is not None else "unknown",
                "result_count": int(len(results)),
                "source_urls": urls,
                "trust_score": float(max(trust_scores or [0.0])),
            }

        self.tools.register(
            ToolSpec(
                "calc",
                "Safely evaluate arithmetic expressions.",
                CalcIn,
                CalcOut,
                lambda m: calc_handler(cast(CalcIn, m)),
                side_effects=False,
            )
        )
        self.tools.register(
            ToolSpec(
                "count_chars",
                "Count occurrences of a character in a string.",
                CountCharsIn,
                CountCharsOut,
                lambda m: count_chars_handler(cast(CountCharsIn, m)),
                side_effects=False,
            )
        )
        self.tools.register(
            ToolSpec(
                "now",
                "Get current local time.",
                NowIn,
                NowOut,
                lambda m: now_handler(cast(NowIn, m)),
                side_effects=False,
            )
        )
        self.tools.register(
            ToolSpec(
                "memory_search",
                "Search notes/evidence/skills (hybrid lexical+vector).",
                MemSearchIn,
                MemSearchOut,
                lambda m: make_mem_search_handler(self.memory)(cast(MemSearchIn, m)),
                side_effects=False,
                source_type="memory_search",
                default_trust_score=0.5,
                evidence_metadata_builder=_mem_search_metadata,
            )
        )
        self.tools.register(
            ToolSpec(
                "read_text_file",
                "Read a UTF-8 text file under allowed roots.",
                ReadFileIn,
                ReadFileOut,
                lambda m: make_read_file_handler(self.cfg.read_root)(cast(ReadFileIn, m)),
                side_effects=False,
            )
        )
        self.tools.register(
            ToolSpec(
                "write_text_file",
                "Write a UTF-8 text file under allowed roots.",
                WriteFileIn,
                WriteFileOut,
                lambda m: make_write_file_handler(self.cfg.write_root)(cast(WriteFileIn, m)),
                side_effects=True,
            )
        )
        if self.cfg.allow_web_search:
            self.tools.register(
                ToolSpec(
                    "web_search",
                    "Search the web (best-effort, allowlist filtered).",
                    WebSearchIn,
                    WebSearchOut,
                    lambda m: make_web_search_handler(
                        allow_domains=self.cfg.web_allow_domains,
                        deny_domains=self.cfg.web_deny_domains,
                        min_result_trust=self.cfg.min_evidence_trust,  # pylint: disable=unexpected-keyword-arg
                    )(cast(WebSearchIn, m)),
                    side_effects=False,
                    source_type="web_search",
                    default_trust_score=0.35,
                    evidence_metadata_builder=_web_search_metadata,
                )
            )

    def handle(
        self,
        user_text: str,
        *,
        on_trace: Callable[[str, JsonObject], None] | None = None,
    ) -> tuple[str, list[JsonObject]]:
        """Handle a single user turn and return `(response, proactive_items)`."""

        trace_events: list[JsonObject] = []
        bus_turn_events: list[JsonObject] = []

        def _record_turn_bus_event(evt: Event) -> None:
            try:
                bus_turn_events.append(
                    {"type": str(evt.type), "ts": float(evt.ts), "payload": cast(JsonObject, dict(evt.payload))}
                )
            except Exception:  # pylint: disable=broad-exception-caught
                return

        # Capture events that happen during this turn (daemons may also emit shortly after).
        self.bus.add_listener(_record_turn_bus_event)

        def _trace(kind: str, payload: JsonObject) -> None:
            try:
                trace_events.append({"kind": str(kind), "payload": payload})
            except Exception:  # pylint: disable=broad-exception-caught
                # Tracing is best-effort.
                pass
            if on_trace is None:
                return
            try:
                on_trace(kind, payload)
            except Exception:  # pylint: disable=broad-exception-caught
                # Trace hooks are best-effort; never break the agent loop.
                return

        # episodic log: user
        ep_user = self.memory.add_episode(self.cfg.session_id, "user", user_text, metadata={})
        _ = self.bus.publish("episode_added", {"episode_id": ep_user, "role": "user"})

        plan: Plan = self.planner.plan(user_text, tools=self.tools, memory=self.memory)
        _trace("plan", cast(JsonObject, pyd_compat.model_dump(plan)))

        tool_outcomes: list[ToolOutcome] = []
        evidence_ids: list[str] = []
        memory_hits: JsonObject = {"notes": [], "evidence": [], "skills": []}

        def _max_hit_trust(hits: JsonObject) -> float:
            best = 0.0
            for k in ("notes", "evidence", "skills"):
                items = hits.get(k) or []
                if not isinstance(items, list):
                    continue
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    ts = it.get("trust_score", 0.0)
                    if isinstance(ts, (int, float)):
                        best = max(best, float(ts))
            return float(best)

        def _max_hit_relevance(hits: JsonObject, query: str) -> float:
            """
            Heuristic: maximum token overlap between the query and any returned hit snippet/title.
            """
            q = str(query or "")
            qt = set(toks(q))
            if not qt:
                return 0.0

            def overlap(s: str) -> float:
                st = set(toks(s))
                if not st:
                    return 0.0
                return len(st & qt) / (len(st | qt) or 1)

            best = 0.0
            for k in ("notes", "evidence", "skills"):
                items = hits.get(k) or []
                if not isinstance(items, list):
                    continue
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    # Common fields across our hit shapes.
                    parts: list[str] = []
                    for f in ("title", "name", "kind", "domain", "content_snip", "snippet", "source_type"):
                        v = it.get(f, "")
                        if isinstance(v, str) and v:
                            parts.append(v)
                    if parts:
                        best = max(best, max(overlap(p) for p in parts))
            return float(best)

        def _eligible_hit_count(hits: JsonObject, query: str) -> int:
            """
            Count hits that are both relevant and trusted enough to avoid web_search.
            """
            min_trust = float(self.cfg.min_evidence_trust)
            qt = set(toks(str(query or "")))
            if not qt:
                return 0

            def overlap(s: str) -> float:
                st = set(toks(s))
                if not st:
                    return 0.0
                return len(st & qt) / (len(st | qt) or 1)

            def is_conversation_note(it: dict[str, Any]) -> bool:
                tags = it.get("tags") or []
                if isinstance(tags, list):
                    tags_l = {str(t).lower() for t in tags}
                    if "conversation" in tags_l or "episode_digest" in tags_l:
                        return True
                return False

            REL = 0.35
            n = 0
            for k in ("notes", "evidence", "skills"):
                items = hits.get(k) or []
                if not isinstance(items, list):
                    continue
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    ts = it.get("trust_score", 0.0)
                    trust = float(ts) if isinstance(ts, (int, float)) else 0.0
                    if trust < min_trust:
                        continue
                    if k == "notes" and is_conversation_note(it):
                        continue
                    parts: list[str] = []
                    for f in ("title", "name", "kind", "domain", "content_snip", "snippet", "source_type"):
                        v = it.get(f, "")
                        if isinstance(v, str) and v:
                            parts.append(v)
                    rel = max((overlap(p) for p in parts), default=0.0)
                    if rel >= REL:
                        n += 1
            return int(n)

        # Execute plan
        for step in plan.steps:
            if isinstance(step, StepMemorySearch):
                _trace("tool_call", {"tool": "memory_search", "arguments": {"query": step.query, "k": step.k}})
                out = self.tools.execute(
                    ToolCall(name="memory_search", arguments={"query": step.query, "k": step.k})
                )
                tool_outcomes.append(out)
                _trace("tool_outcome", cast(JsonObject, pyd_compat.model_dump(out)))
                if out.evidence_id:
                    evidence_ids.append(out.evidence_id)
                if out.ok:
                    memory_hits = cast(JsonObject, out.output)

            elif isinstance(step, StepToolCall):
                _trace("tool_call", {"tool": step.tool, "arguments": cast(JsonObject, step.arguments)})
                out = self.tools.execute(ToolCall(name=step.tool, arguments=step.arguments))
                tool_outcomes.append(out)
                _trace("tool_outcome", cast(JsonObject, pyd_compat.model_dump(out)))
                if out.evidence_id:
                    evidence_ids.append(out.evidence_id)

            elif isinstance(step, StepWriteNote):
                nid = self.memory.add_note(
                    step.title,
                    step.content,
                    tags=step.tags,
                    source_ids=[ep_user],
                    confidence=step.confidence,
                )
                _ = self.bus.publish("note_added", {"note_id": nid})
                ev = self.memory.add_evidence(
                    "note_write",
                    jdump({"note_id": nid, "title": step.title}),
                    metadata={"source_type": "internal", "trust_score": 0.2},
                )
                evidence_ids.append(ev)
                _trace("evidence", {"evidence_id": ev, "kind": "note_write"})

            elif isinstance(step, StepCreateTask):
                tid = self.memory.add_task(
                    step.title,
                    step.description,
                    priority=step.priority,
                    payload=step.payload,
                )
                _ = self.bus.publish("task_added", {"task_id": tid})
                ev = self.memory.add_evidence(
                    "task_create",
                    jdump({"task_id": tid, "title": step.title}),
                    metadata={"source_type": "internal", "trust_score": 0.2},
                )
                evidence_ids.append(ev)
                _trace("evidence", {"evidence_id": ev, "kind": "task_create"})

            else:
                # StepRespond (or unknown future step types) is handled after the loop.
                pass

        # Optional self-hydration: if memory search is empty and web_search is
        # enabled, gather evidence.
        if self.cfg.auto_research and self.cfg.allow_web_search:
            had_mem_search = any(o.ok and o.tool == "memory_search" for o in tool_outcomes)
            had_web_search = any(o.ok and o.tool == "web_search" for o in tool_outcomes)
            empty_hits = not (
                memory_hits.get("notes") or memory_hits.get("evidence") or memory_hits.get("skills")
            )
            max_hit_trust = _max_hit_trust(memory_hits)
            max_hit_rel = _max_hit_relevance(memory_hits, user_text)
            eligible_hits = _eligible_hit_count(memory_hits, user_text)

            # Only auto-research for non-trivial queries; avoid going online for "hi".
            q_tokens = toks(user_text)
            non_trivial = len(q_tokens) >= 3

            # Treat low-trust OR low-relevance hits as "effectively empty" for auto-research.
            insufficient_hits = empty_hits or (eligible_hits == 0)
            _trace(
                "auto_research_gate",
                {
                    "had_mem_search": bool(had_mem_search),
                    "had_web_search": bool(had_web_search),
                    "empty_hits": bool(empty_hits),
                    "max_hit_trust": float(max_hit_trust),
                    "max_hit_relevance": float(max_hit_rel),
                    "eligible_hit_count": int(eligible_hits),
                    "token_count": int(len(q_tokens)),
                    "min_evidence_trust": float(self.cfg.min_evidence_trust),
                    "insufficient_hits": bool(insufficient_hits),
                    "non_trivial": bool(non_trivial),
                },
            )
            if had_mem_search and non_trivial and insufficient_hits and (not had_web_search):
                _trace("tool_call", {"tool": "web_search", "arguments": {"query": user_text, "k": 5}})
                out = self.tools.execute(
                    ToolCall(name="web_search", arguments={"query": user_text, "k": 5})
                )
                tool_outcomes.append(out)
                _trace("tool_outcome", cast(JsonObject, pyd_compat.model_dump(out)))
                if out.evidence_id:
                    evidence_ids.append(out.evidence_id)

        # Evidence map for reasoner
        evidence_map: dict[str, str] = {}
        for eid in evidence_ids:
            ev = self.memory.get_evidence(eid)
            if ev:
                evidence_map[eid] = ev["content"]

        proposed = self.reasoner.propose(
            user_text,
            plan=plan,
            evidence_map=evidence_map,
            memory_hits=memory_hits,
            tool_outcomes=tool_outcomes,
        )
        _trace(
            "proposed",
            {
                "claim_count": int(len(getattr(proposed, "claims", []) or [])),
                "draft": str(getattr(proposed, "draft", "") or "")[:240],
            },
        )
        verified = self.verifier.verify(proposed)
        _trace("verified", cast(JsonObject, pyd_compat.model_dump(verified)))
        response = self.renderer.render(verified)

        # Store last trace for TUI introspection.
        try:
            self.last_trace = {
                "plan": cast(JsonObject, pyd_compat.model_dump(plan)),
                "events": trace_events,
                "bus_events_turn": bus_turn_events,
                "bus_events_recent": list(self._recent_bus_events)[-40:],
                "tool_outcomes": [cast(JsonObject, pyd_compat.model_dump(o)) for o in tool_outcomes],
                "evidence_ids": [str(e) for e in evidence_ids],
                "verified": cast(JsonObject, pyd_compat.model_dump(verified)),
            }
        except Exception:  # pylint: disable=broad-exception-caught
            self.last_trace = None
        finally:
            # Always detach the per-turn bus listener.
            try:
                self.bus.remove_listener(_record_turn_bus_event)
            except Exception:  # pylint: disable=broad-exception-caught
                pass

        # Notary: if auto-research ran and we still can't verify, cut the hard-line and escalate.
        if (
            self.notary is not None
            and (not verified.ok)
            and self.cfg.auto_research
            and self.cfg.allow_web_search
            and any(o.tool == "web_search" for o in tool_outcomes)
        ):
            _ = self.notary.escalate(
                user_text=user_text,
                plan=plan,
                verified=verified,
                tool_outcomes=tool_outcomes,
                reason="abstained after web_search steering",
            )
            response = (
                "I can’t verify any claims from trusted evidence. "
                "I’ve flagged this for human review and will not proceed further on this thread."
            )

        # episodic log: assistant
        ep_bot = self.memory.add_episode(
            self.cfg.session_id,
            "assistant",
            response,
            metadata={"plan": pyd_compat.model_dump(plan)},
        )
        _ = self.bus.publish(
            "episode_added",
            {"episode_id": ep_bot, "role": "assistant"},
        )

        proactive = cast(list[JsonObject], self.initiative.poll(limit=3))
        return response, proactive
