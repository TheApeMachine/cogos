"""CogOS agent runtime.

This module wires together the memory store, tool bus, planner, reasoner, verifier,
and renderer into a single `CogOS` façade.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast, final

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
from .event_bus import EventBus
from .initiative import InitiativeManager
from .llm import ChatModel, LlamaCppChatModel, StubChatModel
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
from .util import jdump
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
    llm_backend: Literal["stub", "llama_cpp"] = "stub"
    llama_model: str = ""
    llama_model_dir: str = "models"
    llama_auto_download: bool = False
    llama_hf_repo: str = DEFAULT_HF_MODEL.repo_id
    llama_hf_file: str = DEFAULT_HF_MODEL.filename
    llama_hf_rev: str = DEFAULT_HF_MODEL.revision
    llama_ctx: int = 4096
    llama_threads: int | None = None
    llama_gpu_layers: int = 0
    planner: Literal["rule", "llm"] = "rule"
    reasoner: Literal["conservative", "llm", "search"] = "conservative"
    search_samples: int = 4
    allow_side_effects: bool = False
    allow_web_search: bool = False
    auto_research: bool = False
    web_allow_domains: tuple[str, ...] = ("wikipedia.org", "arxiv.org", "github.com")
    web_deny_domains: tuple[str, ...] = ()
    min_evidence_trust: float = 0.0
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
    initiative_threshold: float = 0.62


@final
class CogOS:
    """Main runtime: plan → execute → reason → verify → render."""

    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        setup_logging(cfg.log_level, json_logs=cfg.json_logs)

        # embedder
        if cfg.embedder == "st":
            embedder: EmbeddingModel = SentenceTransformerEmbed(cfg.st_model)
        else:
            embedder = HashEmbed(384)

        self.bus = EventBus()
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

        # LLM
        if cfg.llm_backend == "llama_cpp":
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
            self.llm: ChatModel = LlamaCppChatModel(
                model_path,
                n_ctx=cfg.llama_ctx,
                n_threads=cfg.llama_threads,
                n_gpu_layers=cfg.llama_gpu_layers,
            )
        else:
            self.llm = StubChatModel()

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
            if cfg.llm_backend == "stub":
                raise ValueError("--planner llm requires --llm-backend != stub")
            self.planner = LLMPlanner(self.llm)
        else:
            self.planner = RulePlanner()

        # reasoner
        if cfg.reasoner == "llm":
            if cfg.llm_backend == "stub":
                raise ValueError("--reasoner llm requires --llm-backend != stub")
            self.reasoner = LLMReasoner(self.llm)
        elif cfg.reasoner == "search":
            if cfg.llm_backend == "stub":
                raise ValueError("--reasoner search requires --llm-backend != stub")
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
            cfg.llm_backend,
            cfg.planner,
            cfg.reasoner,
            self.memory.fts_ok,
            extra={
                "extra": {
                    "db": cfg.db,
                    "embedder": embedder.name,
                    "llm": cfg.llm_backend,
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

            return {
                "result_count": int(len(all_items)),
                "trust_score": float(max(trust_scores or [0.0])),
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
                    )(cast(WebSearchIn, m)),
                    side_effects=False,
                    source_type="web_search",
                    default_trust_score=0.35,
                    evidence_metadata_builder=_web_search_metadata,
                )
            )

    def handle(self, user_text: str) -> tuple[str, list[JsonObject]]:
        """Handle a single user turn and return `(response, proactive_items)`."""

        # episodic log: user
        ep_user = self.memory.add_episode(self.cfg.session_id, "user", user_text, metadata={})
        _ = self.bus.publish("episode_added", {"episode_id": ep_user, "role": "user"})

        plan: Plan = self.planner.plan(user_text, tools=self.tools, memory=self.memory)

        tool_outcomes: list[ToolOutcome] = []
        evidence_ids: list[str] = []
        memory_hits: JsonObject = {"notes": [], "evidence": [], "skills": []}

        # Execute plan
        for step in plan.steps:
            if isinstance(step, StepMemorySearch):
                out = self.tools.execute(
                    ToolCall(name="memory_search", arguments={"query": step.query, "k": step.k})
                )
                tool_outcomes.append(out)
                if out.evidence_id:
                    evidence_ids.append(out.evidence_id)
                if out.ok:
                    memory_hits = cast(JsonObject, out.output)

            elif isinstance(step, StepToolCall):
                out = self.tools.execute(ToolCall(name=step.tool, arguments=step.arguments))
                tool_outcomes.append(out)
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
                    metadata={},
                )
                evidence_ids.append(ev)

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
                    metadata={},
                )
                evidence_ids.append(ev)

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
            if had_mem_search and empty_hits and (not had_web_search):
                out = self.tools.execute(
                    ToolCall(name="web_search", arguments={"query": user_text, "k": 5})
                )
                tool_outcomes.append(out)
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
        verified = self.verifier.verify(proposed)
        response = self.renderer.render(verified)

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
