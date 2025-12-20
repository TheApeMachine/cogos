from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

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
from .memory import MemoryStore
from .planner import LLMPlanner, Planner, RulePlanner
from .pyd_compat import _model_dump
from .reasoner import ConservativeReasoner, LLMReasoner, Reasoner, SearchReasoner
from .renderer import Renderer
from .tools import (
    CalcIn,
    CalcOut,
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
    WriteFileIn,
    WriteFileOut,
    calc_handler,
    make_mem_search_handler,
    make_read_file_handler,
    make_write_file_handler,
    now_handler,
)
from .util import jdump
from .verifier import Verifier

from .ir import Plan, StepCreateTask, StepMemorySearch, StepRespond, StepToolCall, StepWriteNote


@dataclass
class AgentConfig:
    db: str = "cogos.db"
    session_id: str = "default"
    embedder: Literal["hash", "st"] = "hash"
    st_model: str = "all-MiniLM-L6-v2"
    llm_backend: Literal["stub", "llama_cpp"] = "stub"
    llama_model: str = ""
    llama_ctx: int = 4096
    llama_threads: Optional[int] = None
    llama_gpu_layers: int = 0
    planner: Literal["rule", "llm"] = "rule"
    reasoner: Literal["conservative", "llm", "search"] = "conservative"
    search_samples: int = 4
    allow_side_effects: bool = False
    read_root: Tuple[str, ...] = (".",)
    write_root: Tuple[str, ...] = (".",)
    prune_episodes: bool = False
    episode_keep_last: int = 200
    episode_prune_batch: int = 50
    episode_digest_chars: int = 280
    episode_digest_confidence: float = 0.55
    background: bool = True
    json_logs: bool = False
    log_level: str = "INFO"
    initiative_threshold: float = 0.62


class CogOS:
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
        self.verifier = Verifier(self.memory, require_spans=True, min_span_hits=0.5)
        self.renderer = Renderer(self.memory)

        # LLM
        if cfg.llm_backend == "llama_cpp":
            if not cfg.llama_model:
                raise ValueError("--llama-model is required when --llm-backend llama_cpp")
            self.llm: ChatModel = LlamaCppChatModel(
                cfg.llama_model,
                n_ctx=cfg.llama_ctx,
                n_threads=cfg.llama_threads,
                n_gpu_layers=cfg.llama_gpu_layers,
            )
        else:
            self.llm = StubChatModel()

        # planner
        if cfg.planner == "llm" and cfg.llm_backend != "stub":
            self.planner: Planner = LLMPlanner(self.llm, fallback=RulePlanner())
        else:
            self.planner = RulePlanner()

        # reasoner
        if cfg.reasoner == "llm" and cfg.llm_backend != "stub":
            self.reasoner: Reasoner = LLMReasoner(self.llm)
        elif cfg.reasoner == "search" and cfg.llm_backend != "stub":
            self.reasoner = SearchReasoner(LLMReasoner(self.llm), samples=cfg.search_samples)
        else:
            self.reasoner = ConservativeReasoner()

        # background
        self.bg: Optional[BackgroundRunner] = None
        if cfg.background:
            ctx = DaemonContext(
                memory=self.memory,
                tools=self.tools,
                initiative=self.initiative,
                bus=self.bus,
                llm=None if cfg.llm_backend == "stub" else self.llm,
                session_id=cfg.session_id,
            )
            daemons: List[Daemon] = [ReflectionDaemon(), ConnectionMiner(), ConsistencyAuditor(), TaskSolverDaemon()]
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
            "CogOS started",
            extra={
                "extra": {
                    "db": cfg.db,
                    "embedder": embedder.name,
                    "llm": cfg.llm_backend,
                    "planner": cfg.planner,
                    "reasoner": cfg.reasoner,
                    "fts": self.memory._fts_ok,
                }
            },
        )

    def close(self) -> None:
        if self.bg:
            self.bg.stop()
        self.memory.close()
        log.info("CogOS stopped")

    def _register_tools(self) -> None:
        self.tools.register(
            ToolSpec("calc", "Safely evaluate arithmetic expressions.", CalcIn, CalcOut, calc_handler, side_effects=False)
        )
        self.tools.register(ToolSpec("now", "Get current local time.", NowIn, NowOut, now_handler, side_effects=False))
        self.tools.register(
            ToolSpec(
                "memory_search",
                "Search notes/evidence/skills (hybrid lexical+vector).",
                MemSearchIn,
                MemSearchOut,
                make_mem_search_handler(self.memory),
                side_effects=False,
            )
        )
        self.tools.register(
            ToolSpec(
                "read_text_file",
                "Read a UTF-8 text file under allowed roots.",
                ReadFileIn,
                ReadFileOut,
                make_read_file_handler(self.cfg.read_root),
                side_effects=False,
            )
        )
        self.tools.register(
            ToolSpec(
                "write_text_file",
                "Write a UTF-8 text file under allowed roots.",
                WriteFileIn,
                WriteFileOut,
                make_write_file_handler(self.cfg.write_root),
                side_effects=True,
            )
        )

    def handle(self, user_text: str) -> tuple[str, List[Dict[str, Any]]]:
        # episodic log: user
        ep_user = self.memory.add_episode(self.cfg.session_id, "user", user_text, metadata={})
        self.bus.publish("episode_added", {"episode_id": ep_user, "role": "user"})

        plan: Plan = self.planner.plan(user_text, tools=self.tools, memory=self.memory)

        tool_outcomes: List[ToolOutcome] = []
        evidence_ids: List[str] = []
        memory_hits: Dict[str, Any] = {"notes": [], "evidence": [], "skills": []}

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
                    memory_hits = out.output

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
                self.bus.publish("note_added", {"note_id": nid})
                ev = self.memory.add_evidence("note_write", jdump({"note_id": nid, "title": step.title}), metadata={})
                evidence_ids.append(ev)

            elif isinstance(step, StepCreateTask):
                tid = self.memory.add_task(step.title, step.description, priority=step.priority, payload=step.payload)
                self.bus.publish("task_added", {"task_id": tid})
                ev = self.memory.add_evidence("task_create", jdump({"task_id": tid, "title": step.title}), metadata={})
                evidence_ids.append(ev)

            elif isinstance(step, StepRespond):
                pass

        # Evidence map for reasoner
        evidence_map: Dict[str, str] = {}
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

        # episodic log: assistant
        ep_bot = self.memory.add_episode(
            self.cfg.session_id,
            "assistant",
            response,
            metadata={"plan": _model_dump(plan)},
        )
        self.bus.publish("episode_added", {"episode_id": ep_bot, "role": "assistant"})

        proactive = self.initiative.poll(limit=3)
        return response, proactive

