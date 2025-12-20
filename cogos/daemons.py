from __future__ import annotations

import datetime as dt
import threading
from typing import Any, Dict, List, Optional

from .event_bus import Event, EventBus
from .initiative import ProactiveCandidate
from .logging_utils import log
from .pyd_compat import BaseModel
from .tools import ToolCall
from .util import jdump, short, toks, utc_ts


class DaemonContext(BaseModel):
    memory: Any
    tools: Any
    initiative: Any
    bus: Any
    llm: Optional[Any] = None
    session_id: str = "default"


class Daemon:
    name: str = "daemon"
    tick_every_s: float = 5.0

    def on_event(self, evt: Event, ctx: DaemonContext) -> None:
        return

    def tick(self, ctx: DaemonContext) -> None:
        return


class ReflectionDaemon(Daemon):
    name = "reflection"
    tick_every_s = 2.0

    def on_event(self, evt: Event, ctx: DaemonContext) -> None:
        if evt.type != "episode_added":
            return
        if evt.payload.get("role") != "assistant":
            return
        # Lightweight reflection: store last interaction as a note
        eps = ctx.memory.recent_episodes(ctx.session_id, limit=6)
        last_user = next((e for e in eps if e["role"] == "user"), None)
        last_bot = next((e for e in eps if e["role"] == "assistant"), None)
        if not last_user or not last_bot:
            return
        title = "Conversation fragment"
        content = f"User: {short(last_user['content'], 900)}\nAssistant: {short(last_bot['content'], 900)}"
        nid = ctx.memory.add_note(
            title,
            content,
            tags=["conversation"],
            source_ids=[last_user["id"], last_bot["id"]],
            confidence=0.6,
        )
        ctx.bus.publish("note_added", {"note_id": nid})


class PruningDaemon(Daemon):
    """
    Summarize old episodes into an extractive Note and delete the raw episodes.

    This keeps the episodes table bounded while preserving a searchable digest
    in Notes (which already participate in hybrid retrieval).
    """

    name = "pruning"
    tick_every_s = 30.0

    def __init__(
        self,
        *,
        keep_last: int = 200,
        batch: int = 50,
        digest_chars: int = 280,
        confidence: float = 0.55,
    ):
        self.keep_last = max(0, int(keep_last))
        self.batch = max(1, int(batch))
        self.digest_chars = max(40, int(digest_chars))
        self.confidence = float(confidence)

    def tick(self, ctx: DaemonContext) -> None:
        total = int(ctx.memory.count_episodes(ctx.session_id))
        if total <= self.keep_last:
            return

        n = min(self.batch, total - self.keep_last)
        eps = ctx.memory.oldest_episodes(ctx.session_id, limit=n)
        if not eps:
            return

        start_iso = dt.datetime.fromtimestamp(float(eps[0]["ts"])).isoformat(timespec="seconds")
        end_iso = dt.datetime.fromtimestamp(float(eps[-1]["ts"])).isoformat(timespec="seconds")

        title = f"Episode digest ({ctx.session_id}) {end_iso}"
        lines: List[str] = [
            f"session_id: {ctx.session_id}",
            f"created: {dt.datetime.now().isoformat(timespec='seconds')}",
            f"episodes: {len(eps)} (keeping last {self.keep_last})",
            f"range: {start_iso} .. {end_iso}",
            "",
            "Extractive digest (verbatim snippets):",
        ]
        for e in eps:
            lines.append(f"- {e['id']} {e['role']}: {short(e['content'], self.digest_chars)}")

        nid = ctx.memory.add_note(
            title,
            "\n".join(lines),
            tags=["episode_digest"],
            source_ids=[e["id"] for e in eps],
            confidence=self.confidence,
        )
        ctx.bus.publish("note_added", {"note_id": nid})

        deleted = ctx.memory.delete_episodes([e["id"] for e in eps])
        evid = ctx.memory.add_evidence(
            "episodes_pruned",
            jdump({"note_id": nid, "deleted": deleted, "session_id": ctx.session_id, "range": [start_iso, end_iso]}),
            metadata={"note_id": nid, "session_id": ctx.session_id},
            dedupe=False,
        )
        ctx.bus.publish("episodes_pruned", {"note_id": nid, "deleted": deleted, "evidence_id": evid})

        ctx.initiative.submit(
            ProactiveCandidate(
                message=f"Pruned {deleted} old episode(s) into note {nid}.",
                evidence_ids=[evid],
                expected_utility=0.6,
                confidence=0.7,
                actionability=0.5,
                interruption_cost=0.25,
                risk=0.15,
            )
        )


class ConnectionMiner(Daemon):
    name = "connection_miner"
    tick_every_s = 5.0

    def __init__(self, min_score: float = 0.055):
        # RRF score values are small; threshold accordingly.
        self.min_score = float(min_score)

    def on_event(self, evt: Event, ctx: DaemonContext) -> None:
        if evt.type != "note_added":
            return
        nid = evt.payload.get("note_id")
        if not nid:
            return
        n = ctx.memory.get_note(nid)
        if not n:
            return
        hits = ctx.memory.search_notes(n["title"] + "\n" + n["content"], k=6)
        for h in hits:
            hid = h.get("id")
            if not hid or hid == nid:
                continue
            score = float(h.get("score", 0.0))
            if score >= self.min_score:
                ctx.memory.link_notes(nid, hid, "related", score)
                ctx.initiative.submit(
                    ProactiveCandidate(
                        message=f"New connection: note {nid} looks related to {hid} (rrf≈{score:.3f}).",
                        expected_utility=0.55,
                        confidence=0.6,
                        actionability=0.35,
                        interruption_cost=0.25,
                        risk=0.1,
                    )
                )


class ConsistencyAuditor(Daemon):
    name = "consistency_auditor"
    tick_every_s = 20.0

    def tick(self, ctx: DaemonContext) -> None:
        notes = ctx.memory.list_notes(limit=200)
        by_title: Dict[str, List[Dict[str, Any]]] = {}
        for n in notes:
            by_title.setdefault(n["title"].strip().lower(), []).append(n)
        for t, group in by_title.items():
            if len(group) < 2:
                continue
            # Heuristic: if multiple notes share a title, ensure they aren't wildly dissimilar.
            base = ctx.memory.get_note(group[0]["id"])
            if not base:
                continue
            base_toks = set(toks(base["content"]))
            for g in group[1:]:
                other = ctx.memory.get_note(g["id"])
                if not other:
                    continue
                ot = set(toks(other["content"]))
                j = len(base_toks & ot) / (len(base_toks | ot) or 1)
                if j < 0.15:
                    ctx.initiative.submit(
                        ProactiveCandidate(
                            message=f"Potential inconsistency: notes titled '{t}' differ a lot (overlap≈{j:.2f}).",
                            expected_utility=0.6,
                            confidence=0.45,
                            actionability=0.25,
                            interruption_cost=0.35,
                            risk=0.25,
                        )
                    )
                    return


class TaskSolverDaemon(Daemon):
    """
    Background worker that tries to solve queued tasks using memory/tools.
    In production you'd tailor this to your domain and add more tools.
    """

    name = "task_solver"
    tick_every_s = 3.0

    def __init__(self, max_steps: int = 4):
        self.max_steps = int(max_steps)

    def tick(self, ctx: DaemonContext) -> None:
        task = ctx.memory.fetch_runnable_task()
        if not task:
            return
        tid = task["id"]
        title = task["title"]
        desc = task["description"]

        # Attempt 1: retrieve related memories
        out = ctx.tools.execute(ToolCall(name="memory_search", arguments={"query": f"{title}\n{desc}", "k": 6}))
        evidence_ids: List[str] = []
        if out.ok and out.evidence_id:
            evidence_ids.append(out.evidence_id)

        if out.ok:
            notes = out.output.get("notes") or []
            if notes:
                # Mark done with summary of retrieved notes
                result = {"summary": "Found related notes", "notes": notes[:3]}
                ctx.memory.complete_task(tid, status="done", result=result, evidence_ids=evidence_ids)
                ctx.initiative.submit(
                    ProactiveCandidate(
                        message=f"Task '{title}' resolved via memory: found {len(notes)} related note(s).",
                        evidence_ids=evidence_ids,
                        expected_utility=0.7,
                        confidence=0.7,
                        actionability=0.6,
                        interruption_cost=0.2,
                        risk=0.1,
                    )
                )
                return

        # If no useful info, block and retry later
        ctx.memory.complete_task(
            tid,
            status="blocked",
            result={"summary": "Insufficient memory. Needs more info/tools."},
            evidence_ids=evidence_ids,
            next_run_ts=utc_ts() + 120.0,
        )


class BackgroundRunner:
    def __init__(self, bus: EventBus, ctx: DaemonContext, daemons: List[Daemon]):
        self.bus = bus
        self.ctx = ctx
        self.daemons = daemons
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, name="cogos-bg", daemon=True)
        self._last_tick: Dict[str, float] = {d.name: 0.0 for d in daemons}

    def start(self) -> None:
        self._th.start()

    def stop(self) -> None:
        self._stop.set()
        self._th.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            evt = self.bus.get(timeout=0.2)
            if evt:
                for d in self.daemons:
                    try:
                        d.on_event(evt, self.ctx)
                    except Exception as e:
                        log.warning("daemon.on_event failed", extra={"extra": {"daemon": d.name, "err": str(e)}})
            now = utc_ts()
            for d in self.daemons:
                if now - self._last_tick.get(d.name, 0.0) >= d.tick_every_s:
                    self._last_tick[d.name] = now
                    try:
                        d.tick(self.ctx)
                    except Exception as e:
                        log.warning("daemon.tick failed", extra={"extra": {"daemon": d.name, "err": str(e)}})

