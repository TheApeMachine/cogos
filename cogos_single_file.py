"""
CogOS — Production-Grade Baseline (single-file reference implementation; archived)

This is a *systems* architecture for "LLM-as-compiler" agents:

- Natural language UI (optional local LLM backend; stub fallback)
- Typed tool bus (schema-validated tool calls + provenance/evidence objects)
- Memory OS (SQLite; episodic + evidence + semantic notes + skills + tasks)
- Hybrid retrieval (SQLite FTS5 lexical search + vector similarity + RRF fusion)
- Proof-carrying answers (claims MUST cite evidence IDs and support spans)
- Verifier gate (reject unsupported claims; abstain)
- Continuous learning via memory + background daemons ("deep reasoning" keeps going)
- Initiative manager with impulse threshold + rate limiting

Dependencies:
    pip install "pydantic>=1.10" numpy

Optional (better embeddings):
    pip install sentence-transformers

Optional (local LLM):
    pip install llama-cpp-python

Run:
    python cogos_prod.py chat --db cogos.db

Notes:
- Side-effectful tools are disabled by default. Use --allow-side-effects to enable.
- This file is intentionally a single-file "reference build" you can split into modules.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import datetime as dt
import hashlib
import json
import logging
import math
import os
import queue
import re
import signal
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

# ---- Optional numpy (recommended) ----
try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore


# =============================================================================
# Pydantic compatibility (v1 and v2)
# =============================================================================
try:
    from pydantic import BaseModel, Field, ValidationError
except Exception as e:  # pragma: no cover
    raise RuntimeError("pydantic is required. Install with: pip install pydantic") from e


def _model_dump(m: BaseModel) -> Dict[str, Any]:
    if hasattr(m, "model_dump"):
        return m.model_dump()  # type: ignore[attr-defined]
    return m.dict()  # type: ignore[no-any-return]


def _model_json_schema(model: type[BaseModel]) -> Dict[str, Any]:
    if hasattr(model, "model_json_schema"):
        return model.model_json_schema()  # type: ignore[attr-defined]
    return model.schema()  # type: ignore[no-any-return]


# =============================================================================
# Logging
# =============================================================================

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": dt.datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            payload.update(record.extra)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "INFO", json_logs: bool = False) -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler()
    handler.setLevel(root.level)
    handler.setFormatter(JsonFormatter() if json_logs else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.handlers[:] = [handler]


log = logging.getLogger("cogos")


# =============================================================================
# Small utilities
# =============================================================================

def utc_ts() -> float:
    return time.time()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def jdump(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, sort_keys=True)


def jload(s: Optional[str]) -> Any:
    if not s:
        return None
    return json.loads(s)


def short(s: str, n: int = 200) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else (s[: n - 1] + "…")


def toks(s: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", (s or "").lower())


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# =============================================================================
# Embeddings
# =============================================================================

class EmbeddingModel:
    dim: int
    name: str = "base"

    def embed(self, text: str) -> "np.ndarray":
        raise NotImplementedError


class HashEmbed(EmbeddingModel):
    """Fast, dependency-free, decent baseline embedding via feature hashing."""
    name = "hash"

    def __init__(self, dim: int = 384):
        if np is None:
            raise RuntimeError("numpy is required for embeddings. pip install numpy")
        self.dim = int(dim)

    def _h64(self, token: str) -> int:
        h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(h, "little", signed=False)

    def embed(self, text: str) -> "np.ndarray":
        v = np.zeros((self.dim,), dtype=np.float32)
        ts = toks(text)
        if not ts:
            return v
        for t in ts:
            hv = self._h64(t)
            idx = hv % self.dim
            sign = 1.0 if ((hv >> 63) & 1) == 0 else -1.0
            v[idx] += sign
        n = float(np.linalg.norm(v))
        if n > 0:
            v /= n
        return v


class SentenceTransformerEmbed(EmbeddingModel):
    name = "sentence_transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if np is None:
            raise RuntimeError("numpy is required for embeddings. pip install numpy")
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:
            raise ImportError("sentence-transformers not installed. pip install sentence-transformers") from e
        self._st = SentenceTransformer(model_name)
        self.dim = int(self._st.get_sentence_embedding_dimension())

    def embed(self, text: str) -> "np.ndarray":
        v = self._st.encode([text], normalize_embeddings=True)[0]
        return np.asarray(v, dtype=np.float32)


def cosine(a: "np.ndarray", b: "np.ndarray") -> float:
    if np is None:
        return 0.0
    da = float(np.linalg.norm(a))
    db = float(np.linalg.norm(b))
    if da == 0.0 or db == 0.0:
        return 0.0
    return float(np.dot(a, b) / (da * db))


# =============================================================================
# Event bus
# =============================================================================

@dataclass(frozen=True)
class Event:
    type: str
    ts: float
    payload: Dict[str, Any]
    id: str = dataclasses.field(default_factory=lambda: new_id("evt"))


class EventBus:
    """Thread-safe pub/sub queue."""
    def __init__(self):
        self._q: "queue.Queue[Event]" = queue.Queue()

    def publish(self, type: str, payload: Dict[str, Any]) -> Event:
        evt = Event(type=type, ts=utc_ts(), payload=payload)
        self._q.put(evt)
        return evt

    def get(self, timeout: Optional[float] = None) -> Optional[Event]:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None


# =============================================================================
# Memory Store (SQLite + FTS5)
# =============================================================================

class MemoryStore:
    """
    Production-oriented SQLite store:
    - WAL mode, safe transaction boundaries
    - FTS5 tables for lexical search
    - Embeddings stored as BLOB for vector search
    - Task table for background work
    """

    def __init__(self, db_path: str, embedder: EmbeddingModel):
        if np is None:
            raise RuntimeError("numpy is required. pip install numpy")
        self.db_path = db_path
        self.embedder = embedder
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._fts_ok = False
        self._init_schema()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---- schema ----

    def _init_schema(self) -> None:
        with self._lock:
            c = self._conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS episodes(
                    id TEXT PRIMARY KEY,
                    ts REAL NOT NULL,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT
                );
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS evidence(
                    id TEXT PRIMARY KEY,
                    ts REAL NOT NULL,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    sha256 TEXT,
                    emb BLOB,
                    emb_dim INTEGER
                );
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS notes(
                    id TEXT PRIMARY KEY,
                    created REAL NOT NULL,
                    updated REAL NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tags TEXT,
                    links TEXT,
                    source_ids TEXT,
                    confidence REAL NOT NULL,
                    emb BLOB,
                    emb_dim INTEGER
                );
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS skills(
                    id TEXT PRIMARY KEY,
                    created REAL NOT NULL,
                    updated REAL NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    preconditions TEXT,
                    steps TEXT,
                    tests TEXT,
                    emb BLOB,
                    emb_dim INTEGER
                );
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS tasks(
                    id TEXT PRIMARY KEY,
                    created REAL NOT NULL,
                    updated REAL NOT NULL,
                    status TEXT NOT NULL,             -- queued|running|blocked|done|failed
                    priority INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    payload TEXT,
                    result TEXT,
                    evidence_ids TEXT,
                    error TEXT,
                    attempts INTEGER NOT NULL,
                    next_run_ts REAL
                );
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS proactive(
                    id TEXT PRIMARY KEY,
                    ts REAL NOT NULL,
                    score REAL NOT NULL,
                    message TEXT NOT NULL,
                    evidence_ids TEXT,
                    delivered INTEGER NOT NULL DEFAULT 0
                );
            """)

            # FTS5 (lexical)
            self._fts_ok = self._try_init_fts(c)
            self._conn.commit()

    def _try_init_fts(self, c: sqlite3.Cursor) -> bool:
        try:
            c.execute("CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(id UNINDEXED, title, content);")
            c.execute("CREATE VIRTUAL TABLE IF NOT EXISTS evidence_fts USING fts5(id UNINDEXED, kind, content);")
            c.execute("CREATE VIRTUAL TABLE IF NOT EXISTS skills_fts USING fts5(id UNINDEXED, name, description);")
            return True
        except sqlite3.OperationalError:
            return False

    # ---- embeddings storage ----

    def _to_blob(self, v: "np.ndarray") -> Tuple[bytes, int]:
        v = np.asarray(v, dtype=np.float32)
        return v.tobytes(), int(v.shape[0])

    def _from_blob(self, b: Optional[bytes], dim: Optional[int]) -> "np.ndarray":
        target_dim = int(dim) if dim is not None else int(self.embedder.dim)
        if b is None or target_dim <= 0:
            fallback_dim = target_dim if target_dim > 0 else int(self.embedder.dim)
            return np.zeros((fallback_dim,), dtype=np.float32)
        v = np.frombuffer(b, dtype=np.float32)
        if v.size == target_dim:
            return v
        if v.size > target_dim:
            return v[:target_dim]
        out = np.zeros((target_dim,), dtype=np.float32)
        out[: v.size] = v
        return out

    # ---- episode ----

    def add_episode(self, session_id: str, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        eid = new_id("ep")
        with self._lock:
            self._conn.execute(
                "INSERT INTO episodes(id, ts, session_id, role, content, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                (eid, utc_ts(), session_id, role, content, jdump(metadata or {})),
            )
            self._conn.commit()
        return eid

    def recent_episodes(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, ts, role, content, metadata FROM episodes WHERE session_id=? ORDER BY ts DESC LIMIT ?",
                (session_id, int(limit)),
            ).fetchall()
        out = []
        for r in rows:
            out.append({"id": r["id"], "ts": r["ts"], "role": r["role"], "content": r["content"], "metadata": jload(r["metadata"]) or {}})
        return out

    def count_episodes(self, session_id: str) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(1) AS n FROM episodes WHERE session_id=?", (session_id,)).fetchone()
        return int(row["n"]) if row else 0

    def oldest_episodes(self, session_id: str, limit: int) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, ts, role, content, metadata FROM episodes WHERE session_id=? ORDER BY ts ASC LIMIT ?",
                (session_id, int(limit)),
            ).fetchall()
        out = []
        for r in rows:
            out.append({"id": r["id"], "ts": r["ts"], "role": r["role"], "content": r["content"], "metadata": jload(r["metadata"]) or {}})
        return out

    def delete_episodes(self, episode_ids: Sequence[str]) -> int:
        ids = [str(i) for i in episode_ids if str(i).strip()]
        if not ids:
            return 0
        with self._lock:
            q = ",".join(["?"] * len(ids))
            cur = self._conn.execute(f"DELETE FROM episodes WHERE id IN ({q})", ids)
            self._conn.commit()
        if cur.rowcount is None or cur.rowcount < 0:
            return len(ids)
        return int(cur.rowcount)

    # ---- evidence ----

    def add_evidence(self, kind: str, content: str, metadata: Optional[Dict[str, Any]] = None, *, dedupe: bool = True) -> str:
        content = content if isinstance(content, str) else str(content)
        sha = hashlib.sha256(content.encode("utf-8")).hexdigest()

        if dedupe:
            with self._lock:
                row = self._conn.execute("SELECT id FROM evidence WHERE sha256=? AND kind=?", (sha, kind)).fetchone()
                if row:
                    return str(row["id"])

        evid = new_id("ev")
        emb = self.embedder.embed(content)
        blob, dim = self._to_blob(emb)
        with self._lock:
            self._conn.execute(
                "INSERT INTO evidence(id, ts, kind, content, metadata, sha256, emb, emb_dim) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (evid, utc_ts(), kind, content, jdump(metadata or {}), sha, blob, dim),
            )
            if self._fts_ok:
                self._fts_upsert("evidence_fts", evid, {"kind": kind, "content": content})
            self._conn.commit()
        return evid

    def get_evidence(self, evid: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, ts, kind, content, metadata FROM evidence WHERE id=?",
                (evid,),
            ).fetchone()
        if not row:
            return None
        return {"id": row["id"], "ts": row["ts"], "kind": row["kind"], "content": row["content"], "metadata": jload(row["metadata"]) or {}}

    # ---- notes ----

    def add_note(
        self,
        title: str,
        content: str,
        *,
        tags: Optional[List[str]] = None,
        source_ids: Optional[List[str]] = None,
        confidence: float = 0.7,
        links: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        nid = new_id("note")
        now = utc_ts()
        emb = self.embedder.embed(title + "\n" + content)
        blob, dim = self._to_blob(emb)
        with self._lock:
            self._conn.execute(
                "INSERT INTO notes(id, created, updated, title, content, tags, links, source_ids, confidence, emb, emb_dim) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (nid, now, now, title, content, jdump(tags or []), jdump(links or []), jdump(source_ids or []), float(confidence), blob, dim),
            )
            if self._fts_ok:
                self._fts_upsert("notes_fts", nid, {"title": title, "content": content})
            self._conn.commit()
        return nid

    def get_note(self, nid: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM notes WHERE id=?", (nid,)).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "created": row["created"],
            "updated": row["updated"],
            "title": row["title"],
            "content": row["content"],
            "tags": jload(row["tags"]) or [],
            "links": jload(row["links"]) or [],
            "source_ids": jload(row["source_ids"]) or [],
            "confidence": float(row["confidence"]),
        }

    def update_note(self, nid: str, *, title: Optional[str] = None, content: Optional[str] = None, tags: Optional[List[str]] = None, links: Optional[List[Dict[str, Any]]] = None, confidence: Optional[float] = None) -> bool:
        cur = self.get_note(nid)
        if not cur:
            return False
        new_title = title if title is not None else cur["title"]
        new_content = content if content is not None else cur["content"]

        fields: Dict[str, Any] = {"updated": utc_ts()}
        if title is not None:
            fields["title"] = new_title
        if content is not None:
            fields["content"] = new_content
        if tags is not None:
            fields["tags"] = jdump(tags)
        if links is not None:
            fields["links"] = jdump(links)
        if confidence is not None:
            fields["confidence"] = float(confidence)

        if title is not None or content is not None:
            emb = self.embedder.embed(new_title + "\n" + new_content)
            blob, dim = self._to_blob(emb)
            fields["emb"] = blob
            fields["emb_dim"] = dim

        set_clause = ", ".join([f"{k}=?" for k in fields.keys()])
        vals = list(fields.values()) + [nid]

        with self._lock:
            self._conn.execute(f"UPDATE notes SET {set_clause} WHERE id=?", vals)
            if self._fts_ok and (title is not None or content is not None):
                self._fts_upsert("notes_fts", nid, {"title": new_title, "content": new_content})
            self._conn.commit()
        return True

    def list_notes(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT id, updated, title, tags, links, confidence FROM notes ORDER BY updated DESC LIMIT ?", (int(limit),)).fetchall()
        out = []
        for r in rows:
            out.append({
                "id": r["id"],
                "updated": r["updated"],
                "title": r["title"],
                "tags": jload(r["tags"]) or [],
                "links": jload(r["links"]) or [],
                "confidence": float(r["confidence"]),
            })
        return out

    def link_notes(self, a: str, b: str, relation: str, score: float) -> None:
        na = self.get_note(a)
        nb = self.get_note(b)
        if not na or not nb:
            return

        def add_link(note: Dict[str, Any], to: str) -> List[Dict[str, Any]]:
            links = list(note.get("links") or [])
            if any(l.get("to") == to and l.get("relation") == relation for l in links):
                return links
            links.append({"to": to, "relation": relation, "score": float(score), "ts": utc_ts()})
            return links

        self.update_note(a, links=add_link(na, b))
        self.update_note(b, links=add_link(nb, a))

    # ---- skills ----

    def add_skill(
        self,
        name: str,
        description: str,
        *,
        preconditions: Optional[Dict[str, Any]] = None,
        steps: Optional[List[Dict[str, Any]]] = None,
        tests: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        sid = new_id("skill")
        now = utc_ts()
        emb = self.embedder.embed(name + "\n" + description)
        blob, dim = self._to_blob(emb)
        with self._lock:
            self._conn.execute(
                "INSERT INTO skills(id, created, updated, name, description, preconditions, steps, tests, emb, emb_dim) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (sid, now, now, name, description, jdump(preconditions or {}), jdump(steps or []), jdump(tests or []), blob, dim),
            )
            if self._fts_ok:
                self._fts_upsert("skills_fts", sid, {"name": name, "description": description})
            self._conn.commit()
        return sid

    def list_skills(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT id, updated, name, description FROM skills ORDER BY updated DESC LIMIT ?", (int(limit),)).fetchall()
        return [{"id": r["id"], "updated": r["updated"], "name": r["name"], "description": short(r["description"], 160)} for r in rows]

    # ---- tasks ----

    def add_task(self, title: str, description: str, *, priority: int = 0, payload: Optional[Dict[str, Any]] = None, next_run_ts: Optional[float] = None) -> str:
        tid = new_id("task")
        now = utc_ts()
        with self._lock:
            self._conn.execute(
                "INSERT INTO tasks(id, created, updated, status, priority, title, description, payload, result, evidence_ids, error, attempts, next_run_ts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (tid, now, now, "queued", int(priority), title, description, jdump(payload or {}), jdump({}), jdump([]), "", 0, next_run_ts),
            )
            self._conn.commit()
        return tid

    def list_tasks(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, updated, status, priority, title, description, attempts, next_run_ts FROM tasks ORDER BY priority DESC, updated DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        out = []
        for r in rows:
            out.append({
                "id": r["id"],
                "updated": r["updated"],
                "status": r["status"],
                "priority": r["priority"],
                "title": r["title"],
                "description": short(r["description"], 220),
                "attempts": r["attempts"],
                "next_run_ts": r["next_run_ts"],
            })
        return out

    def fetch_runnable_task(self, now_ts: Optional[float] = None) -> Optional[Dict[str, Any]]:
        now_ts = utc_ts() if now_ts is None else now_ts
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM tasks WHERE status IN ('queued','blocked') AND (next_run_ts IS NULL OR next_run_ts <= ?) "
                "ORDER BY priority DESC, updated ASC LIMIT 1",
                (now_ts,),
            ).fetchone()
            if not row:
                return None
            # mark running
            self._conn.execute("UPDATE tasks SET status='running', updated=? WHERE id=?", (utc_ts(), row["id"]))
            self._conn.commit()
        return dict(row)

    def complete_task(self, tid: str, *, status: str, result: Optional[Dict[str, Any]] = None, evidence_ids: Optional[List[str]] = None, error: str = "", next_run_ts: Optional[float] = None) -> None:
        now = utc_ts()
        with self._lock:
            self._conn.execute(
                "UPDATE tasks SET updated=?, status=?, result=?, evidence_ids=?, error=?, attempts=attempts+1, next_run_ts=? WHERE id=?",
                (now, status, jdump(result or {}), jdump(evidence_ids or []), error, next_run_ts, tid),
            )
            self._conn.commit()

    # ---- proactive ----

    def add_proactive(self, message: str, *, score: float, evidence_ids: Optional[List[str]] = None) -> str:
        pid = new_id("pro")
        with self._lock:
            self._conn.execute(
                "INSERT INTO proactive(id, ts, score, message, evidence_ids, delivered) VALUES (?, ?, ?, ?, ?, 0)",
                (pid, utc_ts(), float(score), message, jdump(evidence_ids or [])),
            )
            self._conn.commit()
        return pid

    def fetch_undelivered_proactive(self, limit: int = 3) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, ts, score, message, evidence_ids FROM proactive WHERE delivered=0 ORDER BY score DESC, ts ASC LIMIT ?",
                (int(limit),),
            ).fetchall()
            ids = [r["id"] for r in rows]
            if ids:
                self._conn.executemany("UPDATE proactive SET delivered=1 WHERE id=?", [(i,) for i in ids])
                self._conn.commit()
        return [{"id": r["id"], "ts": r["ts"], "score": r["score"], "message": r["message"], "evidence_ids": jload(r["evidence_ids"]) or []} for r in rows]

    # ---- FTS maintenance ----

    def _fts_upsert(self, table: str, doc_id: str, cols: Dict[str, str]) -> None:
        if not self._fts_ok:
            return
        # delete then insert (simple, robust)
        self._conn.execute(f"DELETE FROM {table} WHERE id=?", (doc_id,))
        fields = ["id"] + list(cols.keys())
        placeholders = ",".join(["?"] * len(fields))
        values = [doc_id] + [cols[k] for k in cols.keys()]
        self._conn.execute(f"INSERT INTO {table}({','.join(fields)}) VALUES ({placeholders})", values)

    # ---- hybrid retrieval ----

    def _fts_search(self, fts_table: str, query: str, k: int) -> List[str]:
        if not self._fts_ok:
            return []
        ts = toks(query)
        if not ts:
            return []
        # Construct a "safe" FTS query from tokens to avoid MATCH syntax errors
        # (e.g. "User:" is parsed as a column selector).
        q = " OR ".join(ts[:64])
        with self._lock:
            # bm25 smaller is better; we only use ranking order.
            rows = self._conn.execute(
                f"SELECT id, bm25({fts_table}) AS r FROM {fts_table} WHERE {fts_table} MATCH ? ORDER BY r LIMIT ?",
                (q, int(k)),
            ).fetchall()
        return [str(r["id"]) for r in rows]

    def _vector_candidates(self, table: str, query_vec: "np.ndarray", k: int) -> List[Tuple[str, float]]:
        if np is None:
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        q_dim = int(q.shape[0])
        if q_dim <= 0:
            return []

        kk = int(k)
        if kk <= 0:
            return []

        with self._lock:
            rows = self._conn.execute(
                f"SELECT id, emb FROM {table} WHERE emb IS NOT NULL AND emb_dim=?",
                (q_dim,),
            ).fetchall()
        if not rows:
            return []

        ids: List[str] = []
        embs = np.empty((len(rows), q_dim), dtype=np.float32)
        for i, r in enumerate(rows):
            ids.append(str(r["id"]))
            try:
                embs[i, :] = np.frombuffer(r["emb"], dtype=np.float32, count=q_dim)
            except Exception:
                embs[i, :] = 0.0

        qn = float(np.linalg.norm(q))
        if qn == 0.0:
            return []
        en = np.linalg.norm(embs, axis=1) * qn
        en = np.where(en == 0.0, 1e-12, en)
        scores = (embs @ q) / en

        if scores.shape[0] <= kk:
            idx = np.argsort(-scores)
        else:
            idx_part = np.argpartition(-scores, kk - 1)[:kk]
            idx = idx_part[np.argsort(-scores[idx_part])]

        return [(ids[i], float(scores[i])) for i in idx]

    @staticmethod
    def _rrf_fuse(rankings: List[List[str]], *, k: int = 60) -> List[Tuple[str, float]]:
        scores: Dict[str, float] = {}
        for ranking in rankings:
            for rank, doc_id in enumerate(ranking, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def search_notes(self, query: str, k: int = 5, *, fts_k: int = 30, vec_k: int = 30) -> List[Dict[str, Any]]:
        qv = self.embedder.embed(query)
        lex_ids = self._fts_search("notes_fts", query, fts_k)
        vec = self._vector_candidates("notes", qv, vec_k)
        vec_ids = [i for i, _ in vec]
        fused = self._rrf_fuse([lex_ids, vec_ids])[: int(k)]

        out = []
        for doc_id, score in fused:
            n = self.get_note(doc_id)
            if not n:
                continue
            out.append({
                "id": n["id"],
                "title": n["title"],
                "content_snip": short(n["content"], 800),
                "tags": n["tags"],
                "links": n["links"],
                "confidence": n["confidence"],
                "score": float(score),
            })
        return out

    def search_evidence(self, query: str, k: int = 5, *, fts_k: int = 30, vec_k: int = 30) -> List[Dict[str, Any]]:
        qv = self.embedder.embed(query)
        lex_ids = self._fts_search("evidence_fts", query, fts_k)
        vec = self._vector_candidates("evidence", qv, vec_k)
        vec_ids = [i for i, _ in vec]
        fused = self._rrf_fuse([lex_ids, vec_ids])[: int(k)]

        out = []
        for doc_id, score in fused:
            ev = self.get_evidence(doc_id)
            if not ev:
                continue
            out.append({
                "id": ev["id"],
                "kind": ev["kind"],
                "content_snip": short(ev["content"], 800),
                "score": float(score),
            })
        return out

    def search_skills(self, query: str, k: int = 5, *, fts_k: int = 30, vec_k: int = 30) -> List[Dict[str, Any]]:
        qv = self.embedder.embed(query)
        lex_ids = self._fts_search("skills_fts", query, fts_k)
        vec = self._vector_candidates("skills", qv, vec_k)
        vec_ids = [i for i, _ in vec]
        fused = self._rrf_fuse([lex_ids, vec_ids])[: int(k)]

        out = []
        for doc_id, score in fused:
            with self._lock:
                row = self._conn.execute("SELECT id, name, description FROM skills WHERE id=?", (doc_id,)).fetchone()
            if not row:
                continue
            out.append({
                "id": row["id"],
                "name": row["name"],
                "description": short(row["description"], 800),
                "score": float(score),
            })
        return out


# =============================================================================
# Tools (typed)
# =============================================================================

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
            out.append({
                "name": spec.name,
                "description": spec.description,
                "side_effects": spec.side_effects,
                "input_schema": _model_json_schema(spec.input_model),
                "output_schema": _model_json_schema(spec.output_model),
            })
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

        evid_id = self.memory.add_evidence(kind=f"tool:{call.name}", content=jdump(_model_dump(out)), metadata={"tool": call.name, "args": call.arguments})
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
            if isinstance(node.op, ast.Add): return a + b
            if isinstance(node.op, ast.Sub): return a - b
            if isinstance(node.op, ast.Mult): return a * b
            if isinstance(node.op, ast.Div): return a / b
            if isinstance(node.op, ast.FloorDiv): return a // b
            if isinstance(node.op, ast.Mod): return a % b
            if isinstance(node.op, ast.Pow): return a ** b
            raise ValueError("Operator not allowed.")
        if isinstance(node, ast.UnaryOp):
            v = self.visit(node.operand)
            if isinstance(node.op, ast.UAdd): return +v
            if isinstance(node.op, ast.USub): return -v
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


# =============================================================================
# LLM interface (pluggable)
# =============================================================================

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatModel:
    def generate_text(self, messages: List[ChatMessage], *, temperature: float = 0.2, max_tokens: int = 800) -> str:
        raise NotImplementedError

    def generate_json(self, messages: List[ChatMessage], schema: type[BaseModel], *, temperature: float = 0.2, max_tokens: int = 1200) -> BaseModel:
        txt = self.generate_text(messages, temperature=temperature, max_tokens=max_tokens)
        data = _extract_first_json_object(txt)
        return schema(**data)


class StubChatModel(ChatModel):
    def generate_text(self, messages: List[ChatMessage], *, temperature: float = 0.0, max_tokens: int = 256) -> str:
        # Always abstain; forces grounded/tool-only behavior.
        return json.dumps({"claims": [], "draft": "I don't know.", "proactive": []})


class LlamaCppChatModel(ChatModel):
    def __init__(self, model_path: str, *, n_ctx: int = 4096, n_threads: Optional[int] = None, n_gpu_layers: int = 0):
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as e:
            raise ImportError("llama-cpp-python not installed. pip install llama-cpp-python") from e
        self._llm = Llama(model_path=model_path, n_ctx=int(n_ctx), n_threads=n_threads, n_gpu_layers=int(n_gpu_layers))

    def _fmt(self, messages: List[ChatMessage]) -> str:
        # Generic formatting; for best results use a model-specific chat template.
        parts: List[str] = []
        for m in messages:
            parts.append(f"{m.role.upper()}: {m.content}")
        parts.append("ASSISTANT:")
        return "\n".join(parts)

    def generate_text(self, messages: List[ChatMessage], *, temperature: float = 0.2, max_tokens: int = 800) -> str:
        prompt = self._fmt(messages)
        out = self._llm(prompt, max_tokens=int(max_tokens), temperature=float(temperature), stop=["USER:", "SYSTEM:"])
        return str(out["choices"][0]["text"]).strip()


def _extract_first_json_object(text: str) -> Dict[str, Any]:
    """
    Extract the first JSON object from arbitrary text by scanning balanced braces.

    This is intentionally conservative. If it can't parse safely, it raises ValueError.
    """
    s = text.strip()
    # Fast path: direct parse
    try:
        if s.startswith("{") and s.endswith("}"):
            return json.loads(s)
    except Exception:
        pass

    start = s.find("{")
    if start < 0:
        raise ValueError("No JSON object found.")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = s[start : i + 1]
                    return json.loads(chunk)
    raise ValueError("Unbalanced JSON braces.")


# =============================================================================
# Typed IR (Plan + Claims)
# =============================================================================

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


# =============================================================================
# Planner
# =============================================================================

class Planner:
    def plan(self, user_text: str, *, tools: ToolBus, memory: MemoryStore) -> Plan:
        raise NotImplementedError


class RulePlanner(Planner):
    _ARITH = re.compile(r"(?<!\w)(?:\d+\s*[\+\-\*/\^]\s*\d+|calculate|compute|eval|evaluate)(?!\w)", re.IGNORECASE)

    def plan(self, user_text: str, *, tools: ToolBus, memory: MemoryStore) -> Plan:
        steps: List[PlanStep] = [StepMemorySearch(query=user_text, k=6)]
        if self._ARITH.search(user_text):
            expr = user_text
            m = re.search(r"(?:calculate|compute|eval|evaluate)\s*[:\-]?\s*(.*)$", user_text, re.IGNORECASE)
            if m and m.group(1).strip():
                expr = m.group(1).strip()
            steps.append(StepToolCall(tool="calc", arguments={"expression": expr}))
        steps.append(StepRespond())
        return Plan(steps=steps)


class LLMPlanner(Planner):
    def __init__(self, model: ChatModel, *, fallback: Optional[Planner] = None):
        self.model = model
        self.fallback = fallback or RulePlanner()

    class _Schema(BaseModel):
        steps: List[Dict[str, Any]] = Field(default_factory=list)

    def plan(self, user_text: str, *, tools: ToolBus, memory: MemoryStore) -> Plan:
        compact_tools = []
        for t in tools.list_tools():
            props = (t["input_schema"].get("properties") or {})
            compact_tools.append({
                "name": t["name"],
                "description": t["description"],
                "side_effects": t["side_effects"],
                "args": list(props.keys()),
            })

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
        user = "User request:\n" + user_text + "\n\nAvailable tools:\n" + jdump(compact_tools) + "\n\nReturn {steps:[...]} only."
        msgs = [ChatMessage(role="system", content=sys), ChatMessage(role="user", content=user)]
        try:
            raw = self.model.generate_json(msgs, self._Schema, temperature=0.1, max_tokens=900)
            return Plan(steps=raw.steps)  # pydantic validates union
        except Exception as e:
            log.warning("LLMPlanner failed; falling back.", extra={"extra": {"err": str(e)}})
            return self.fallback.plan(user_text, tools=tools, memory=memory)


# =============================================================================
# Reasoner
# =============================================================================

class Reasoner:
    def propose(
        self,
        user_text: str,
        *,
        plan: Plan,
        evidence_map: Dict[str, str],
        memory_hits: Dict[str, Any],
        tool_outcomes: List["ToolOutcome"],
    ) -> ProposedAnswer:
        raise NotImplementedError


class ConservativeReasoner(Reasoner):
    """No LLM: produces only tool-derived and memory-search derived claims."""
    def propose(self, user_text: str, *, plan: Plan, evidence_map: Dict[str, str], memory_hits: Dict[str, Any], tool_outcomes: List["ToolOutcome"]) -> ProposedAnswer:
        claims: List[Claim] = []
        # calc results
        for out in tool_outcomes:
            if out.ok and out.evidence_id and "result" in out.output:
                claims.append(Claim(
                    text=f"Computed result: {out.output['result']}",
                    evidence_ids=[out.evidence_id],
                    support_spans=[str(out.output["result"])],
                    kind="math",
                ))

        # surface top memory hits as "related items" (not asserting facts)
        mem_evid = next((o for o in tool_outcomes if o.ok and o.tool == "memory_search" and o.evidence_id), None)
        if mem_evid and mem_evid.evidence_id:
            notes = memory_hits.get("notes") or []
            if notes:
                line = ", ".join([f"{n.get('title')}({n.get('id')})" for n in notes[:3] if n.get("id")])
                claims.append(Claim(
                    text=f"Related notes: {line}",
                    evidence_ids=[mem_evid.evidence_id],
                    support_spans=[notes[0].get("id", "")],
                    kind="fact",
                ))
        return ProposedAnswer(claims=claims, draft="Here is what I can ground from tools/memory.", proactive=[])


class LLMReasoner(Reasoner):
    def __init__(self, model: ChatModel):
        self.model = model

    class _Schema(BaseModel):
        claims: List[Dict[str, Any]] = Field(default_factory=list)
        draft: str = ""
        proactive: List[Dict[str, Any]] = Field(default_factory=list)

    def propose(self, user_text: str, *, plan: Plan, evidence_map: Dict[str, str], memory_hits: Dict[str, Any], tool_outcomes: List["ToolOutcome"]) -> ProposedAnswer:
        def _build_span_menu(evidence_text: str, user_text: str, *, max_spans: int = 14) -> List[str]:
            """
            Build a small menu of candidate support spans extracted from `evidence_text`.

            The model cites spans by index (support_span_ids), and we map indices back to
            exact substrings. This avoids brittle exact-copy behavior.
            """
            txt = str(evidence_text or "")
            if not txt:
                return []
            MAX_LEN = 120
            max_spans = max(1, int(max_spans))
            user_tokens = set(toks(user_text or ""))

            def score(seg: str) -> float:
                st = set(toks(seg))
                if not st or not user_tokens:
                    return 0.0
                return len(st & user_tokens) / len(st)

            def add(menu: List[str], seen: set, seg: str) -> None:
                if not seg:
                    return
                s = str(seg)
                if "\n" in s:
                    return
                s = s.strip()
                if not s:
                    return
                if len(s) > MAX_LEN:
                    s = s[:MAX_LEN]
                if s in seen:
                    return
                seen.add(s)
                menu.append(s)

            kv_pat = re.compile(
                r'"[^"\n]{1,60}"\s*:\s*(?:"[^"\n]{0,60}"|-?\d+(?:\.\d+)?|true|false|null)',
                re.IGNORECASE,
            )
            kv = [m.group(0) for m in kv_pat.finditer(txt)]
            quote_pat = re.compile(r'"[^"\n]{1,80}"')
            quotes = [m.group(0) for m in quote_pat.finditer(txt)]
            nums = re.findall(r"-?\d+(?:\.\d+)?", txt)
            lines = [ln for ln in txt.splitlines() if ln and ln.strip()]

            kv_sorted = sorted(kv, key=lambda s: (-score(s), len(s)))
            quotes_sorted = sorted(quotes, key=lambda s: (-score(s), len(s)))
            lines_sorted = sorted(lines, key=lambda s: (-score(s), len(s)))

            menu: List[str] = []
            seen: set = set()

            for ln in lines:
                add(menu, seen, ln)
                break
            for seg in kv_sorted[: max_spans * 2]:
                add(menu, seen, seg)
                if len(menu) >= max_spans:
                    return menu
            for n in nums[: max_spans * 2]:
                add(menu, seen, n)
                if len(menu) >= max_spans:
                    return menu
            for seg in quotes_sorted:
                add(menu, seen, seg)
                if len(menu) >= max_spans:
                    return menu
            for seg in lines_sorted:
                add(menu, seen, seg)
                if len(menu) >= max_spans:
                    return menu
            return menu[:max_spans]

        def _ev_excerpt(s: str, n: int) -> str:
            if not s:
                return ""
            return s if len(s) <= n else s[:n]

        blocks: List[str] = []
        span_menus: Dict[str, List[str]] = {}
        for eid, txt in list(evidence_map.items())[:8]:
            menu = _build_span_menu(txt, user_text, max_spans=14)
            span_menus[eid] = menu
            menu_lines = "\n".join([f"{i}: {json.dumps(sp, ensure_ascii=False)}" for i, sp in enumerate(menu)]) or "(empty)"
            blocks.append(f"[{eid}]\nEXCERPT:\n{_ev_excerpt(txt, 900)}\n\nSPAN_MENU (cite by index):\n{menu_lines}")

        sys = (
            "You are a reasoning compiler. Output JSON only.\n"
            "Goal: answer the user using atomic claims grounded in evidence.\n\n"
            "Rules (hard):\n"
            "- Each claim MUST cite exactly 1 evidence_id.\n"
            "- Each claim MUST include evidence_ids (existing IDs) AND support_span_ids.\n"
            "- support_span_ids MUST be 1 or 2 integers indexing into the cited evidence's SPAN_MENU.\n"
            "- Do NOT introduce facts not supported by evidence.\n"
            "- If evidence is insufficient, return claims=[] and draft='I don't know'.\n\n"
            "Examples (illustrative only; do not copy placeholder IDs):\n"
            "Evidence: [<EVID>] with SPAN_MENU indices.\n"
            "Good claim: {\"text\":\"10/4 equals 2.5\",\"evidence_ids\":[\"<EVID>\"],\"support_span_ids\":[0,1],\"kind\":\"math\"}\n\n"
            "Output JSON format:\n"
            "{claims:[{text,evidence_ids,support_span_ids,kind}], draft:str, proactive:list}\n"
        )
        user = (
            f"User question:\n{user_text}\n\n"
            "Evidence you may cite:\n" + "\n".join(blocks) + "\n\n"
            "Return JSON only."
        )
        msgs = [ChatMessage(role="system", content=sys), ChatMessage(role="user", content=user)]
        raw = self.model.generate_json(msgs, self._Schema, temperature=0.2, max_tokens=1200)

        claims: List[Claim] = []
        for rc in raw.claims:
            try:
                text = str(rc.get("text", "")).strip()
                eids = list(rc.get("evidence_ids") or [])
                span_ids = list(rc.get("support_span_ids") or [])
                kind = rc.get("kind") or "fact"
                if not (text and eids and span_ids):
                    continue
                evid = str(eids[0])
                menu = span_menus.get(evid) or []
                spans: List[str] = []
                ok = True
                for sid in span_ids:
                    try:
                        i = int(sid)
                    except Exception:
                        ok = False
                        break
                    if i < 0 or i >= len(menu):
                        ok = False
                        break
                    sp = menu[i]
                    if sp and (sp in (evidence_map.get(evid, "") or "")):
                        spans.append(sp)
                    else:
                        ok = False
                        break
                if ok and spans:
                    claims.append(Claim(text=text, evidence_ids=[evid], support_spans=spans, kind=kind))
            except Exception:
                continue

        return ProposedAnswer(claims=claims, draft=str(raw.draft or ""), proactive=list(raw.proactive or []))


class SearchReasoner(Reasoner):
    """
    Best-of-N reasoning: sample multiple candidate claim sets and choose the one
    that *appears most verifiable* against the provided evidence.

    This is a practical approximation of "explore multiple reasoning paths"
    (Tree-of-Thoughts style) without implementing a full tree search.
    """
    def __init__(self, base: LLMReasoner, samples: int = 4):
        self.base = base
        self.samples = int(samples)

    @staticmethod
    def _looks_supported(claim: Claim, evidence_map: Dict[str, str]) -> float:
        # Evidence existence
        ev_texts = [evidence_map.get(eid, "") for eid in claim.evidence_ids if eid in evidence_map]
        if not ev_texts:
            return 0.0

        joined = "\n".join(ev_texts)

        # Span hits (exact substring)
        spans = list(claim.support_spans or [])
        if not spans:
            return 0.0
        hit = sum(1 for sp in spans if sp and (sp in joined))
        span_rate = hit / max(1, len(spans))

        # Numeric grounding
        nums = re.findall(r"-?\d+(?:\.\d+)?", claim.text or "")
        num_ok = 1.0
        if nums:
            num_ok = 1.0 if all(n in joined for n in nums) else 0.0

        # Token overlap (weak)
        ct = set(toks(claim.text or ""))
        et = set(toks(joined))
        j = len(ct & et) / (len(ct | et) or 1)

        return 0.70 * span_rate + 0.20 * num_ok + 0.10 * j

    def propose(self, user_text: str, *, plan: Plan, evidence_map: Dict[str, str], memory_hits: Dict[str, Any], tool_outcomes: List["ToolOutcome"]) -> ProposedAnswer:
        best: Optional[ProposedAnswer] = None
        best_score = -1.0

        for _ in range(self.samples):
            try:
                cand = self.base.propose(user_text, plan=plan, evidence_map=evidence_map, memory_hits=memory_hits, tool_outcomes=tool_outcomes)
            except Exception:
                continue

            # Score candidate by how supported its claims look.
            if not cand.claims:
                score = 0.0
            else:
                per = [self._looks_supported(c, evidence_map) for c in cand.claims]
                score = sum(per) / max(1, len(per))

            # Tie-breaker: more claims isn't always better, but it can be if supported.
            score = score + 0.02 * len(cand.claims)

            if score > best_score:
                best_score = score
                best = cand

        return best or ProposedAnswer(claims=[], draft="I don't know.", proactive=[])


# =============================================================================
# Verifier (proof-carrying gate)
# =============================================================================

class Verifier:
    """
    Enforces "no unsupported claims":
    - Evidence IDs must exist.
    - Support spans must appear verbatim in evidence texts.
    - If claim text contains numbers, those numbers must appear in evidence.
    """
    def __init__(self, memory: MemoryStore, *, require_spans: bool = True, min_span_hits: float = 0.5):
        self.memory = memory
        self.require_spans = require_spans
        self.min_span_hits = float(min_span_hits)

    def _ev_text(self, evid: str) -> Optional[str]:
        ev = self.memory.get_evidence(evid)
        if not ev:
            return None
        return ev["content"] or ""

    @staticmethod
    def _numbers(s: str) -> List[str]:
        return re.findall(r"-?\d+(?:\.\d+)?", s)

    def verify_claim(self, c: Claim) -> Claim:
        # Evidence existence
        ev_texts: Dict[str, str] = {}
        for evid in c.evidence_ids:
            t = self._ev_text(evid)
            if t is not None:
                ev_texts[evid] = t
        if not ev_texts:
            return c.model_copy(update={"status": "rejected", "score": 0.0}) if hasattr(c, "model_copy") else c.copy(update={"status": "rejected", "score": 0.0})  # type: ignore

        # Support spans
        spans = list(c.support_spans or [])
        if self.require_spans and not spans:
            return c.model_copy(update={"status": "rejected", "score": 0.0}) if hasattr(c, "model_copy") else c.copy(update={"status": "rejected", "score": 0.0})  # type: ignore

        hit = 0
        for sp in spans:
            found = any(sp in txt for txt in ev_texts.values())
            if found:
                hit += 1
        span_hit_rate = hit / max(1, len(spans))

        # Numeric grounding
        nums = self._numbers(c.text)
        num_ok = True
        if nums:
            joined = "\n".join(ev_texts.values())
            for n in nums:
                if n not in joined:
                    num_ok = False
                    break

        # Token overlap (weak secondary signal)
        ct = set(toks(c.text))
        et = set(toks("\n".join(ev_texts.values())))
        j = len(ct & et) / (len(ct | et) or 1)

        # Score and decision
        score = 0.65 * span_hit_rate + 0.20 * (1.0 if num_ok else 0.0) + 0.15 * j
        ok = (span_hit_rate >= self.min_span_hits) and num_ok

        status = "verified" if ok else "rejected"
        updated = {"status": status, "score": float(score)}
        if hasattr(c, "model_copy"):
            return c.model_copy(update=updated)
        return c.copy(update=updated)  # type: ignore

    def verify(self, p: ProposedAnswer) -> VerifiedAnswer:
        verified: List[Claim] = []
        rejected = 0
        for c in p.claims:
            vc = self.verify_claim(c)
            if vc.status == "verified":
                verified.append(vc)
            else:
                rejected += 1

        warns: List[str] = []
        if rejected:
            warns.append(f"Rejected {rejected} unsupported claim(s).")
        if not verified:
            return VerifiedAnswer(ok=False, claims=[], response="", warnings=warns + ["No verified claims. Abstaining."])
        return VerifiedAnswer(ok=True, claims=verified, response="", warnings=warns)


# =============================================================================
# Renderer
# =============================================================================

class Renderer:
    def __init__(self, memory: MemoryStore):
        self.memory = memory

    def render(self, v: VerifiedAnswer) -> str:
        if not v.ok:
            suffix = ""
            if v.warnings:
                suffix = " (" + "; ".join(v.warnings) + ")"
            return "I don’t know — I can’t verify any claims from evidence." + suffix

        lines: List[str] = []
        if v.warnings:
            lines.append("⚠️ " + " ".join(v.warnings))
        lines.append("Here’s what I can support from evidence:")
        for c in v.claims:
            lines.append(f"- {c.text}  [evidence: {', '.join(c.evidence_ids)}]")
        return "\n".join(lines)


# =============================================================================
# Initiative manager (impulse threshold + rate limiting)
# =============================================================================

class ProactiveCandidate(BaseModel):
    message: str
    evidence_ids: List[str] = Field(default_factory=list)
    expected_utility: float = 0.5
    confidence: float = 0.5
    actionability: float = 0.5
    interruption_cost: float = 0.3
    risk: float = 0.2


class InitiativeManager:
    def __init__(self, memory: MemoryStore, *, threshold: float = 0.62, cooldown_s: float = 15.0, max_per_hour: int = 10):
        self.memory = memory
        self.threshold = float(threshold)
        self.cooldown_s = float(cooldown_s)
        self.max_per_hour = int(max_per_hour)
        self._lock = threading.Lock()
        self._last_emit = 0.0
        self._recent: List[float] = []

    def _score(self, c: ProactiveCandidate) -> float:
        impulse = (c.expected_utility * c.confidence * c.actionability) - (c.interruption_cost + c.risk)
        return clamp(0.5 + impulse, 0.0, 1.0)

    def submit(self, c: ProactiveCandidate) -> Optional[str]:
        score = self._score(c)
        now = utc_ts()
        with self._lock:
            # cooldown
            if now - self._last_emit < self.cooldown_s:
                return None
            # rate limit
            cutoff = now - 3600.0
            self._recent = [t for t in self._recent if t >= cutoff]
            if len(self._recent) >= self.max_per_hour:
                return None

            if score >= self.threshold:
                pid = self.memory.add_proactive(c.message, score=score, evidence_ids=c.evidence_ids)
                self._last_emit = now
                self._recent.append(now)
                return pid
        return None

    def poll(self, limit: int = 3) -> List[Dict[str, Any]]:
        return self.memory.fetch_undelivered_proactive(limit=limit)


# =============================================================================
# Background daemons (deep reasoning that continues after reply)
# =============================================================================

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
        nid = ctx.memory.add_note(title, content, tags=["conversation"], source_ids=[last_user["id"], last_bot["id"]], confidence=0.6)
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

        ctx.initiative.submit(ProactiveCandidate(
            message=f"Pruned {deleted} old episode(s) into note {nid}.",
            evidence_ids=[evid],
            expected_utility=0.6,
            confidence=0.7,
            actionability=0.5,
            interruption_cost=0.25,
            risk=0.15,
        ))


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
                ctx.initiative.submit(ProactiveCandidate(
                    message=f"New connection: note {nid} looks related to {hid} (rrf≈{score:.3f}).",
                    expected_utility=0.55,
                    confidence=0.6,
                    actionability=0.35,
                    interruption_cost=0.25,
                    risk=0.1,
                ))


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
                    ctx.initiative.submit(ProactiveCandidate(
                        message=f"Potential inconsistency: notes titled '{t}' differ a lot (overlap≈{j:.2f}).",
                        expected_utility=0.6,
                        confidence=0.45,
                        actionability=0.25,
                        interruption_cost=0.35,
                        risk=0.25,
                    ))
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
                ctx.initiative.submit(ProactiveCandidate(
                    message=f"Task '{title}' resolved via memory: found {len(notes)} related note(s).",
                    evidence_ids=evidence_ids,
                    expected_utility=0.7,
                    confidence=0.7,
                    actionability=0.6,
                    interruption_cost=0.2,
                    risk=0.1,
                ))
                return

        # If no useful info, block and retry later
        ctx.memory.complete_task(tid, status="blocked", result={"summary": "Insufficient memory. Needs more info/tools."}, evidence_ids=evidence_ids, next_run_ts=utc_ts() + 120.0)


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


# =============================================================================
# Agent kernel (CogOS)
# =============================================================================

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
            self.llm: ChatModel = LlamaCppChatModel(cfg.llama_model, n_ctx=cfg.llama_ctx, n_threads=cfg.llama_threads, n_gpu_layers=cfg.llama_gpu_layers)
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
            ctx = DaemonContext(memory=self.memory, tools=self.tools, initiative=self.initiative, bus=self.bus, llm=None if cfg.llm_backend == "stub" else self.llm, session_id=cfg.session_id)
            daemons: List[Daemon] = [ReflectionDaemon(), ConnectionMiner(), ConsistencyAuditor(), TaskSolverDaemon()]
            if cfg.prune_episodes:
                daemons.insert(1, PruningDaemon(
                    keep_last=cfg.episode_keep_last,
                    batch=cfg.episode_prune_batch,
                    digest_chars=cfg.episode_digest_chars,
                    confidence=cfg.episode_digest_confidence,
                ))
            self.bg = BackgroundRunner(self.bus, ctx, daemons)
            self.bg.start()

        log.info("CogOS started", extra={"extra": {"db": cfg.db, "embedder": embedder.name, "llm": cfg.llm_backend, "planner": cfg.planner, "reasoner": cfg.reasoner, "fts": self.memory._fts_ok}})

    def close(self) -> None:
        if self.bg:
            self.bg.stop()
        self.memory.close()
        log.info("CogOS stopped")

    def _register_tools(self) -> None:
        self.tools.register(ToolSpec("calc", "Safely evaluate arithmetic expressions.", CalcIn, CalcOut, calc_handler, side_effects=False))
        self.tools.register(ToolSpec("now", "Get current local time.", NowIn, NowOut, now_handler, side_effects=False))
        self.tools.register(ToolSpec("memory_search", "Search notes/evidence/skills (hybrid lexical+vector).", MemSearchIn, MemSearchOut, make_mem_search_handler(self.memory), side_effects=False))
        self.tools.register(ToolSpec("read_text_file", "Read a UTF-8 text file under allowed roots.", ReadFileIn, ReadFileOut, make_read_file_handler(self.cfg.read_root), side_effects=False))
        self.tools.register(ToolSpec("write_text_file", "Write a UTF-8 text file under allowed roots.", WriteFileIn, WriteFileOut, make_write_file_handler(self.cfg.write_root), side_effects=True))

    def handle(self, user_text: str) -> Tuple[str, List[Dict[str, Any]]]:
        # episodic log: user
        ep_user = self.memory.add_episode(self.cfg.session_id, "user", user_text, metadata={})
        self.bus.publish("episode_added", {"episode_id": ep_user, "role": "user"})

        plan = self.planner.plan(user_text, tools=self.tools, memory=self.memory)

        tool_outcomes: List[ToolOutcome] = []
        evidence_ids: List[str] = []
        memory_hits: Dict[str, Any] = {"notes": [], "evidence": [], "skills": []}

        # Execute plan
        for step in plan.steps:
            if isinstance(step, StepMemorySearch):
                out = self.tools.execute(ToolCall(name="memory_search", arguments={"query": step.query, "k": step.k}))
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
                nid = self.memory.add_note(step.title, step.content, tags=step.tags, source_ids=[ep_user], confidence=step.confidence)
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

        proposed = self.reasoner.propose(user_text, plan=plan, evidence_map=evidence_map, memory_hits=memory_hits, tool_outcomes=tool_outcomes)
        verified = self.verifier.verify(proposed)
        response = self.renderer.render(verified)

        # episodic log: assistant
        ep_bot = self.memory.add_episode(self.cfg.session_id, "assistant", response, metadata={"plan": _model_dump(plan)})
        self.bus.publish("episode_added", {"episode_id": ep_bot, "role": "assistant"})

        proactive = self.initiative.poll(limit=3)
        return response, proactive


# =============================================================================
# CLI
# =============================================================================

def _install_sigint_handler(stop_flag: threading.Event) -> None:
    def _h(sig, frame):  # noqa: ARG001
        stop_flag.set()
    signal.signal(signal.SIGINT, _h)
    signal.signal(signal.SIGTERM, _h)


def cmd_chat(args: argparse.Namespace) -> int:
    cfg = AgentConfig(
        db=args.db,
        session_id=args.session_id,
        embedder=args.embedder,
        st_model=args.st_model,
        llm_backend=args.llm_backend,
        llama_model=args.llama_model,
        llama_ctx=args.llama_ctx,
        llama_threads=args.llama_threads,
        llama_gpu_layers=args.llama_gpu_layers,
        planner=args.planner,
        reasoner=args.reasoner,
        search_samples=args.search_samples,
        allow_side_effects=args.allow_side_effects,
        read_root=tuple(args.read_root),
        write_root=tuple(args.write_root),
        prune_episodes=args.prune_episodes,
        episode_keep_last=args.episode_keep_last,
        episode_prune_batch=args.episode_prune_batch,
        episode_digest_chars=args.episode_digest_chars,
        episode_digest_confidence=args.episode_digest_confidence,
        background=not args.no_background,
        json_logs=args.json_logs,
        log_level=args.log_level,
        initiative_threshold=args.initiative_threshold,
    )

    agent = CogOS(cfg)
    stop_flag = threading.Event()
    _install_sigint_handler(stop_flag)

    try:
        print("CogOS production baseline. /help for commands. Ctrl-C or /quit to exit.\n")
        while not stop_flag.is_set():
            try:
                user = input("you> ").strip()
            except EOFError:
                break
            except KeyboardInterrupt:
                print()
                break

            if not user:
                continue
            if user in ("/q", "/quit", "/exit"):
                break

            # --- commands ---
            if user == "/help":
                print("Commands:")
                print("  /tools               list tools")
                print("  /notes               list recent notes")
                print("  /skills              list recent skills")
                print("  /episodes            list recent episodes")
                print("  /tasks               list tasks")
                print("  /task add <title>::<desc>   create a task")
                print("  /poll                show proactive messages")
                print("  /quit                exit")
                continue

            if user == "/tools":
                for t in agent.tools.list_tools():
                    print(f"- {t['name']} (side_effects={t['side_effects']}): {t['description']}")
                continue

            if user == "/notes":
                for n in agent.memory.list_notes(limit=10):
                    print(f"- {n['id']}  {short(n['title'], 60)}  tags={n['tags']}  links={len(n['links'])}  conf={n['confidence']:.2f}")
                continue

            if user == "/skills":
                for s in agent.memory.list_skills(limit=10):
                    print(f"- {s['id']}  {short(s['name'], 60)}  {s['description']}")
                continue

            if user == "/episodes":
                eps = agent.memory.recent_episodes(agent.cfg.session_id, limit=10)
                for e in reversed(eps):
                    ts = dt.datetime.fromtimestamp(e["ts"]).isoformat(timespec="seconds")
                    print(f"[{ts}] {e['role']}: {short(e['content'], 120)}")
                continue

            if user == "/tasks":
                tasks = agent.memory.list_tasks(limit=20)
                for t in tasks:
                    ts = dt.datetime.fromtimestamp(t["updated"]).isoformat(timespec="seconds")
                    print(f"- {t['id']}  [{t['status']}] p={t['priority']} tries={t['attempts']}  {t['title']}  ({ts})")
                    print(f"    {t['description']}")
                continue

            if user.startswith("/task add "):
                rest = user[len("/task add "):].strip()
                if "::" in rest:
                    title, desc = rest.split("::", 1)
                else:
                    title, desc = rest, ""
                tid = agent.memory.add_task(title.strip() or "Untitled task", desc.strip() or "No description", priority=1)
                print(f"Created task: {tid}")
                continue

            if user == "/poll":
                msgs = agent.initiative.poll(limit=5)
                if not msgs:
                    print("(no proactive messages)")
                for m in msgs:
                    print(f"[initiative score={m['score']:.2f}] {m['message']}")
                continue

            # --- normal turn ---
            ans, proactive = agent.handle(user)
            print(f"bot> {ans}")
            for pm in proactive:
                print(f"\n[initiative score={pm['score']:.2f}] {pm['message']}\n")
    finally:
        agent.close()
    return 0



def cmd_selftest(args: argparse.Namespace) -> int:
    """
    Quick sanity checks you can run in CI or after installation.

    This is not a full test suite, but it catches common environment/DB issues:
    - SQLite write/read
    - FTS5 availability (if present)
    - calc tool
    - hybrid retrieval (vector + lexical fusion)
    """
    import tempfile

    # Create an isolated temp DB unless user explicitly supplies one.
    tmp_path = None
    db_path = args.db
    if db_path == ":memory:":
        db_path = ":memory:"
    elif not db_path:
        fd, tmp_path = tempfile.mkstemp(prefix="cogos_selftest_", suffix=".db")
        os.close(fd)
        db_path = tmp_path

    cfg = AgentConfig(
        db=db_path,
        session_id="selftest",
        embedder="hash",
        llm_backend="stub",
        planner="rule",
        reasoner="conservative",
        background=False,
        log_level="WARNING",
    )

    agent = CogOS(cfg)
    try:
        # Tool: calc
        out = agent.tools.execute(ToolCall(name="calc", arguments={"expression": "2+2"}))
        assert out.ok, out.error
        assert float(out.output.get("result", -1)) == 4.0

        # Memory: add/search notes
        nid = agent.memory.add_note("Test Note", "hello world", tags=["test"], confidence=0.9)
        hits = agent.memory.search_notes("hello", k=3)
        assert any(h.get("id") == nid for h in hits), "note not retrievable"

        # Tool: memory_search
        ms = agent.tools.execute(ToolCall(name="memory_search", arguments={"query": "hello", "k": 3}))
        assert ms.ok, ms.error
        assert isinstance(ms.output.get("notes"), list)

        # End-to-end: handle() should return grounded calc result
        ans, _ = agent.handle("calculate 10/4")
        assert "2.5" in ans or "2.50" in ans, ans

        print("Selftest OK ✅")
        return 0
    except AssertionError as e:
        print("Selftest FAILED ❌")
        print(str(e))
        return 1
    finally:
        agent.close()
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cogos_prod", description="CogOS production-grade baseline (single-file).")
    sub = p.add_subparsers(dest="cmd", required=True)

    chat = sub.add_parser("chat", help="Interactive chat loop.")
    chat.add_argument("--db", default="cogos.db")
    chat.add_argument("--session-id", default="default")

    chat.add_argument("--embedder", choices=["hash", "st"], default="hash")
    chat.add_argument("--st-model", default="all-MiniLM-L6-v2")

    chat.add_argument("--llm-backend", choices=["stub", "llama_cpp"], default="stub")
    chat.add_argument("--llama-model", default=os.environ.get("COGOS_LLAMA_MODEL", ""))
    chat.add_argument("--llama-ctx", type=int, default=int(os.environ.get("COGOS_LLAMA_CTX", "4096")))
    chat.add_argument("--llama-threads", type=int, default=None)
    chat.add_argument("--llama-gpu-layers", type=int, default=int(os.environ.get("COGOS_LLAMA_GPU_LAYERS", "0")))

    chat.add_argument("--planner", choices=["rule", "llm"], default="rule")
    chat.add_argument("--reasoner", choices=["conservative", "llm", "search"], default="conservative")
    chat.add_argument("--search-samples", type=int, default=4)

    chat.add_argument("--allow-side-effects", action="store_true")
    chat.add_argument("--read-root", action="append", default=["."])
    chat.add_argument("--write-root", action="append", default=["."])

    chat.add_argument("--prune-episodes", action="store_true", help="Summarize+delete old episodes into Notes (background daemon).")
    chat.add_argument("--episode-keep-last", type=int, default=200)
    chat.add_argument("--episode-prune-batch", type=int, default=50)
    chat.add_argument("--episode-digest-chars", type=int, default=280)
    chat.add_argument("--episode-digest-confidence", type=float, default=0.55)

    chat.add_argument("--initiative-threshold", type=float, default=0.62)
    chat.add_argument("--no-background", action="store_true")

    chat.add_argument("--log-level", default="INFO")
    chat.add_argument("--json-logs", action="store_true")

    chat.set_defaults(func=cmd_chat)

    selftest = sub.add_parser("selftest", help="Run quick internal sanity tests.")
    selftest.add_argument("--db", default="")
    selftest.set_defaults(func=cmd_selftest)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
