from __future__ import annotations

import hashlib
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .embeddings import EmbeddingModel
from .logging_utils import log
from .np_compat import np
from .util import jdump, jload, new_id, short, toks, utc_ts


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
        # `timeout` is sqlite3's busy timeout (seconds); also set PRAGMA busy_timeout (ms) for clarity.
        self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=5.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA busy_timeout=5000;")
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._fts_ok = False
        self._init_schema()

    @property
    def fts_ok(self) -> bool:
        return bool(self._fts_ok)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---- schema ----

    def _init_schema(self) -> None:
        with self._lock:
            c = self._conn.cursor()
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS episodes(
                    id TEXT PRIMARY KEY,
                    ts REAL NOT NULL,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT
                );
            """
            )
            c.execute(
                """
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
            """
            )
            c.execute(
                """
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
            """
            )
            c.execute(
                """
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
            """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks(
                    id TEXT PRIMARY KEY,
                    created REAL NOT NULL,
                    updated REAL NOT NULL,
                    status TEXT NOT NULL,             -- queued|running|blocked|done|failed
                    priority INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    parent_id TEXT,                   -- parent task id (for subtasks / dependencies)
                    payload TEXT,
                    result TEXT,
                    evidence_ids TEXT,
                    error TEXT,
                    attempts INTEGER NOT NULL,
                    next_run_ts REAL
                );
            """
            )
            self._ensure_task_schema(c)
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS proactive(
                    id TEXT PRIMARY KEY,
                    ts REAL NOT NULL,
                    score REAL NOT NULL,
                    message TEXT NOT NULL,
                    evidence_ids TEXT,
                    delivered INTEGER NOT NULL DEFAULT 0
                );
            """
            )

            # FTS5 (lexical)
            self._fts_ok = self._try_init_fts(c)
            self._conn.commit()

    def _ensure_task_schema(self, c: sqlite3.Cursor) -> None:
        """
        Best-effort task table migrations for existing DBs.

        Keeps the runtime robust even when the schema evolves across versions.
        """
        try:
            cols = {str(r["name"]) for r in c.execute("PRAGMA table_info(tasks)").fetchall()}
        except Exception:
            cols = set()
        if "parent_id" not in cols:
            try:
                c.execute("ALTER TABLE tasks ADD COLUMN parent_id TEXT;")
            except Exception:
                # If the column exists already or ALTER TABLE isn't possible, ignore.
                pass
        # Helpful for dependency checks
        try:
            c.execute("CREATE INDEX IF NOT EXISTS idx_tasks_parent_id ON tasks(parent_id);")
        except Exception:
            pass

    def _try_init_fts(self, c: sqlite3.Cursor) -> bool:
        try:
            c.execute("CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(id UNINDEXED, title, content);")
            c.execute("CREATE VIRTUAL TABLE IF NOT EXISTS evidence_fts USING fts5(id UNINDEXED, kind, content);")
            c.execute("CREATE VIRTUAL TABLE IF NOT EXISTS skills_fts USING fts5(id UNINDEXED, name, description);")
            return True
        except sqlite3.OperationalError as e:
            log.warning("FTS5 unavailable; lexical search disabled: %s", e, exc_info=True)
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

    def add_episode(
        self, session_id: str, role: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
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
            out.append(
                {
                    "id": r["id"],
                    "ts": r["ts"],
                    "role": r["role"],
                    "content": r["content"],
                    "metadata": jload(r["metadata"]) or {},
                }
            )
        return out

    def count_episodes(self, session_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(1) AS n FROM episodes WHERE session_id=?", (session_id,)
            ).fetchone()
        return int(row["n"]) if row else 0

    def oldest_episodes(self, session_id: str, limit: int) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, ts, role, content, metadata FROM episodes WHERE session_id=? ORDER BY ts ASC LIMIT ?",
                (session_id, int(limit)),
            ).fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "id": r["id"],
                    "ts": r["ts"],
                    "role": r["role"],
                    "content": r["content"],
                    "metadata": jload(r["metadata"]) or {},
                }
            )
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

    def add_evidence(
        self, kind: str, content: str, metadata: Optional[Dict[str, Any]] = None, *, dedupe: bool = True
    ) -> str:
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
        return {
            "id": row["id"],
            "ts": row["ts"],
            "kind": row["kind"],
            "content": row["content"],
            "metadata": jload(row["metadata"]) or {},
        }

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
                (
                    nid,
                    now,
                    now,
                    title,
                    content,
                    jdump(tags or []),
                    jdump(links or []),
                    jdump(source_ids or []),
                    float(confidence),
                    blob,
                    dim,
                ),
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

    def update_note(
        self,
        nid: str,
        *,
        title: Optional[str] = None,
        content: Optional[str] = None,
        tags: Optional[List[str]] = None,
        links: Optional[List[Dict[str, Any]]] = None,
        confidence: Optional[float] = None,
    ) -> bool:
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
            rows = self._conn.execute(
                "SELECT id, updated, title, tags, links, confidence FROM notes ORDER BY updated DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "id": r["id"],
                    "updated": r["updated"],
                    "title": r["title"],
                    "tags": jload(r["tags"]) or [],
                    "links": jload(r["links"]) or [],
                    "confidence": float(r["confidence"]),
                }
            )
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
            rows = self._conn.execute(
                "SELECT id, updated, name, description FROM skills ORDER BY updated DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [
            {"id": r["id"], "updated": r["updated"], "name": r["name"], "description": short(r["description"], 160)}
            for r in rows
        ]

    # ---- tasks ----

    def add_task(
        self,
        title: str,
        description: str,
        *,
        priority: int = 0,
        payload: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
        next_run_ts: Optional[float] = None,
    ) -> str:
        tid = new_id("task")
        now = utc_ts()
        # Allow parent linkage to be passed via payload for convenience/back-compat.
        if parent_id is None and payload:
            pid = payload.get("parent_id") or payload.get("parent_task_id") or payload.get("parent")
            if isinstance(pid, str) and pid.strip():
                parent_id = pid.strip()
        with self._lock:
            self._conn.execute(
                "INSERT INTO tasks(id, created, updated, status, priority, title, description, parent_id, payload, result, evidence_ids, error, attempts, next_run_ts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    tid,
                    now,
                    now,
                    "queued",
                    int(priority),
                    title,
                    description,
                    parent_id,
                    jdump(payload or {}),
                    jdump({}),
                    jdump([]),
                    "",
                    0,
                    next_run_ts,
                ),
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
            out.append(
                {
                    "id": r["id"],
                    "updated": r["updated"],
                    "status": r["status"],
                    "priority": r["priority"],
                    "title": r["title"],
                    "description": short(r["description"], 220),
                    "attempts": r["attempts"],
                    "next_run_ts": r["next_run_ts"],
                }
            )
        return out

    def fetch_runnable_task(self, now_ts: Optional[float] = None) -> Optional[Dict[str, Any]]:
        now_ts = utc_ts() if now_ts is None else now_ts
        with self._lock:
            # Avoid running a parent task before its children reach a terminal status.
            row = self._conn.execute(
                "SELECT * FROM tasks t "
                "WHERE t.status IN ('queued','blocked') "
                "AND (t.next_run_ts IS NULL OR t.next_run_ts <= ?) "
                "AND NOT EXISTS ("
                "  SELECT 1 FROM tasks c WHERE c.parent_id = t.id AND c.status NOT IN ('done','failed')"
                ") "
                "ORDER BY t.priority DESC, t.updated ASC LIMIT 1",
                (now_ts,),
            ).fetchone()
            if not row:
                return None
            # mark running
            self._conn.execute("UPDATE tasks SET status='running', updated=? WHERE id=?", (utc_ts(), row["id"]))
            self._conn.commit()
        return dict(row)

    def complete_task(
        self,
        tid: str,
        *,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        evidence_ids: Optional[List[str]] = None,
        error: str = "",
        next_run_ts: Optional[float] = None,
    ) -> None:
        now = utc_ts()
        with self._lock:
            # Capture parent linkage (if any) so we can wake parents when children finish.
            parent_id = None
            try:
                prow = self._conn.execute("SELECT parent_id FROM tasks WHERE id=?", (tid,)).fetchone()
                if prow and prow["parent_id"]:
                    parent_id = str(prow["parent_id"])
            except Exception:
                parent_id = None
            self._conn.execute(
                "UPDATE tasks SET updated=?, status=?, result=?, evidence_ids=?, error=?, attempts=attempts+1, next_run_ts=? WHERE id=?",
                (now, status, jdump(result or {}), jdump(evidence_ids or []), error, next_run_ts, tid),
            )
            # If this is a child task and it reached a terminal state, allow its parent to run sooner.
            if parent_id and status in ("done", "failed"):
                try:
                    row = self._conn.execute(
                        "SELECT COUNT(1) AS n FROM tasks WHERE parent_id=? AND status NOT IN ('done','failed')",
                        (parent_id,),
                    ).fetchone()
                    remaining = int(row["n"]) if row else 0
                    if remaining == 0:
                        self._conn.execute(
                            "UPDATE tasks SET updated=?, status=CASE WHEN status='blocked' THEN 'queued' ELSE status END, next_run_ts=NULL "
                            "WHERE id=? AND status NOT IN ('done','failed')",
                            (now, parent_id),
                        )
                except Exception:
                    pass
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
        return [
            {
                "id": r["id"],
                "ts": r["ts"],
                "score": r["score"],
                "message": r["message"],
                "evidence_ids": jload(r["evidence_ids"]) or [],
            }
            for r in rows
        ]

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
            tags = list(n.get("tags") or [])
            source_ids = list(n.get("source_ids") or [])
            conf = float(n.get("confidence", 0.5))
            # Heuristic trust: user-sourced notes are easier to poison; treat as lower trust by default.
            trust = 0.25 + 0.5 * conf
            if any(str(s).startswith("ep_") for s in source_ids):
                trust = min(trust, 0.35)
            if ("conversation" in tags) or ("episode_digest" in tags):
                trust = min(trust, 0.45)
            trust = max(0.0, min(1.0, trust))
            out.append(
                {
                    "id": n["id"],
                    "title": n["title"],
                    "content_snip": short(n["content"], 800),
                    "tags": tags,
                    "links": n["links"],
                    "source_ids": source_ids,
                    "confidence": conf,
                    "trust_score": float(trust),
                    "score": float(score),
                }
            )
        return out

    def search_evidence(
        self, query: str, k: int = 5, *, fts_k: int = 30, vec_k: int = 30
    ) -> List[Dict[str, Any]]:
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
            md = ev.get("metadata") or {}
            try:
                # Default to a conservative trust when metadata is missing. Evidence
                # without an explicit trust_score should not be treated as fully trusted.
                trust = float(md.get("trust_score", 0.5))
            except Exception:
                trust = 0.5
            out.append(
                {
                    "id": ev["id"],
                    "kind": ev["kind"],
                    "content_snip": short(ev["content"], 800),
                    "source_type": str(md.get("source_type", "")),
                    "trust_score": float(trust),
                    "score": float(score),
                }
            )
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
            out.append(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "description": short(row["description"], 800),
                    "score": float(score),
                }
            )
        return out
