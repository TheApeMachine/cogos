from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .agent import AgentConfig, CogOS
from .model import HFModelSpec, ensure_hf_model
from .tools import ToolCall
from .util import short


def _install_sigint_handler(stop_flag: threading.Event) -> None:
    def _h(sig, frame):  # noqa: ARG001
        stop_flag.set()

    signal.signal(signal.SIGINT, _h)
    signal.signal(signal.SIGTERM, _h)


class _Style:
    def __init__(self, enabled: bool):
        self.enabled = bool(enabled)

    def _wrap(self, code: str, s: str) -> str:
        if not self.enabled:
            return s
        return f"\033[{code}m{s}\033[0m"

    def dim(self, s: str) -> str:
        return self._wrap("2", s)

    def bold(self, s: str) -> str:
        return self._wrap("1", s)

    def red(self, s: str) -> str:
        return self._wrap("31", s)

    def green(self, s: str) -> str:
        return self._wrap("32", s)

    def yellow(self, s: str) -> str:
        return self._wrap("33", s)

    def cyan(self, s: str) -> str:
        return self._wrap("36", s)


class _Spinner:
    def __init__(self, *, enabled: bool, text: str = "working"):
        self.enabled = bool(enabled)
        self.text = text
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._th: threading.Thread | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        if self._th and self._th.is_alive():
            return
        self._stop.clear()
        self._th = threading.Thread(target=self._run, name="cogos-spinner", daemon=True)
        self._th.start()

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop.set()
        th = self._th
        if th:
            th.join(timeout=0.5)
        with self._lock:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()

    def println(self, s: str) -> None:
        # Print without fighting the spinner line.
        if not self.enabled:
            print(s)
            return
        with self._lock:
            sys.stdout.write("\r\033[K")
            sys.stdout.write(s + "\n")
            sys.stdout.flush()

    def _run(self) -> None:
        frames = ["|", "/", "-", "\\"]
        i = 0
        while not self._stop.is_set():
            with self._lock:
                sys.stdout.write(f"\r{frames[i % len(frames)]} {self.text}...\033[K")
                sys.stdout.flush()
            i += 1
            time.sleep(0.08)


def _pretty_json_or_text(s: str) -> Tuple[str, bool]:
    """
    Best-effort pretty print for JSON strings.
    Returns (text, was_json).
    """
    txt = str(s or "").strip()
    if not txt:
        return "", False
    try:
        obj = json.loads(txt)
    except Exception:
        return txt, False
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2), True
    except Exception:
        return txt, True


def cmd_chat(args: argparse.Namespace) -> int:
    if args.web_allow_domain is None:
        web_allow_domain = ["wikipedia.org", "arxiv.org", "github.com"]
    else:
        web_allow_domain = args.web_allow_domain

    planner = args.planner or ("rule" if args.llm_backend == "stub" else "llm")
    reasoner = args.reasoner or ("conservative" if args.llm_backend == "stub" else "search")

    cfg = AgentConfig(
        db=args.db,
        session_id=args.session_id,
        embedder=args.embedder,
        st_model=args.st_model,
        llm_backend=args.llm_backend,
        llama_model=args.llama_model,
        llama_model_dir=args.llama_model_dir,
        llama_auto_download=args.llama_auto_download,
        llama_hf_repo=args.llama_hf_repo,
        llama_hf_file=args.llama_hf_file,
        llama_hf_rev=args.llama_hf_rev,
        llama_ctx=args.llama_ctx,
        llama_threads=args.llama_threads,
        llama_gpu_layers=args.llama_gpu_layers,
        llama_verbose=args.llama_verbose,
        ollama_host=args.ollama_host,
        ollama_model=args.ollama_model,
        planner=planner,
        reasoner=reasoner,
        search_samples=args.search_samples,
        allow_side_effects=args.allow_side_effects,
        allow_web_search=args.allow_web_search,
        auto_research=args.auto_research,
        web_allow_domains=tuple(web_allow_domain),
        web_deny_domains=tuple(args.web_deny_domain or []),
        min_evidence_trust=args.min_evidence_trust,
        notary=args.notary,
        notary_priority=args.notary_priority,
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
        llm_name = getattr(agent.llm, "name", str(agent.cfg.llm_backend))
        is_tty = sys.stdout.isatty()
        style = _Style(enabled=is_tty and (not args.no_color))
        spinner = _Spinner(enabled=is_tty and (not args.no_spinner), text="thinking")
        show_tools = bool(args.show_tools)

        print(style.dim("—" * 72))
        print(
            style.bold("CogOS")
            + " "
            + style.dim(f"(llm={llm_name}, planner={agent.cfg.planner}, reasoner={agent.cfg.reasoner})")
        )
        print(style.dim("Type /help for commands. Ctrl-C or /quit to exit."))
        print(style.dim("—" * 72) + "\n")
        if llm_name == "stub":
            print(
                style.yellow("NOTE: ")
                + "You are running with the stub LLM backend (deterministic, tool/memory-only mode). "
                "Use `--llm-backend auto|llama_cpp|ollama` for a real LLM.\n"
            )
        while not stop_flag.is_set():
            try:
                user = input(style.green("you> ") if style.enabled else "you> ").strip()
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
                print("  /ev <evidence_id> [full]    show evidence content")
                print("  /trace               show last plan + tool outcomes")
                print("  /show tools          toggle tool-call tracing")
                print("  /quit                exit")
                continue

            if user == "/tools":
                for t in agent.tools.list_tools():
                    print(f"- {t['name']} (side_effects={t['side_effects']}): {t['description']}")
                continue

            if user == "/notes":
                for n in agent.memory.list_notes(limit=10):
                    print(
                        f"- {n['id']}  {short(n['title'], 60)}  tags={n['tags']}  links={len(n['links'])}  conf={n['confidence']:.2f}"
                    )
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
                    print(
                        f"- {t['id']}  [{t['status']}] p={t['priority']} tries={t['attempts']}  {t['title']}  ({ts})"
                    )
                    print(f"    {t['description']}")
                continue

            if user.startswith("/task add "):
                rest = user[len("/task add ") :].strip()
                if "::" in rest:
                    title, desc = rest.split("::", 1)
                else:
                    title, desc = rest, ""
                tid = agent.memory.add_task(
                    title.strip() or "Untitled task",
                    desc.strip() or "No description",
                    priority=1,
                )
                print(f"Created task: {tid}")
                continue

            if user == "/poll":
                msgs = agent.initiative.poll(limit=5)
                if not msgs:
                    print("(no proactive messages)")
                for m in msgs:
                    print(f"[initiative score={m['score']:.2f}] {m['message']}")
                continue

            if user.startswith("/show "):
                rest = user[len("/show ") :].strip().lower()
                if rest in ("tools", "tool", "trace"):
                    show_tools = not show_tools
                    print(f"show_tools={show_tools}")
                else:
                    print("Usage: /show tools")
                continue

            if user.startswith("/ev ") or user.startswith("/evidence "):
                parts = user.split()
                if len(parts) < 2:
                    print("Usage: /ev <evidence_id> [full]")
                    continue
                evid = parts[1].strip()
                full = any(p.lower() == "full" for p in parts[2:])
                ev = agent.memory.get_evidence(evid)
                if not ev:
                    print(style.red("No such evidence: ") + evid)
                    continue
                md = ev.get("metadata") or {}
                hdr = f"{ev.get('id')}  kind={ev.get('kind')}  ts={ev.get('ts')}"
                print(style.bold(hdr) if style.enabled else hdr)
                try:
                    ts = float(md.get("trust_score", 0.0))
                except Exception:
                    ts = 0.0
                src = str(md.get("source_type", md.get("tool", "")) or "")
                meta_line = f"source={src} trust_score={ts:.2f}" if src else f"trust_score={ts:.2f}"
                print(style.dim(meta_line) if style.enabled else meta_line)
                content = str(ev.get("content") or "")
                pretty, was_json = _pretty_json_or_text(content)
                if (not full) and len(pretty) > 2200:
                    pretty = pretty[:2199] + "…"
                label = "content (json):" if was_json else "content:"
                print(style.dim(label) if style.enabled else label)
                print(pretty)
                continue

            if user == "/trace":
                tr = agent.last_trace or {}
                plan = tr.get("plan") or {}
                steps = (plan.get("steps") or []) if isinstance(plan, dict) else []
                if steps:
                    hdr = "plan:"
                    print(style.bold(hdr) if style.enabled else hdr)
                    for st in steps:
                        if not isinstance(st, dict):
                            continue
                        t = st.get("type")
                        if t == "tool_call":
                            print(f"  - tool_call {st.get('tool')} args={st.get('arguments')}")
                        elif t == "memory_search":
                            print(f"  - memory_search query={short(str(st.get('query','')), 80)} k={st.get('k')}")
                        else:
                            print(f"  - {t}")
                outs = tr.get("tool_outcomes") or []
                if outs:
                    hdr = "tool outcomes:"
                    print(style.bold(hdr) if style.enabled else hdr)
                    for o in outs:
                        if not isinstance(o, dict):
                            continue
                        tool = o.get("tool")
                        ok = bool(o.get("ok"))
                        evid = o.get("evidence_id")
                        line = f"  - {tool} ok={ok}"
                        if evid:
                            line += f" evidence={evid}"
                        if (not ok) and o.get("error"):
                            line += f" error={short(str(o.get('error')), 160)}"
                        print(line)
                verified = tr.get("verified") or {}
                claims = (verified.get("claims") or []) if isinstance(verified, dict) else []
                if claims:
                    hdr = "verified claims:"
                    print(style.bold(hdr) if style.enabled else hdr)
                    for c in claims:
                        if not isinstance(c, dict):
                            continue
                        txt = str(c.get("text") or "").strip()
                        kind = str(c.get("kind") or "")
                        score = c.get("score")
                        eids = c.get("evidence_ids") or []
                        spans = c.get("support_spans") or []
                        print(f"  - ({kind}) score={score} evidence={eids}")
                        print(f"    text: {short(txt, 180)}")
                        if spans:
                            print(f"    spans: {spans}")
                if not steps and not outs:
                    print("(no trace available yet)")
                continue

            # --- normal turn ---
            def _on_trace(kind: str, payload: Dict[str, Any]) -> None:
                if not show_tools:
                    return
                if kind == "tool_call":
                    tool = str(payload.get("tool") or "")
                    args_ = payload.get("arguments") or {}
                    line = f"{style.dim('→') if style.enabled else '->'} tool {tool} {style.dim(str(args_)) if style.enabled else str(args_)}"
                    spinner.println(line)
                elif kind == "tool_outcome":
                    ok = bool(payload.get("ok"))
                    tool = str(payload.get("tool") or "")
                    evid = str(payload.get("evidence_id") or "")
                    if ok:
                        spinner.println(f"{style.dim('←') if style.enabled else '<-'} tool {tool} ok evidence={evid}")
                    else:
                        err = short(str(payload.get('error') or ''), 160)
                        spinner.println(f"{style.dim('←') if style.enabled else '<-'} tool {tool} ERROR {err}")

            try:
                spinner.start()
                ans, proactive = agent.handle(user, on_trace=_on_trace)
            except Exception as e:
                spinner.stop()
                if style.enabled:
                    print(style.cyan("bot> ") + style.red(f"ERROR: {e}"))
                else:
                    print(f"bot> ERROR: {e}")
                traceback.print_exc()
                continue
            finally:
                spinner.stop()
            print((style.cyan("bot> ") if style.enabled else "bot> ") + ans)
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


def cmd_download_model(args: argparse.Namespace) -> int:
    spec = HFModelSpec(repo_id=args.hf_repo, filename=args.hf_file, revision=args.hf_rev)
    path = ensure_hf_model(spec, model_dir=Path(args.model_dir).expanduser().resolve())
    print(str(path))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cogos_prod", description="CogOS production-grade baseline (split into modules).")
    sub = p.add_subparsers(dest="cmd", required=True)

    chat = sub.add_parser("chat", help="Interactive chat loop.")
    chat.add_argument("--db", default="cogos.db")
    chat.add_argument("--session-id", default="default")

    chat.add_argument("--embedder", choices=["hash", "st"], default="hash")
    chat.add_argument("--st-model", default="all-MiniLM-L6-v2")

    chat.add_argument("--llm-backend", choices=["auto", "llama_cpp", "ollama", "stub"], default="auto")
    chat.add_argument(
        "--llama-model",
        default=os.environ.get("COGOS_LLAMA_MODEL", ""),
        help="Local `.gguf` path or `hf://org/repo/path/to/file.gguf[@rev]`",
    )
    chat.add_argument("--llama-model-dir", default=os.environ.get("COGOS_LLAMA_MODEL_DIR", "models"))
    chat.add_argument("--llama-auto-download", action="store_true", help="Download default model if --llama-model is empty.")
    chat.add_argument("--llama-hf-repo", default=os.environ.get("COGOS_LLAMA_HF_REPO", "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"))
    chat.add_argument("--llama-hf-file", default=os.environ.get("COGOS_LLAMA_HF_FILE", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"))
    chat.add_argument("--llama-hf-rev", default=os.environ.get("COGOS_LLAMA_HF_REV", "main"))
    chat.add_argument("--llama-ctx", type=int, default=int(os.environ.get("COGOS_LLAMA_CTX", "4096")))
    chat.add_argument("--llama-threads", type=int, default=None)
    chat.add_argument("--llama-gpu-layers", type=int, default=int(os.environ.get("COGOS_LLAMA_GPU_LAYERS", "0")))
    chat.add_argument("--llama-verbose", action="store_true", help="Enable verbose llama.cpp logging.")

    chat.add_argument("--ollama-host", default=os.environ.get("COGOS_OLLAMA_HOST", "http://localhost:11434"))
    chat.add_argument("--ollama-model", default=os.environ.get("COGOS_OLLAMA_MODEL", ""))

    chat.add_argument("--planner", choices=["rule", "llm"], default=None)
    chat.add_argument("--reasoner", choices=["conservative", "llm", "search"], default=None)
    chat.add_argument("--search-samples", type=int, default=4)

    chat.add_argument("--allow-side-effects", action="store_true")
    chat.add_argument(
        "--allow-web-search",
        action="store_true",
        help="Enable the web_search tool (network). If disabled, CogOS is fully offline/local-first.",
    )
    chat.add_argument(
        "--auto-research",
        action="store_true",
        help="If memory_search yields no hits (or only low-trust hits), automatically run web_search once to gather evidence (context firewall).",
    )
    chat.add_argument(
        "--web-allow-domain",
        action="append",
        default=None,
        help="Allowlist domains for web_search (repeatable).",
    )
    chat.add_argument(
        "--web-deny-domain",
        action="append",
        default=[],
        help="Denylist domains for web_search (repeatable).",
    )
    chat.add_argument(
        "--min-evidence-trust",
        type=float,
        default=0.5,
        help="Verifier threshold: reject claims that cite evidence with trust_score below this value (0.0-1.0).",
    )
    chat.add_argument(
        "--notary",
        action="store_true",
        help="Enable the Notary: if auto-research steering fails to produce verified claims, escalate for human review.",
    )
    chat.add_argument(
        "--notary-priority",
        type=int,
        default=10,
        help="Priority used for Notary-created human review tasks.",
    )
    chat.add_argument("--read-root", action="append", default=["."])
    chat.add_argument("--write-root", action="append", default=["."])

    chat.add_argument(
        "--prune-episodes",
        action="store_true",
        help="Summarize+delete old episodes into Notes (background daemon).",
    )
    chat.add_argument("--episode-keep-last", type=int, default=200)
    chat.add_argument("--episode-prune-batch", type=int, default=50)
    chat.add_argument("--episode-digest-chars", type=int, default=280)
    chat.add_argument("--episode-digest-confidence", type=float, default=0.55)

    chat.add_argument("--initiative-threshold", type=float, default=0.62)
    chat.add_argument("--no-background", action="store_true")

    chat.add_argument("--log-level", default="INFO")
    chat.add_argument("--json-logs", action="store_true")
    chat.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")
    chat.add_argument("--no-spinner", action="store_true", help="Disable the in-progress spinner.")
    chat.add_argument("--show-tools", action="store_true", help="Show tool calls/outcomes for each turn.")

    chat.set_defaults(func=cmd_chat)

    selftest = sub.add_parser("selftest", help="Run quick internal sanity tests.")
    selftest.add_argument("--db", default="")
    selftest.set_defaults(func=cmd_selftest)

    dl = sub.add_parser("download-model", help="Download a default GGUF model from Hugging Face.")
    dl.add_argument("--model-dir", default=os.environ.get("COGOS_LLAMA_MODEL_DIR", "models"))
    dl.add_argument("--hf-repo", default=os.environ.get("COGOS_LLAMA_HF_REPO", "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"))
    dl.add_argument("--hf-file", default=os.environ.get("COGOS_LLAMA_HF_FILE", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"))
    dl.add_argument("--hf-rev", default=os.environ.get("COGOS_LLAMA_HF_REV", "main"))
    dl.set_defaults(func=cmd_download_model)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
