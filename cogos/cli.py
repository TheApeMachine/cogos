from __future__ import annotations

import argparse
import datetime as dt
import os
import signal
import threading
import traceback
from pathlib import Path
from typing import List, Optional

from .agent import AgentConfig, CogOS
from .model import HFModelSpec, ensure_hf_model
from .tools import ToolCall
from .util import short


def _install_sigint_handler(stop_flag: threading.Event) -> None:
    def _h(sig, frame):  # noqa: ARG001
        stop_flag.set()

    signal.signal(signal.SIGINT, _h)
    signal.signal(signal.SIGTERM, _h)


def cmd_chat(args: argparse.Namespace) -> int:
    if args.web_allow_domain is None:
        web_allow_domain = ["wikipedia.org", "arxiv.org", "github.com"]
    else:
        web_allow_domain = args.web_allow_domain

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
        planner=args.planner,
        reasoner=args.reasoner,
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
        print(
            f"CogOS production baseline (llm={agent.cfg.llm_backend}, planner={agent.cfg.planner}, reasoner={agent.cfg.reasoner}). "
            "/help for commands. Ctrl-C or /quit to exit.\n"
        )
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

            # --- normal turn ---
            try:
                ans, proactive = agent.handle(user)
            except Exception as e:
                print(f"bot> ERROR: {e}")
                traceback.print_exc()
                continue

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

    chat.add_argument("--llm-backend", choices=["stub", "llama_cpp"], default="stub")
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

    chat.add_argument("--planner", choices=["rule", "llm"], default="rule")
    chat.add_argument("--reasoner", choices=["conservative", "llm", "search"], default="conservative")
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
        help="If memory_search yields no hits, automatically run web_search once to gather evidence (context firewall).",
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
        default=0.0,
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
