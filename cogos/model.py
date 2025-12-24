from __future__ import annotations

import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from .logging_utils import log


@dataclass(frozen=True)
class HFModelSpec:
    repo_id: str
    filename: str
    revision: str = "main"


DEFAULT_HF_MODEL = HFModelSpec(
    repo_id=os.environ.get("COGOS_LLAMA_HF_REPO", "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"),
    filename=os.environ.get("COGOS_LLAMA_HF_FILE", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"),
    revision=os.environ.get("COGOS_LLAMA_HF_REV", "main"),
)


def _parse_hf_ref(ref: str) -> HFModelSpec:
    """
    Parse `hf://org/repo/path/to/file.gguf[@revision]`.
    """
    if not ref.startswith("hf://"):
        raise ValueError("hf ref must start with 'hf://'")
    rest = ref[len("hf://") :]
    revision = "main"
    if "@" in rest:
        rest, revision = rest.rsplit("@", 1)
        revision = revision.strip() or "main"

    parts = [p for p in rest.split("/") if p]
    if len(parts) < 3:
        raise ValueError("hf ref must be 'hf://org/repo/<filename>'")

    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:])
    return HFModelSpec(repo_id=repo_id, filename=filename, revision=revision)


def _hf_resolve_url(spec: HFModelSpec) -> str:
    # Works for public repos; will 302 to the actual storage backend (LFS).
    return f"https://huggingface.co/{spec.repo_id}/resolve/{spec.revision}/{spec.filename}?download=true"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    req = urllib.request.Request(url, headers={"User-Agent": "cogos/0.1 (+https://github.com/theapemachine/architecture)"})
    try:
        with urllib.request.urlopen(req) as r, tmp.open("wb") as f:  # noqa: S310
            shutil.copyfileobj(r, f)
        os.replace(tmp, dest)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def ensure_hf_model(spec: HFModelSpec, *, model_dir: Path) -> Path:
    """
    Ensure the model is present on disk, downloading if needed.

    Files are stored under: `<model_dir>/hf/<repo_id>/<revision>/<filename>`.
    """
    dest = model_dir / "hf" / spec.repo_id / spec.revision / spec.filename
    if dest.exists():
        return dest

    url = _hf_resolve_url(spec)
    log.info("Downloading GGUF model: %s -> %s", url, dest)
    _download(url, dest)
    return dest


def resolve_llama_model_path(
    llama_model: str,
    *,
    auto_download: bool,
    model_dir: str,
    default_spec: Optional[HFModelSpec] = None,
) -> str:
    """
    Resolve a llama.cpp model reference into a local file path.

    Supported values:
    - local filesystem path to a `.gguf` file
    - `hf://org/repo/path/to/file.gguf[@revision]` (download if missing)
    - empty + `auto_download=True` (download DEFAULT_HF_MODEL unless overridden)
    """
    model_dir_path = Path(model_dir).expanduser().resolve()
    spec = default_spec or DEFAULT_HF_MODEL

    m = (llama_model or "").strip()
    if m.lower() in ("default", "auto"):
        m = ""
        auto_download = True

    if m.startswith("hf://"):
        hf_spec = _parse_hf_ref(m)
        return str(ensure_hf_model(hf_spec, model_dir=model_dir_path))

    if not m:
        # If the default model is already present on disk, use it even when
        # auto_download is disabled. (No network, no surprise downloads.)
        default_path = model_dir_path / "hf" / spec.repo_id / spec.revision / spec.filename
        if default_path.exists():
            return str(default_path.resolve())
        if not auto_download:
            raise ValueError(
                "--llama-model is required (or place the default model under models/hf/..., "
                "or use --llama-auto-download / --llama-model hf://...)"
            )
        return str(ensure_hf_model(spec, model_dir=model_dir_path))

    p = Path(m).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"llama model not found: {p}")
    return str(p.resolve())

