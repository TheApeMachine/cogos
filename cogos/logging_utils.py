from __future__ import annotations

import datetime as dt
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
from typing import cast


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        extra = record.__dict__.get("extra")
        if isinstance(extra, dict):
            payload.update(cast(dict[str, object], extra))
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(
    level: str = "INFO",
    json_logs: bool = False,
    *,
    log_file: str = "cogos.log",
    log_file_level: str = "DEBUG",
    log_file_max_bytes: int = 10_000_000,
    log_file_backup_count: int = 3,
) -> None:
    root = logging.getLogger()
    # Capture everything at the root, then let handlers filter. This ensures we can
    # always write full logs to file even when console is INFO/WARNING/etc.
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = JsonFormatter() if json_logs else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(fmt)

    handlers: list[logging.Handler] = [handler]

    if log_file:
        try:
            p = Path(str(log_file)).expanduser()
            # Create parent directories if needed.
            if str(p.parent) not in ("", "."):
                p.parent.mkdir(parents=True, exist_ok=True)
            fh = RotatingFileHandler(
                p,
                maxBytes=int(log_file_max_bytes),
                backupCount=int(log_file_backup_count),
                encoding="utf-8",
            )
            fh.setLevel(getattr(logging, str(log_file_level).upper(), logging.DEBUG))
            fh.setFormatter(fmt)
            handlers.append(fh)
        except OSError as e:
            # Don't break startup; keep console logging and warn to stderr.
            sys.stderr.write(f"cogos: WARNING: failed to open log file {log_file!r}: {e}\n")

    root.handlers[:] = handlers


log = logging.getLogger("cogos")

