"""
CogOS — Production-Grade Baseline

This file is a thin entrypoint wrapper. The implementation lives in `cogos/`.

Run:
    python v1.py chat --db cogos.db

Or:
    python cogos_prod.py chat --db cogos.db

The original single-file reference is preserved as `cogos_single_file.py`.
"""

from __future__ import annotations

from cogos.cli import main


if __name__ == "__main__":
    raise SystemExit(main())

