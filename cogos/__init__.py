from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .agent import AgentConfig, CogOS
    from .cli import main

__all__ = ["AgentConfig", "CogOS", "main"]


def __getattr__(name: str):  # pragma: no cover
    # Lazy exports to avoid importing heavy dependencies at package import time.
    if name in {"AgentConfig", "CogOS"}:
        from .agent import AgentConfig, CogOS

        return {"AgentConfig": AgentConfig, "CogOS": CogOS}[name]
    if name == "main":
        from .cli import main

        return main
    raise AttributeError(name)

