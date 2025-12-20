from __future__ import annotations

from typing import Any, Dict

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

