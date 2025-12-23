from __future__ import annotations

from typing import Any, Callable, cast


# Define these unconditionally so static analyzers always see them as exported.
class ValidationError(RuntimeError):  # noqa: D101
    pass


class BaseModel:  # noqa: D101
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("pydantic is required. Install with: pip install pydantic")

    def dict(self, *args: Any, **kwargs: Any) -> dict[str, object]:
        raise RuntimeError("pydantic is required. Install with: pip install pydantic")

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, object]:
        raise RuntimeError("pydantic is required. Install with: pip install pydantic")


def Field(  # noqa: D103
    default: object = None,
    *,
    default_factory: Callable[[], object] | None = None,
    **kwargs: Any,
) -> object:
    _ = kwargs
    if default_factory is not None:
        return default_factory()
    return default


try:  # pragma: no cover
    from pydantic import BaseModel as _PBaseModel
    from pydantic import Field as _PField
    from pydantic import ValidationError as _PValidationError

    BaseModel = _PBaseModel
    Field = _PField
    ValidationError = _PValidationError
except Exception:
    pass


def model_dump(m: BaseModel) -> dict[str, object]:
    """
    Compatibility wrapper for pydantic v1/v2.
    """

    # v2
    fn = getattr(m, "model_dump", None)
    if callable(fn):
        return cast(dict[str, object], fn())

    # v1
    fn = getattr(m, "dict", None)
    if callable(fn):
        return cast(dict[str, object], fn())

    return {}


def model_json_schema(model: type[BaseModel]) -> dict[str, object]:
    """
    Compatibility wrapper for pydantic v1/v2.
    """

    # v2
    fn = getattr(model, "model_json_schema", None)
    if callable(fn):
        return cast(dict[str, object], fn())

    # v1
    fn = getattr(model, "schema", None)
    if callable(fn):
        return cast(dict[str, object], fn())

    return {}


# Back-compat aliases (avoid importing the underscored names in new code).
_model_dump = model_dump
_model_json_schema = model_json_schema


__all__ = [
    "BaseModel",
    "Field",
    "ValidationError",
    "model_dump",
    "model_json_schema",
]

