from __future__ import annotations

from typing import Any, Callable, Self, TypeVar, overload

T = TypeVar("T")


class ValidationError(Exception): ...


class BaseModel:
    # Minimal pydantic-like surface used by this project.
    def __init__(self, *args: object, **kwargs: object) -> None: ...
    def dict(self, *args: object, **kwargs: object) -> dict[str, object]: ...
    def model_dump(self, *args: object, **kwargs: object) -> dict[str, object]: ...
    def copy(self, *args: object, **kwargs: object) -> Self: ...
    def model_copy(self, *args: object, **kwargs: object) -> Self: ...


@overload
def Field(
    default: T,
    *,
    default_factory: None = ...,
    **kwargs: object,
) -> T: ...

@overload
def Field(*, default_factory: None = ..., **kwargs: object) -> object: ...


@overload
def Field(
    default: object = ...,
    *,
    default_factory: Callable[[], T],
    **kwargs: object,
) -> T: ...


def model_dump(m: BaseModel) -> dict[str, object]: ...
def model_json_schema(model: type[BaseModel]) -> dict[str, object]: ...

# Back-compat aliases.
_model_dump = model_dump
_model_json_schema = model_json_schema


