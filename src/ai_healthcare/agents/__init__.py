from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ai_healthcare.agents.commander import (
        Commander,
        CommanderError,
        CommanderInvalidJSONError,
        CommanderOutputError,
        CommanderPolicyError,
        CommanderSchemaError,
    )


__all__ = [
    "Commander",
    "CommanderError",
    "CommanderInvalidJSONError",
    "CommanderOutputError",
    "CommanderPolicyError",
    "CommanderSchemaError",
]


def __getattr__(name: str) -> Any:
    """Load Commander exports only when they are requested."""

    if name not in __all__:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )

    module = import_module(".commander", package=__name__)
    value = getattr(module, name)

    globals()[name] = value

    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
