"""Lazy re-exports -- see ss/__init__.py for rationale."""

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ._system import (
        ContinuousTimeSystem,
        DiscreteTimeSystem,
        System,
        simulate,
    )
    from .cart_pole import CartPoleSystem
    from .discrete_hmm import HiddenMarkovModel
    # from .linear import (
    #     ContinuousTimeLinearSystem,
    #     DiscreteTimeLinearSystem,
    # )
    # from .nonlinear import (
    #     ContinuousTimeNonlinearSystem,
    #     DiscreteTimeNonlinearSystem,
    # )

_EXPORTS: dict[str, str] = {
    "ContinuousTimeSystem": "_system",
    "DiscreteTimeSystem": "_system",
    "System": "_system",
    "simulate": "_system",
    "CartPoleSystem": "cart_pole",
    "HiddenMarkovModel": "discrete_hmm",
    # "ContinuousTimeLinearSystem": "linear",
    # "DiscreteTimeLinearSystem": "linear",
    # "ContinuousTimeNonlinearSystem": "nonlinear",
    # "DiscreteTimeNonlinearSystem": "nonlinear",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return __all__
