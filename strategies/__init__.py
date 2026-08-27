"""Strategy package — auto-discovers all Strategy subclasses."""

import importlib
import pkgutil
from typing import Type

from .strategy import Strategy, TradeResult

_registry: dict[str, Type[Strategy]] = {}


def _discover_strategies():
    """Import all modules in this package and register Strategy subclasses."""
    import strategies as pkg

    for importer, modname, ispkg in pkgutil.iter_modules(pkg.__path__):
        if modname.startswith("_") or modname == "strategy":
            continue
        module = importlib.import_module(f".{modname}", package="strategies")
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, Strategy)
                and attr is not Strategy
                and hasattr(attr, "name")
            ):
                _registry[attr.name] = attr


def get_registry() -> dict[str, Type[Strategy]]:
    """Return the strategy registry, discovering if empty."""
    if not _registry:
        _discover_strategies()
    return _registry


def list_strategies() -> list[str]:
    """Return sorted list of strategy names."""
    return sorted(get_registry().keys())


__all__ = ["Strategy", "TradeResult", "get_registry", "list_strategies"]