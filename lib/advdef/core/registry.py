"""Simple string-to-class registries for advdef components."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Dict, Generic, Optional, Type, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Lightweight registry for mapping string identifiers to callables or classes."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: Dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        """Decorator to register an item under the provided name."""

        def decorator(obj: T) -> T:
            if name in self._items:
                raise ValueError(f"{self._kind!r} '{name}' already registered.")
            self._items[name] = obj
            return obj

        return decorator

    def add(self, name: str, obj: T) -> None:
        if name in self._items:
            raise ValueError(f"{self._kind!r} '{name}' already registered.")
        self._items[name] = obj

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._items)) or "<none>"
            raise KeyError(
                f"Unknown {self._kind!r} '{name}'. Available: {available}"
            ) from exc

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __iter__(self):
        return iter(self._items.items())

    def items(self):
        return self._items.items()

    def as_mapping(self) -> Mapping[str, T]:
        return dict(self._items)


DATASETS: Registry[Any] = Registry("dataset")
ATTACKS: Registry[Any] = Registry("attack")
DEFENSES: Registry[Any] = Registry("defense")
MODELS: Registry[Any] = Registry("model")
INFERENCE_BACKENDS: Registry[Any] = Registry("inference backend")
EVALUATORS: Registry[Any] = Registry("evaluator")


def register_dataset(name: str) -> Callable[[T], T]:
    return DATASETS.register(name)


def register_attack(name: str) -> Callable[[T], T]:
    return ATTACKS.register(name)


def register_defense(name: str) -> Callable[[T], T]:
    return DEFENSES.register(name)


def register_model(name: str) -> Callable[[T], T]:
    return MODELS.register(name)


def register_inference(name: str) -> Callable[[T], T]:
    return INFERENCE_BACKENDS.register(name)


def register_evaluator(name: str) -> Callable[[T], T]:
    return EVALUATORS.register(name)
