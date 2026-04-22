from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, Mapping, TypeVar, Union, get_args, get_origin, get_type_hints

from omegaconf import DictConfig, OmegaConf


@dataclass(slots=True)
class RuntimeConfig:
    """Runtime-only settings shared by all experiments."""

    seed: int = field(default=42, metadata={"help": "Random seed for the whole run."})
    device: str = field(
        default="auto",
        metadata={"help": "Accelerate device override, such as auto, cpu, cuda, or cuda:0."},
    )
    output_dir: str = field(default="outputs", metadata={"help": "Root output directory for Hydra run folders."})


@dataclass(slots=True)
class AppConfig:
    """Top-level application config composed by Hydra."""

    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    dataset: dict[str, Any] | None = field(default=None)
    model: dict[str, Any] | None = field(default=None)
    trainer: dict[str, Any] | None = field(default=None)


ConfigT = TypeVar("ConfigT")


def project_root() -> Path:
    """Return the repository root inferred from this file location."""

    return Path(__file__).resolve().parents[2]


def configs_dir() -> Path:
    """Return the root directory that stores Hydra config groups."""

    return project_root() / "configs"


def instantiate_dataclass(config_cls: type[ConfigT], config: Mapping[str, Any] | DictConfig | None) -> ConfigT:
    """Instantiate a config dataclass from a plain mapping or OmegaConf node."""

    if not is_dataclass(config_cls):
        raise TypeError(f"{config_cls!r} is not a dataclass type.")

    values = _normalize_mapping(config)
    type_hints = get_type_hints(config_cls)
    field_names = {item.name for item in fields(config_cls)}
    unexpected = sorted(set(values) - field_names)
    if unexpected:
        raise ValueError(f"Unexpected keys for {config_cls.__name__}: {unexpected}")
    normalized_values = {
        item.name: _coerce_value(type_hints.get(item.name, item.type), values[item.name])
        for item in fields(config_cls)
        if item.name in values
    }
    return config_cls(**normalized_values)


def _normalize_mapping(config: Mapping[str, Any] | DictConfig | None) -> dict[str, Any]:
    """Convert supported config inputs into a plain dictionary."""

    if config is None:
        return {}
    if isinstance(config, DictConfig):
        resolved = OmegaConf.to_container(config, resolve=True)
        if resolved is None:
            return {}
        if not isinstance(resolved, dict):
            raise TypeError(f"Expected dict-like component config, got {type(resolved)!r}.")
        return resolved
    return dict(config)


def _coerce_value(type_hint: Any, value: Any) -> Any:
    """Recursively instantiate nested dataclass-like config values."""

    if value is None:
        return None
    if type_hint is Any:
        return value
    if is_dataclass(type_hint):
        if not isinstance(value, (Mapping, DictConfig)):
            raise TypeError(f"Expected mapping-like value for nested dataclass {type_hint!r}, got {type(value)!r}.")
        return instantiate_dataclass(type_hint, value)

    origin = get_origin(type_hint)
    if origin in (tuple, list):
        args = get_args(type_hint)
        items = list(value)
        if origin is tuple and len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce_value(args[0], item) for item in items)
        if origin is tuple and len(args) == len(items):
            return tuple(_coerce_value(arg, item) for arg, item in zip(args, items, strict=True))
        if origin is list and args:
            return [_coerce_value(args[0], item) for item in items]
        return tuple(items) if origin is tuple else items

    if origin in (Union, UnionType):
        for option in get_args(type_hint):
            if option is type(None):
                continue
            return _coerce_value(option, value)
        return value

    return value
