from __future__ import annotations

from dataclasses import MISSING, Field, dataclass, field, fields, is_dataclass
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


@dataclass(frozen=True, slots=True)
class ParameterDoc:
    """Rendered description of one config field for docs or CLI help."""

    name: str
    type_name: str
    default: Any
    help_text: str
    required: bool


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


def parameter_docs(config_cls: type[Any]) -> list[ParameterDoc]:
    """Collect normalized parameter docs from a config dataclass."""

    if not is_dataclass(config_cls):
        raise TypeError(f"{config_cls!r} is not a dataclass type.")

    type_hints = get_type_hints(config_cls)
    docs: list[ParameterDoc] = []
    for item in fields(config_cls):
        docs.append(
            ParameterDoc(
                name=item.name,
                type_name=_type_name(type_hints.get(item.name, item.type)),
                default=_field_default(item),
                help_text=str(item.metadata.get("help", "")).strip(),
                required=_is_required(item),
            )
        )
    return docs


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


def _field_default(item: Field[Any]) -> Any:
    """Return a printable default marker for one dataclass field."""

    if item.default_factory is not MISSING:  # type: ignore[attr-defined]
        return "<factory>"
    if item.default is MISSING:
        return "<required>"
    return item.default


def _is_required(item: Field[Any]) -> bool:
    """Return whether the field must be supplied by the caller."""

    return item.default is MISSING and item.default_factory is MISSING  # type: ignore[attr-defined]


def _type_name(type_hint: Any) -> str:
    """Render a readable type name for documentation output."""

    if hasattr(type_hint, "__name__"):
        return type_hint.__name__
    return str(type_hint).replace("typing.", "")
