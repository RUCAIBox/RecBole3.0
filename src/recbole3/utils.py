from __future__ import annotations

from omegaconf import DictConfig


def require_component_cfg(cfg: DictConfig, component: str) -> DictConfig:
    """Return component config node and fail fast if missing.

    Args:
        cfg: The configuration dict.
        component: The component name (e.g., "dataset", "model", "trainer").

    Returns:
        The component configuration.

    Raises:
        ValueError: If the component is missing from config.
        TypeError: If the component is not a DictConfig.
    """
    value = cfg.get(component)
    if value is None:
        raise ValueError(
            f"Missing `{component}` configuration. Add a config group override such as "
            f"`{component}=your_component`."
        )
    if not isinstance(value, DictConfig):
        raise TypeError(f"`{component}` config must be a DictConfig, got {type(value)!r}.")
    return value


def require_component_name(component_cfg: DictConfig, component: str) -> str:
    """Return registered component name from component config.

    Args:
        component_cfg: The component configuration.
        component: The component name (e.g., "dataset", "model").

    Returns:
        The registered component name.

    Raises:
        ValueError: If the component name is not set.
    """
    name = component_cfg.get("name")
    if not name:
        raise ValueError(f"`{component}.name` must be set to a known component name.")
    return str(name)


__all__ = [
    "require_component_cfg",
    "require_component_name",
]