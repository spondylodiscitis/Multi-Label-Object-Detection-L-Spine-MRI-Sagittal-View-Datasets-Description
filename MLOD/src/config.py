from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigNode(dict):
    """Dictionary with attribute access."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        if isinstance(value, dict) and not isinstance(value, ConfigNode):
            value = ConfigNode(value)
            self[key] = value
        return value

    __setattr__ = dict.__setitem__


def _to_node(value: Any) -> Any:
    if isinstance(value, dict):
        return ConfigNode({key: _to_node(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_node(item) for item in value]
    return value


def load_config(path: str | Path) -> ConfigNode:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        data: Dict[str, Any] = yaml.safe_load(file) or {}

    return _to_node(deepcopy(data))
