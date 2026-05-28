from __future__ import annotations

import copy
import ast
import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path = "configs/zuco_mvp.yaml") -> dict[str, Any]:
    """Load a YAML config with a helpful dependency error."""
    path = Path(path)
    try:
        import yaml
    except ImportError as exc:
        cfg = _load_simple_yaml(path)
        cfg["_config_path"] = str(path)
        return cfg

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = str(path)
    return cfg


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return {}
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            inner = value[1:-1].strip()
            if not inner:
                return []
            return [_parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip("\"'")


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    """Small fallback parser for the repository's config YAML.

    It supports nested dictionaries by indentation and scalar values. It is not
    a general YAML parser; install PyYAML for full YAML support.
    """
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        parsed = _parse_scalar(value)
        parent[key.strip()] = parsed
        if isinstance(parsed, dict):
            stack.append((indent, parsed))
    return root


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in config.items() if not k.startswith("_")}
    try:
        import yaml
    except ImportError:
        path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        return
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(serializable, f, sort_keys=False)


def deep_get(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    cur: Any = config
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def deep_set(config: dict[str, Any], dotted_key: str, value: Any) -> dict[str, Any]:
    out = copy.deepcopy(config)
    cur = out
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value
    return out
