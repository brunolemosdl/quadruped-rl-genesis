"""YAML/JSON helpers, deep merge, and nested path updates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


class _NoAliasSafeDumper(yaml.SafeDumper):
    """YAML dumper that keeps repeated values expanded for readability.

    Disables YAML anchor/alias expansion so repeated Python objects are
    serialized inline instead of as references.
    """

    def ignore_aliases(self, data: Any) -> bool:
        """Always disable YAML anchors for repeated Python objects.

        Args:
            data (Any): Value about to be serialized by ``yaml.dump``.

        Returns:
            bool: Always ``True`` so repeated values are emitted inline instead
                of through YAML aliases.
        """
        del data
        return True


def ensure_parent_directory(path: Path) -> None:
    """Create the parent directory for a file path if it does not exist.

    Args:
        path (Path): Target file path whose parent directory should be ensured.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file and validate that the root object is a mapping.

    Args:
        path (Path): YAML file to read.

    Returns:
        dict[str, Any]: Parsed YAML payload. Empty files are treated as ``{}``.

    Raises:
        TypeError: If the YAML root is not a dictionary-like mapping.
    """
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping in YAML file: {path}")

    return data


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Write a mapping to YAML using a stable and readable layout.

    Args:
        path (Path): Destination YAML file.
        payload (dict[str, Any]): Mapping to serialize.
    """
    ensure_parent_directory(path)

    with path.open("w", encoding="utf-8") as handle:
        yaml.dump(
            payload,
            handle,
            Dumper=_NoAliasSafeDumper,
            sort_keys=False,
            allow_unicode=False,
        )


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file and validate that the root object is a mapping.

    Args:
        path (Path): JSON file to read.

    Returns:
        dict[str, Any]: Parsed JSON payload.

    Raises:
        TypeError: If the JSON root is not a dictionary-like mapping.
    """
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping in JSON file: {path}")

    return data


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a mapping to JSON using indentation suitable for inspection.

    Args:
        path (Path): Destination JSON file.
        payload (dict[str, Any]): Mapping to serialize.
    """
    ensure_parent_directory(path)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries, favoring override values.

    Nested dictionaries are merged key by key, while non-dictionary values from
    ``override`` replace the corresponding values from ``base``.

    Args:
        base (dict[str, Any]): Original mapping.
        override (dict[str, Any]): Mapping whose values take precedence.

    Returns:
        dict[str, Any]: New merged mapping. The input dictionaries are not
            mutated.
    """
    merged: dict[str, Any] = dict(base)

    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value

    return merged


def set_nested_value_by_path(
    payload: dict[str, Any], dotted_path: str, value: Any
) -> None:
    """Assign a value in a nested mapping using dotted path notation.

    Missing intermediate dictionaries are created automatically.

    Args:
        payload (dict[str, Any]): Mapping updated in place.
        dotted_path (str): Path such as ``"algorithm.policy_kwargs.net_arch"``.
        value (Any): Value assigned at the final path segment.
    """
    parts = dotted_path.split(".")
    current = payload

    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}

        current = current[part]

    current[parts[-1]] = value


def to_json_compatible(value: Any) -> Any:
    """Recursively convert common project values into JSON-compatible data.

    This helper preserves plain JSON-compatible types and normalizes a few
    project-specific objects such as ``Path`` instances and nested tuples.

    Args:
        value (Any): Value to normalize.

    Returns:
        Any: JSON-compatible structure when conversion is supported. Unknown
            scalar values are returned unchanged.
    """
    if isinstance(value, dict):
        return {str(key): to_json_compatible(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_json_compatible(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    return value
