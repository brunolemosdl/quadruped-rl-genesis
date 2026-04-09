"""Named static algorithm version files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from quadruped_rl_genesis.settings import AppSettings
from quadruped_rl_genesis.utils.io import read_yaml


def get_versions_path(
    settings: AppSettings,
    algorithm: str,
    version_name: str,
) -> Path:
    """Return the YAML path for a named static algorithm version file.

    Args:
        settings (AppSettings): Global application settings.
        algorithm (str): Algorithm name.
        version_name (str): Named version label.

    Returns:
        Path: Version YAML path under ``configs/algorithms/<algorithm>``.
    """
    return settings.configs_root / "algorithms" / algorithm / f"{version_name}.yaml"


def load_named_algorithm_version(
    settings: AppSettings,
    experiment_name: str,
    algorithm: str,
    version_name: str,
) -> dict[str, Any]:
    """Load one static named algorithm version.

    Version files may either contain a direct payload for ``version_name`` or
    an experiment-variant mapping under ``variants``.

    Args:
        settings (AppSettings): Global application settings.
        experiment_name (str): Experiment profile used for variant selection.
        algorithm (str): Algorithm name.
        version_name (str): Static version label, such as ``utd_gs2``.

    Returns:
        dict[str, Any]: Algorithm override payload selected for the experiment.

    Raises:
        FileNotFoundError: If the requested version file does not exist.
        TypeError: If the version file structure is malformed.
        KeyError: If the requested version or experiment variant is missing.
    """
    version_path = get_versions_path(settings, algorithm, version_name)
    if not version_path.exists():
        raise FileNotFoundError(
            f"Named hyperparameter source '{version_name}' was not found at {version_path}."
        )

    payload = read_yaml(version_path)
    versions = payload.get("algorithm_versions")

    if not isinstance(versions, dict):
        raise TypeError(
            f"Expected 'algorithm_versions' mapping in named algorithm config: {version_path}"
        )

    if version_name not in versions:
        raise KeyError(
            f"Named hyperparameter source '{version_name}' is missing in {version_path}."
        )

    version_payload = versions[version_name]
    if not isinstance(version_payload, dict):
        raise TypeError(
            f"Expected a mapping for named hyperparameter source '{version_name}' in {version_path}."
        )

    if "variants" in version_payload:
        variants = version_payload["variants"]

        if not isinstance(variants, dict):
            raise TypeError(
                f"Expected 'variants' mapping for named hyperparameter source '{version_name}' in {version_path}."
            )

        if experiment_name not in variants:
            available = ", ".join(sorted(variants)) or "<none>"
            raise KeyError(
                f"Experiment '{experiment_name}' is not available in named hyperparameter "
                f"source '{version_name}'. Available variants: {available}."
            )

        selected = variants[experiment_name]
    else:
        selected = version_payload

    if not isinstance(selected, dict):
        raise TypeError(
            f"Expected a mapping for experiment '{experiment_name}' in {version_path}."
        )

    return selected
