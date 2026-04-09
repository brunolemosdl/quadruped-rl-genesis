"""Load and merge experiment and algorithm YAML configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from quadruped_rl_genesis.config.versions import load_named_algorithm_version
from quadruped_rl_genesis.services.artifacts import ArtifactStore
from quadruped_rl_genesis.settings import AppSettings
from quadruped_rl_genesis.utils.io import deep_merge, read_yaml


def get_experiment_config_path(settings: AppSettings, experiment_name: str) -> Path:
    """Return the YAML path for an experiment configuration profile.

    Args:
        settings (AppSettings): Global application settings.
        experiment_name (str): Experiment profile name without extension.

    Returns:
        Path: Experiment configuration file path.
    """
    return settings.configs_root / "experiments" / f"{experiment_name}.yaml"


def get_algorithm_config_path(settings: AppSettings, algorithm: str) -> Path:
    """Return the YAML path for an algorithm default profile.

    Args:
        settings (AppSettings): Global application settings.
        algorithm (str): Algorithm name.

    Returns:
        Path: Default algorithm configuration path.
    """
    return settings.configs_root / "algorithms" / algorithm / "default.yaml"


def get_optuna_config_path(settings: AppSettings) -> Path:
    """Return the YAML path for the shared Optuna study configuration.

    Args:
        settings (AppSettings): Global application settings.

    Returns:
        Path: Shared Optuna configuration path.
    """
    return settings.configs_root / "optuna" / "default.yaml"


def load_experiment_config(
    settings: AppSettings,
    experiment_name: str,
) -> dict[str, Any]:
    """Load the YAML payload for an experiment profile.

    Args:
        settings (AppSettings): Global application settings.
        experiment_name (str): Experiment profile name.

    Returns:
        dict[str, Any]: Parsed experiment configuration.
    """
    return read_yaml(get_experiment_config_path(settings, experiment_name))


def load_algorithm_config(settings: AppSettings, algorithm: str) -> dict[str, Any]:
    """Load the YAML payload for an algorithm profile.

    Args:
        settings (AppSettings): Global application settings.
        algorithm (str): Algorithm name.

    Returns:
        dict[str, Any]: Parsed algorithm configuration.
    """
    return read_yaml(get_algorithm_config_path(settings, algorithm))


def load_optuna_config(settings: AppSettings) -> dict[str, Any]:
    """Load the shared Optuna configuration payload.

    Args:
        settings (AppSettings): Global application settings.

    Returns:
        dict[str, Any]: Parsed Optuna configuration.
    """
    return read_yaml(get_optuna_config_path(settings))


def load_resolved_config(
    settings: AppSettings,
    artifact_store: ArtifactStore,
    experiment_name: str,
    algorithm: str,
    hyperparams_source: str,
) -> dict[str, Any]:
    """Resolve experiment and algorithm profiles into one runtime payload.

    When a named hyperparameter source is requested, the algorithm defaults are
    merged with the selected override version before returning the final payload.

    Args:
        settings (AppSettings): Global application settings.
        artifact_store (ArtifactStore): Artifact helper used to resolve named
            algorithm versions.
        experiment_name (str): Experiment profile name.
        algorithm (str): Algorithm name.
        hyperparams_source (str): Named hyperparameter source such as
            ``"default"`` or ``"optuna"``.

    Returns:
        dict[str, Any]: Combined runtime configuration with ``experiment``,
            ``algorithm``, and ``runtime`` sections.
    """
    experiment_config = load_experiment_config(settings, experiment_name)
    if algorithm == "random" and hyperparams_source != "default":
        raise ValueError(
            "Algorithm 'random' supports only hyperparams_source 'default' "
            "(no Optuna or named algorithm variants)."
        )

    algorithm_config = load_algorithm_config(settings, algorithm)

    resolved = {
        "experiment": experiment_config,
        "algorithm": algorithm_config,
        "runtime": {
            "experiment_name": experiment_name,
            "algorithm": algorithm,
            "hyperparams_source": hyperparams_source,
        },
    }

    if hyperparams_source == "optuna":
        best_overrides_path = artifact_store.get_optuna_best_overrides_path(
            experiment_name,
            algorithm,
        )
        if not best_overrides_path.exists():
            raise FileNotFoundError(
                "Optuna hyperparameter source was requested, but no tuned "
                f"overrides were found at {best_overrides_path}."
            )

        best_overrides = read_yaml(best_overrides_path)
        base_hyperparams_source = str(
            best_overrides.get("base_hyperparams_source", "default")
        )

        if base_hyperparams_source != "default":
            base_algorithm_override = load_named_algorithm_version(
                settings=settings,
                experiment_name=experiment_name,
                algorithm=algorithm,
                version_name=base_hyperparams_source,
            )
            resolved["algorithm"]["algorithm"] = deep_merge(
                resolved["algorithm"]["algorithm"],
                base_algorithm_override,
            )

        algorithm_override = best_overrides.get("algorithm", {})
        if isinstance(algorithm_override, dict):
            resolved["algorithm"]["algorithm"] = deep_merge(
                resolved["algorithm"]["algorithm"],
                algorithm_override,
            )

        experiment_override = best_overrides.get("experiment", {})
        if isinstance(experiment_override, dict):
            resolved["experiment"] = deep_merge(
                resolved["experiment"],
                experiment_override,
            )

        resolved["runtime"]["base_hyperparams_source"] = base_hyperparams_source
        return resolved

    if hyperparams_source != "default":
        named_algorithm_config = load_named_algorithm_version(
            settings=settings,
            experiment_name=experiment_name,
            algorithm=algorithm,
            version_name=hyperparams_source,
        )
        resolved["algorithm"]["algorithm"] = deep_merge(
            resolved["algorithm"]["algorithm"],
            named_algorithm_config,
        )

    return resolved
