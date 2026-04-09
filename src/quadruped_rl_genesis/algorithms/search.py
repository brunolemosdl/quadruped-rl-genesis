"""Optuna trial helpers for nested hyperparameter overrides."""

from __future__ import annotations

from typing import Any

import optuna

from quadruped_rl_genesis.utils.io import set_nested_value_by_path


def _suggest_optuna_value(trial: optuna.Trial, name: str, spec: dict[str, Any]) -> Any:
    """Sample one value from a project-defined Optuna search-space spec.

    Args:
        trial (optuna.Trial): Active Optuna trial.
        name (str): Parameter name used inside Optuna.
        spec (dict[str, Any]): Search-space specification with a ``type`` key
            and the corresponding bounds or choices.

    Returns:
        Any: Value sampled by Optuna according to the given specification.

    Raises:
        ValueError: If the search-space ``type`` is not supported.
    """
    suggestion_type = spec["type"]

    if suggestion_type == "float":
        return trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            log=bool(spec.get("log", False)),
            step=spec.get("step"),
        )
    if suggestion_type == "int":
        return trial.suggest_int(
            name,
            int(spec["low"]),
            int(spec["high"]),
            log=bool(spec.get("log", False)),
            step=int(spec.get("step", 1)),
        )
    if suggestion_type == "categorical":
        choices = spec.get("choices")
        if isinstance(choices, dict):
            label = trial.suggest_categorical(name, list(choices.keys()))
            return choices[label]

        return trial.suggest_categorical(name, list(choices))

    raise ValueError(f"Unsupported Optuna suggestion type: {suggestion_type}")


def build_trial_parameter_overrides(
    trial: optuna.Trial,
    algorithm_config_payload: dict[str, Any],
) -> dict[str, Any]:
    """Build nested configuration overrides for a single Optuna trial.

    The function walks the configured search space, samples each parameter, and
    writes the result into a nested dictionary that mirrors the project config
    structure.

    Args:
        trial (optuna.Trial): Active Optuna trial.
        algorithm_config_payload (dict[str, Any]): Algorithm config payload that
            may contain an ``optuna.search_space`` mapping.

    Returns:
        dict[str, Any]: Nested override payload ready to be merged into the base
            algorithm configuration.
    """
    overrides: dict[str, Any] = {}
    search_space = algorithm_config_payload.get("optuna", {}).get("search_space", {})

    for dotted_path, spec in search_space.items():
        suggested_value = _suggest_optuna_value(
            trial, dotted_path.replace(".", "__"), spec
        )
        set_nested_value_by_path(overrides, dotted_path, suggested_value)

    return overrides


def build_trial_environment_overrides(
    trial: optuna.Trial,
    experiment_config_payload: dict[str, Any],
) -> dict[str, Any]:
    """Build ``environment`` overrides from experiment-level Optuna search space.

    Search-space paths are interpreted relative to the ``environment`` section.
    For convenience, paths starting with ``environment.`` are also accepted.

    Args:
        trial (optuna.Trial): Active Optuna trial.
        experiment_config_payload (dict[str, Any]): Experiment config payload
            that may include ``optuna.search_space``.

    Returns:
        dict[str, Any]: Nested overrides for ``environment`` only.
    """
    overrides: dict[str, Any] = {}
    search_space = experiment_config_payload.get("optuna", {}).get("search_space", {})

    for dotted_path, spec in search_space.items():
        env_path = (
            dotted_path.removeprefix("environment.")
            if dotted_path.startswith("environment.")
            else dotted_path
        )
        suggested_value = _suggest_optuna_value(
            trial, f"env__{env_path.replace('.', '__')}", spec
        )
        set_nested_value_by_path(overrides, env_path, suggested_value)

    return overrides
