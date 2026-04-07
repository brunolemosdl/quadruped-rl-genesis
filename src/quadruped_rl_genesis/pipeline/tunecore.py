"""Shared Optuna tuning helpers (trial status, phase plans, search-space filtering)."""

from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path
from typing import Any

import optuna

from quadruped_rl_genesis.utils.io import to_json_compatible, write_json


def write_trial_status(path: Path, payload: dict[str, Any]) -> None:
    """Persist the current Optuna trial status to JSON.

    Args:
        path (Path): Output JSON path.
        payload (dict[str, Any]): Status payload to serialize.
    """
    write_json(path, to_json_compatible(payload))


def study_trial_counts(study: optuna.Study) -> Counter[str]:
    """Count study trials by Optuna state name.

    Args:
        study (optuna.Study): Optuna study to inspect.

    Returns:
        Counter[str]: Number of trials grouped by ``TrialState.name``.
    """
    return Counter(trial.state.name for trial in study.get_trials(deepcopy=False))


def include_stochastic_metrics(algorithm: str) -> bool:
    """Check whether an algorithm should also be evaluated stochastically.

    Args:
        algorithm (str): Algorithm name.

    Returns:
        bool: ``True`` for off-policy algorithms that benefit from stochastic
            evaluation.
    """
    return algorithm.lower() in {"sac", "td3"}


def spec_enabled_for_phase(spec: dict[str, Any], phase_name: str | None) -> bool:
    """Return whether one Optuna search-space entry applies to the named phase.

    Args:
        spec (dict[str, Any]): Single search-space definition, optionally with
            ``phase`` or ``phases``.
        phase_name (str | None): Active phase label, or ``None`` to accept all.

    Returns:
        bool: ``True`` when the spec has no phase filter or matches ``phase_name``.
    """
    if phase_name is None:
        return True
    if "phase" in spec:
        return str(spec["phase"]) == phase_name
    if "phases" in spec:
        phases = spec["phases"]
        if isinstance(phases, str):
            return phases == phase_name
        if isinstance(phases, list):
            return phase_name in {str(item) for item in phases}
    return True


def filtered_search_space_payload(
    payload: dict[str, Any],
    *,
    phase_name: str | None,
) -> dict[str, Any]:
    """Return a deep copy of ``payload`` with phase-filtered ``optuna.search_space``.

    Args:
        payload (dict[str, Any]): Experiment or algorithm config containing ``optuna``.
        phase_name (str | None): Phase used by :func:`spec_enabled_for_phase`.

    Returns:
        dict[str, Any]: New mapping where irrelevant search-space keys are removed.
    """
    payload_copy = copy.deepcopy(payload)
    search_space = payload_copy.get("optuna", {}).get("search_space", {})
    filtered = {
        key: value
        for key, value in search_space.items()
        if isinstance(value, dict) and spec_enabled_for_phase(value, phase_name)
    }
    if "optuna" not in payload_copy:
        payload_copy["optuna"] = {}
    payload_copy["optuna"]["search_space"] = filtered
    return payload_copy


def phase_plan(optuna_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the ordered list of tuning phases from shared Optuna YAML.

    Args:
        optuna_config (dict[str, Any]): Parsed ``configs/optuna/default.yaml`` root.

    Returns:
        list[dict[str, Any]]: Phase dicts with ``name``, ``n_trials``, timesteps, and
            evaluation cadence fields. Returns a single pseudo-phase when phases are
            disabled.
    """
    phases_cfg = optuna_config.get("phases", {})
    if not bool(phases_cfg.get("enabled", False)):
        return [
            {
                "name": "single",
                "n_trials": int(optuna_config["n_trials"]),
                "trial_timesteps": int(optuna_config["trial_timesteps"]),
                "eval_episodes": int(optuna_config["eval_episodes"]),
                "eval_freq_env_steps": int(optuna_config["eval_freq_env_steps"]),
                "timeout_seconds": optuna_config.get("timeout_seconds"),
            }
        ]

    sequence = list(phases_cfg.get("sequence", []))
    if not sequence:
        return [
            {
                "name": "single",
                "n_trials": int(optuna_config["n_trials"]),
                "trial_timesteps": int(optuna_config["trial_timesteps"]),
                "eval_episodes": int(optuna_config["eval_episodes"]),
                "eval_freq_env_steps": int(optuna_config["eval_freq_env_steps"]),
                "timeout_seconds": optuna_config.get("timeout_seconds"),
            }
        ]

    plan: list[dict[str, Any]] = []
    for phase_name in sequence:
        phase_cfg = phases_cfg.get(str(phase_name), {})
        plan.append(
            {
                "name": str(phase_name),
                "n_trials": int(phase_cfg.get("n_trials", optuna_config["n_trials"])),
                "trial_timesteps": int(
                    phase_cfg.get("trial_timesteps", optuna_config["trial_timesteps"])
                ),
                "eval_episodes": int(
                    phase_cfg.get("eval_episodes", optuna_config["eval_episodes"])
                ),
                "eval_freq_env_steps": int(
                    phase_cfg.get(
                        "eval_freq_env_steps", optuna_config["eval_freq_env_steps"]
                    )
                ),
                "timeout_seconds": phase_cfg.get(
                    "timeout_seconds", optuna_config.get("timeout_seconds")
                ),
            }
        )
    return plan
