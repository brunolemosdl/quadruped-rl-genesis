"""Load policies and run offline evaluation on the navigation task."""

from __future__ import annotations

import gc
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from quadruped_rl_genesis.algorithms.factory import load_sb3_model
from quadruped_rl_genesis.config.loader import (
    load_experiment_config,
    load_resolved_config,
)
from quadruped_rl_genesis.environments.factory import build_vector_env
from quadruped_rl_genesis.services.artifacts import (
    ArtifactStore,
    find_run_root_from_model_path,
    find_vecnormalize_path_from_model_path,
)
from quadruped_rl_genesis.services.logger import configure_logging, get_logger
from quadruped_rl_genesis.services.metrics import (
    RandomPolicy,
    build_evaluation_protocol,
    evaluate_policy_variants,
    summarize_evaluation_bundle,
)
from quadruped_rl_genesis.services.platform import collect_runtime_context
from quadruped_rl_genesis.services.runtime import initialize_genesis
from quadruped_rl_genesis.settings import AppSettings
from quadruped_rl_genesis.utils.io import read_yaml, to_json_compatible, write_json
from quadruped_rl_genesis.utils.seed import set_global_seeds

LOGGER = get_logger(__name__)


def apply_eval_experiment_environment_overlay(
    settings: AppSettings,
    resolved_config: dict[str, Any],
    eval_experiment_name: str,
) -> None:
    """Replace ``experiment.environment`` using another experiment profile YAML.

    Used to evaluate a policy trained on one terrain while simulating another
    (for example ``irregular`` or ``rough``).

    Args:
        settings (AppSettings): Application paths for resolving experiment files.
        resolved_config (dict[str, Any]): Training-time resolved config mutated in-place.
        eval_experiment_name (str): Name under ``configs/experiments`` (without ``.yaml``).

    Raises:
        ValueError: When the overlay profile has no top-level ``environment`` block.
    """
    exp_block = resolved_config.get("experiment") or {}
    train_name = (exp_block.get("experiment") or {}).get("name")
    if train_name == eval_experiment_name:
        return
    overlay = load_experiment_config(settings, eval_experiment_name)
    if "environment" not in overlay:
        raise ValueError(
            f"Experiment profile {eval_experiment_name!r} has no top-level 'environment'."
        )
    resolved_config["experiment"]["environment"] = deepcopy(overlay["environment"])
    resolved_config.setdefault("runtime", {})["eval_environment_experiment"] = (
        eval_experiment_name
    )
    LOGGER.info(
        "Eval environment overlay | train_experiment=%s eval_experiment=%s",
        train_name,
        eval_experiment_name,
    )


def _include_stochastic_metrics(algorithm: str) -> bool:
    """Check whether an algorithm should also be evaluated stochastically.

    Args:
        algorithm (str): Algorithm name.

    Returns:
        bool: ``True`` for off-policy algorithms that benefit from stochastic
            evaluation.
    """
    return algorithm.lower() in {"sac", "td3"}


def run_evaluation(
    settings: AppSettings,
    experiment_name: str,
    algorithm: str,
    hyperparams_source: str,
    seed: int,
    genesis_device: str,
    algorithm_device: str,
    model_kind: str,
    model_path: str | None,
    run_id: str | None,
    train_seed: int | None,
    episodes: int | None,
    eval_experiment_name: str | None = None,
) -> dict[str, Any]:
    """Evaluate a saved policy and persist a JSON summary.

    Args:
        settings (AppSettings): Global application settings.
        experiment_name (str): Experiment profile name.
        algorithm (str): Algorithm name.
        hyperparams_source (str): Hyperparameter source used when the run cannot
            be inferred from the model path.
        seed (int): Evaluation seed.
        genesis_device (str): Device for Genesis simulation.
        algorithm_device (str): Device for SB3/RL algorithm.
        model_kind (str): Requested model kind, typically ``"best"`` or
            ``"final"``.
        model_path (str | None): Optional explicit model path.
        run_id (str | None): Optional explicit training run identifier.
        train_seed (int | None): Optional training seed used to resolve the
            target run when ``model_path`` and ``run_id`` are not provided.
        episodes (int | None): Optional override for evaluation episode count.
        eval_experiment_name (str | None): If set, replace the training run's
            ``environment`` block with that profile (e.g. ``"irregular"`` or
            ``"rough"`` to test a flat-trained policy on non-flat terrain).

    Returns:
        dict[str, Any]: Evaluation summary persisted to disk.

    Raises:
        FileNotFoundError: If no model can be resolved for evaluation.
    """
    artifact_store = ArtifactStore(settings)
    is_random_baseline = algorithm.lower() == "random"
    resolved_model_path: Path | None = (
        None if is_random_baseline else (Path(model_path) if model_path else None)
    )
    selected_run_root: Path | None = None

    if is_random_baseline:
        resolved_config = load_resolved_config(
            settings=settings,
            artifact_store=artifact_store,
            experiment_name=experiment_name,
            algorithm=algorithm,
            hyperparams_source=hyperparams_source,
        )
        if eval_experiment_name:
            apply_eval_experiment_environment_overlay(
                settings,
                resolved_config,
                eval_experiment_name,
            )
        output_root = artifact_store.create_ad_hoc_execution_dir(
            "evaluate",
            experiment_name,
            "random",
            seed,
            label="random_uniform",
        )
        log_file = output_root / "evaluation.log"
        resolved_genesis_device = initialize_genesis(seed=seed, device=genesis_device)
        configure_logging(log_file, settings.log_level)
        set_global_seeds(seed)
        experiment_config = resolved_config["experiment"]
        evaluation_episodes_preview = episodes or int(
            experiment_config["training"]["n_eval_episodes"]
        )
        LOGGER.info(
            "Evaluating random uniform policy baseline | experiment=%s episodes=%s",
            experiment_name,
            evaluation_episodes_preview,
        )
        runtime_config = resolved_config.get("runtime", {})
        effective_experiment_name = str(
            runtime_config.get("experiment_name", experiment_name)
        )
        evaluation_episodes = evaluation_episodes_preview
        evaluation_env = None
        summary_path = None
        try:
            evaluation_env = build_vector_env(
                experiment_config=experiment_config,
                num_envs=int(experiment_config["training"]["eval_num_envs"]),
                monitor=True,
                disable_reward_curriculum=True,
                vecnormalize_path=None,
                for_training=False,
                norm_reward=False,
            )
            rng = np.random.default_rng(seed)
            model = RandomPolicy(evaluation_env.action_space, rng)
            evaluation_bundle = evaluate_policy_variants(
                model=model,
                env=evaluation_env,
                n_eval_episodes=evaluation_episodes,
                experiment_config=experiment_config,
                include_stochastic=False,
                deterministic_eval=bool(
                    experiment_config["training"]["deterministic_eval"]
                ),
            )
            summary = {
                "experiment_name": effective_experiment_name,
                "eval_environment_experiment": eval_experiment_name,
                "algorithm": "random",
                "hyperparams_source": hyperparams_source,
                "model_kind": "random_uniform",
                "seed": seed,
                "train_seed": None,
                "run_id": None,
                "run_root": None,
                "model_path": None,
                "episodes": evaluation_episodes,
                "mean_reward": float(evaluation_bundle["mean_reward"]),
                "std_reward": float(evaluation_bundle["std_reward"]),
                "task_score": float(evaluation_bundle["task_score"]),
                "selection_score": float(evaluation_bundle["selection_score"]),
                "task_eligible": bool(evaluation_bundle["task_eligible"]),
                "scientific_gates_passed": bool(
                    evaluation_bundle["metrics_deterministic"].get(
                        "scientific_gates_passed", True
                    )
                ),
                "scientific_gates": evaluation_bundle["metrics_deterministic"].get(
                    "scientific_gates", {}
                ),
                "device": resolved_genesis_device,
                "genesis_device": resolved_genesis_device,
                "algorithm_device": algorithm_device,
                "runtime_device": resolved_genesis_device,
                "evaluation_protocol": build_evaluation_protocol(experiment_config),
                "metrics_deterministic": evaluation_bundle["metrics_deterministic"],
                "metrics_stochastic": evaluation_bundle.get("metrics_stochastic"),
                "summary": summarize_evaluation_bundle(evaluation_bundle),
                "runtime_context": collect_runtime_context(),
            }
            summary_path = output_root / "evaluation_summary.json"
            write_json(summary_path, to_json_compatible(summary))
        finally:
            if evaluation_env is not None:
                evaluation_env.close()

            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        if summary_path is not None:
            LOGGER.info("Evaluation finished. Summary saved to %s", summary_path)

        return summary

    if resolved_model_path is None and run_id is not None:
        selected_run_root = artifact_store.get_run_root(
            experiment_name,
            algorithm,
            run_id,
        )
    elif resolved_model_path is None and train_seed is not None:
        selected_run_root = artifact_store.find_latest_run(
            experiment_name,
            algorithm,
            hyperparams_source=hyperparams_source,
            train_seed=train_seed,
            completed_only=True,
        )
    elif resolved_model_path is None:
        selected_run_root = artifact_store.find_latest_run(
            experiment_name,
            algorithm,
            hyperparams_source=hyperparams_source,
            completed_only=True,
        )

    if resolved_model_path is None and selected_run_root is not None:
        resolved_model_path = artifact_store.get_model_path_for_run(
            selected_run_root,
            model_kind,
        )

    if resolved_model_path is None or not resolved_model_path.exists():
        raise FileNotFoundError("No model was found for evaluation.")

    run_root = selected_run_root or find_run_root_from_model_path(resolved_model_path)
    output_root: Path | None = None
    if run_root is not None:
        resolved_config = read_yaml(run_root / "config" / "resolved_config.yaml")
        log_file = run_root / "logs" / "evaluation.log"
    else:
        resolved_config = load_resolved_config(
            settings=settings,
            artifact_store=artifact_store,
            experiment_name=experiment_name,
            algorithm=algorithm,
            hyperparams_source=hyperparams_source,
        )
        output_root = artifact_store.create_ad_hoc_execution_dir(
            "evaluate",
            experiment_name,
            algorithm,
            seed,
            label=resolved_model_path.stem,
        )
        log_file = output_root / "evaluation.log"

    resolved_genesis_device = initialize_genesis(seed=seed, device=genesis_device)
    configure_logging(log_file, settings.log_level)
    set_global_seeds(seed)

    if eval_experiment_name:
        apply_eval_experiment_environment_overlay(
            settings,
            resolved_config,
            eval_experiment_name,
        )

    experiment_config = resolved_config["experiment"]
    runtime_config = resolved_config.get("runtime", {})
    effective_experiment_name = str(
        runtime_config.get("experiment_name", experiment_name)
    )
    effective_algorithm = str(runtime_config.get("algorithm", algorithm))
    effective_hyperparams_source = str(
        runtime_config.get("hyperparams_source", hyperparams_source)
    )
    parsed_run = (
        ArtifactStore.parse_run_id(run_root.name) if run_root is not None else None
    )
    evaluation_episodes = episodes or int(
        experiment_config["training"]["n_eval_episodes"]
    )
    evaluation_env = None
    summary_path = None
    try:
        evaluation_env = build_vector_env(
            experiment_config=experiment_config,
            num_envs=int(experiment_config["training"]["eval_num_envs"]),
            monitor=True,
            disable_reward_curriculum=True,
            vecnormalize_path=find_vecnormalize_path_from_model_path(
                resolved_model_path
            ),
            for_training=False,
            norm_reward=False,
        )
        model = load_sb3_model(
            algorithm=algorithm,
            model_path=resolved_model_path,
            env=evaluation_env,
            device=algorithm_device,
        )

        evaluation_bundle = evaluate_policy_variants(
            model=model,
            env=evaluation_env,
            n_eval_episodes=evaluation_episodes,
            experiment_config=experiment_config,
            include_stochastic=_include_stochastic_metrics(algorithm),
            deterministic_eval=bool(
                experiment_config["training"]["deterministic_eval"]
            ),
        )

        summary = {
            "experiment_name": effective_experiment_name,
            "eval_environment_experiment": eval_experiment_name,
            "algorithm": effective_algorithm,
            "hyperparams_source": effective_hyperparams_source,
            "model_kind": model_kind,
            "seed": seed,
            "train_seed": (
                int(parsed_run["seed"])
                if parsed_run is not None
                else (int(train_seed) if train_seed is not None else None)
            ),
            "run_id": run_root.name if run_root is not None else None,
            "run_root": str(run_root) if run_root is not None else None,
            "model_path": str(resolved_model_path),
            "episodes": evaluation_episodes,
            "mean_reward": float(evaluation_bundle["mean_reward"]),
            "std_reward": float(evaluation_bundle["std_reward"]),
            "task_score": float(evaluation_bundle["task_score"]),
            "selection_score": float(evaluation_bundle["selection_score"]),
            "task_eligible": bool(evaluation_bundle["task_eligible"]),
            "scientific_gates_passed": bool(
                evaluation_bundle["metrics_deterministic"].get(
                    "scientific_gates_passed", True
                )
            ),
            "scientific_gates": evaluation_bundle["metrics_deterministic"].get(
                "scientific_gates", {}
            ),
            "device": resolved_genesis_device,
            "genesis_device": resolved_genesis_device,
            "algorithm_device": algorithm_device,
            "runtime_device": resolved_genesis_device,
            "evaluation_protocol": build_evaluation_protocol(experiment_config),
            "metrics_deterministic": evaluation_bundle["metrics_deterministic"],
            "metrics_stochastic": evaluation_bundle.get("metrics_stochastic"),
            "summary": summarize_evaluation_bundle(evaluation_bundle),
            "runtime_context": collect_runtime_context(),
        }

        if run_root is not None:
            if model_path:
                summary_name = f"evaluation_explicit_{ArtifactStore.sanitize_label(resolved_model_path.stem) or 'model'}.json"
            else:
                summary_name = f"evaluation_{model_kind}.json"
            summary_path = run_root / "eval" / summary_name
        else:
            assert output_root is not None
            summary_path = output_root / "evaluation_summary.json"
        write_json(summary_path, to_json_compatible(summary))
    finally:
        if evaluation_env is not None:
            evaluation_env.close()

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    if summary_path is not None:
        LOGGER.info("Evaluation finished. Summary saved to %s", summary_path)

    return summary
