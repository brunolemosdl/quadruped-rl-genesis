"""Optuna hyperparameter search over nested config overrides."""

from __future__ import annotations

import copy
import gc
from pathlib import Path
from typing import Any

import optuna
from optuna.trial import TrialState
from stable_baselines3.common.vec_env import sync_envs_normalization

from quadruped_rl_genesis.algorithms.factory import create_sb3_model, load_sb3_model
from quadruped_rl_genesis.algorithms.search import (
    build_trial_environment_overrides,
    build_trial_parameter_overrides,
)
from quadruped_rl_genesis.config.loader import (
    load_optuna_config,
    load_resolved_config,
)
from quadruped_rl_genesis.environments.factory import build_vector_env
from quadruped_rl_genesis.pipeline.callbacks import TrialProgressTaskEvalCallback
from quadruped_rl_genesis.pipeline.tunecore import (
    filtered_search_space_payload,
    include_stochastic_metrics,
    study_trial_counts,
    write_trial_status,
)
from quadruped_rl_genesis.pipeline.tunecore import (
    phase_plan as build_phase_plan,
)
from quadruped_rl_genesis.services.artifacts import (
    ArtifactStore,
    find_vecnormalize_path_from_model_path,
)
from quadruped_rl_genesis.services.logger import configure_logging, get_logger
from quadruped_rl_genesis.services.metrics import (
    build_evaluation_protocol,
    evaluate_policy_variants,
)
from quadruped_rl_genesis.services.platform import collect_runtime_context
from quadruped_rl_genesis.services.runtime import (
    initialize_genesis,
    resolve_runtime_device,
    shutdown_genesis,
)
from quadruped_rl_genesis.settings import AppSettings
from quadruped_rl_genesis.utils.io import (
    deep_merge,
    to_json_compatible,
    write_json,
    write_yaml,
)
from quadruped_rl_genesis.utils.seed import set_global_seeds

LOGGER = get_logger(__name__)


def run_tuning(
    settings: AppSettings,
    experiment_name: str,
    algorithm: str,
    seed: int,
    genesis_device: str,
    algorithm_device: str,
    base_hyperparams_source: str = "default",
) -> dict[str, Any]:
    """Run an Optuna study and persist the best parameter set.

    Args:
        settings (AppSettings): Global application settings.
        experiment_name (str): Experiment profile name.
        algorithm (str): Algorithm name.
        seed (int): Base Optuna seed.
        genesis_device (str): Device for Genesis simulation.
        algorithm_device (str): Device for SB3/RL algorithm.
        base_hyperparams_source (str, optional): Named base hyperparameter
            source used before Optuna trial overrides are applied.

    Returns:
        dict[str, Any]: Study summary persisted to disk.
    """
    if base_hyperparams_source == "optuna":
        raise ValueError(
            "Optuna tuning cannot use 'optuna' as its base hyperparameter source. "
            "Use 'default' or a named base profile such as 'utd_gs2'."
        )

    artifact_store = ArtifactStore(settings)
    base_resolved_config = load_resolved_config(
        settings=settings,
        artifact_store=artifact_store,
        experiment_name=experiment_name,
        algorithm=algorithm,
        hyperparams_source=base_hyperparams_source,
    )
    base_experiment_config = base_resolved_config["experiment"]
    base_algorithm_config = base_resolved_config["algorithm"]
    optuna_config = load_optuna_config(settings)["optuna"]
    optuna_root = artifact_store.get_optuna_dir(experiment_name, algorithm)

    resolved_genesis_device = resolve_runtime_device(genesis_device)
    resolved_algorithm_device = resolve_runtime_device(algorithm_device)
    configure_logging(optuna_root / "optuna.log", settings.log_level)
    set_global_seeds(seed)

    LOGGER.info(
        "Starting Optuna study | experiment=%s algorithm=%s base_hyperparams_source=%s seed=%s genesis=%s algorithm=%s",
        experiment_name,
        algorithm,
        base_hyperparams_source,
        seed,
        resolved_genesis_device,
        resolved_algorithm_device,
    )
    if settings.optuna_storage.startswith("sqlite:///"):
        sqlite_path = Path(settings.optuna_storage.replace("sqlite:///", "", 1))
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    phase_plan = build_phase_plan(optuna_config)
    multi_phase = len(phase_plan) > 1
    current_algorithm_config = copy.deepcopy(base_algorithm_config)
    current_experiment_config = copy.deepcopy(base_experiment_config)
    cumulative_algorithm_overrides: dict[str, Any] = {}
    cumulative_experiment_environment_overrides: dict[str, Any] = {}
    phase_summaries: list[dict[str, Any]] = []

    for phase_index, phase in enumerate(phase_plan):
        phase_name = str(phase["name"])
        trial_status_path = artifact_store.get_optuna_phase_current_trial_path(
            experiment_name,
            algorithm,
            phase_name,
        )
        phase_algorithm_payload = filtered_search_space_payload(
            current_algorithm_config,
            phase_name=phase_name if multi_phase else None,
        )
        phase_experiment_payload = filtered_search_space_payload(
            current_experiment_config,
            phase_name=phase_name if multi_phase else None,
        )

        study_name = (
            f"{experiment_name}_{algorithm}_{base_hyperparams_source}_{phase_name}"
            if multi_phase
            else f"{experiment_name}_{algorithm}_{base_hyperparams_source}"
        )
        sampler = optuna.samplers.TPESampler(
            seed=int(optuna_config["sampler_seed"]) + phase_index
        )
        study = optuna.create_study(
            study_name=study_name,
            storage=settings.optuna_storage,
            direction=str(optuna_config["study_direction"]),
            load_if_exists=True,
            sampler=sampler,
        )
        total_trials = int(phase["n_trials"])
        trial_counts = study_trial_counts(study)
        completed_trials = int(trial_counts.get(TrialState.COMPLETE.name, 0))
        remaining_trials = max(total_trials - completed_trials, 0)

        if trial_counts.get(TrialState.RUNNING.name, 0):
            LOGGER.warning(
                (
                    "Study contains %s pre-existing RUNNING trials. "
                    "It will continue until %s COMPLETE trials in phase '%s'."
                ),
                trial_counts[TrialState.RUNNING.name],
                total_trials,
                phase_name,
            )

        def objective(trial: optuna.Trial) -> float:
            """Train, evaluate, and return the scalar objective for one Optuna trial.

            Args:
                trial (optuna.Trial): Active trial providing sampled hyperparameters.

            Returns:
                float: Objective value passed to Optuna (higher is better when
                    ``study_direction`` is ``maximize``).
            """
            trial_seed = seed + trial.number
            set_global_seeds(trial_seed)
            initialize_genesis(seed=trial_seed, device=resolved_genesis_device)
            completed_before_trial = len(
                study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
            )
            target_index = min(completed_before_trial + 1, total_trials)
            trial_dir = artifact_store.get_optuna_trial_dir(
                experiment_name,
                algorithm,
                phase_name,
                trial.number,
            )

            algorithm_overrides = build_trial_parameter_overrides(
                trial,
                phase_algorithm_payload,
            )
            environment_overrides = build_trial_environment_overrides(
                trial,
                phase_experiment_payload,
            )
            trial_overrides = {
                "algorithm": algorithm_overrides.get("algorithm", {}),
                "experiment": {"environment": environment_overrides},
            }
            trial_algorithm_config = deep_merge(
                copy.deepcopy(current_algorithm_config),
                trial_overrides["algorithm"],
            )
            trial_experiment_config = copy.deepcopy(current_experiment_config)
            trial_experiment_config["environment"] = deep_merge(
                trial_experiment_config["environment"],
                environment_overrides,
            )
            resolved_config = {
                "experiment": trial_experiment_config,
                "algorithm": trial_algorithm_config,
                "runtime": {
                    "experiment_name": experiment_name,
                    "algorithm": algorithm,
                    "base_hyperparams_source": base_hyperparams_source,
                    "hyperparams_source": "optuna_trial",
                    "tuning_phase": phase_name,
                },
            }

            write_yaml(
                artifact_store.get_optuna_trial_overrides_path(
                    experiment_name,
                    algorithm,
                    phase_name,
                    trial.number,
                ),
                to_json_compatible(trial_overrides),
            )
            write_yaml(
                artifact_store.get_optuna_trial_resolved_config_path(
                    experiment_name,
                    algorithm,
                    phase_name,
                    trial.number,
                ),
                to_json_compatible(resolved_config),
            )
            LOGGER.info(
                (
                    "Starting trial %s/%s (optuna #%s) | phase=%s experiment=%s "
                    "algorithm=%s trial_seed=%s trial_dir=%s overrides=%s"
                ),
                target_index,
                total_trials,
                trial.number,
                phase_name,
                experiment_name,
                algorithm,
                trial_seed,
                trial_dir,
                trial_overrides,
            )
            write_trial_status(
                trial_status_path,
                {
                    "status": "starting",
                    "phase": phase_name,
                    "experiment_name": experiment_name,
                    "algorithm": algorithm,
                    "base_hyperparams_source": base_hyperparams_source,
                    "trial_number": trial.number,
                    "target_index": target_index,
                    "trial_index": trial.number + 1,
                    "total_trials": total_trials,
                    "trial_seed": trial_seed,
                    "trial_dir": str(trial_dir),
                    "overrides": trial_overrides,
                },
            )
            training_env = None
            evaluation_env = None

            try:
                training_env = build_vector_env(
                    experiment_config=trial_experiment_config,
                    num_envs=int(trial_experiment_config["training"]["train_num_envs"]),
                    monitor=True,
                    for_training=True,
                )
                evaluation_env = build_vector_env(
                    experiment_config=trial_experiment_config,
                    num_envs=int(trial_experiment_config["training"]["eval_num_envs"]),
                    monitor=True,
                    for_training=False,
                    norm_reward=False,
                )

                model = create_sb3_model(
                    algorithm_config_payload=trial_algorithm_config,
                    env=training_env,
                    tensorboard_log=trial_dir / "tensorboard",
                    device=resolved_algorithm_device,
                )

                def _status_writer(payload: dict[str, Any]) -> None:
                    write_trial_status(
                        trial_status_path,
                        {
                            "base_hyperparams_source": base_hyperparams_source,
                            "phase": phase_name,
                            **payload,
                        },
                    )

                eval_callback = TrialProgressTaskEvalCallback(
                    logger=LOGGER,
                    experiment_name=experiment_name,
                    algorithm=algorithm,
                    trial_number=trial.number,
                    target_index=target_index,
                    total_trials=total_trials,
                    status_writer=_status_writer,
                    eval_env=evaluation_env,
                    experiment_config=trial_experiment_config,
                    eval_freq=max(
                        int(phase["eval_freq_env_steps"])
                        // int(trial_experiment_config["training"]["train_num_envs"]),
                        1,
                    ),
                    n_eval_episodes=int(phase["eval_episodes"]),
                    deterministic_eval=bool(
                        trial_experiment_config["training"]["deterministic_eval"]
                    ),
                    include_stochastic_eval=include_stochastic_metrics(algorithm),
                    best_model_save_path=trial_dir / "best_model",
                    log_dir=trial_dir / "eval",
                    verbose=0,
                )
                eval_callback.record_initial_evaluation(model)

                model.learn(
                    total_timesteps=int(phase["trial_timesteps"]),
                    callback=eval_callback,
                    progress_bar=False,
                )

                final_model_path = trial_dir / "final_model.zip"
                model.save(str(final_model_path))
                vec_normalize_env = model.get_vec_normalize_env()
                if vec_normalize_env is not None:
                    vec_normalize_env.save(str(trial_dir / "vecnormalize.pkl"))
                best_model_path = trial_dir / "best_model" / "best_model.zip"
                selected_model_path = (
                    best_model_path if best_model_path.exists() else final_model_path
                )
                if vec_normalize_env is not None:
                    try:
                        sync_envs_normalization(training_env, evaluation_env)
                    except Exception:
                        pass
                selected_model = load_sb3_model(
                    algorithm=algorithm,
                    model_path=selected_model_path,
                    env=evaluation_env,
                    device=resolved_algorithm_device,
                )
                evaluation_bundle = evaluate_policy_variants(
                    model=selected_model,
                    env=evaluation_env,
                    n_eval_episodes=int(phase["eval_episodes"]),
                    experiment_config=trial_experiment_config,
                    include_stochastic=include_stochastic_metrics(algorithm),
                    deterministic_eval=bool(
                        trial_experiment_config["training"]["deterministic_eval"]
                    ),
                )
                deterministic_metrics = evaluation_bundle["metrics_deterministic"]
                result_payload = {
                    "phase": phase_name,
                    "trial_number": trial.number,
                    "seed": trial_seed,
                    "base_hyperparams_source": base_hyperparams_source,
                    "selected_model_path": str(selected_model_path),
                    "vecnormalize_path": str(
                        find_vecnormalize_path_from_model_path(selected_model_path)
                    )
                    if find_vecnormalize_path_from_model_path(selected_model_path)
                    is not None
                    else None,
                    "mean_reward": float(evaluation_bundle["mean_reward"]),
                    "task_score": float(evaluation_bundle["task_score"]),
                    "selection_score": float(evaluation_bundle["selection_score"]),
                    "task_eligible": bool(evaluation_bundle["task_eligible"]),
                    "evaluation_protocol": build_evaluation_protocol(
                        trial_experiment_config
                    ),
                    "metrics_deterministic": deterministic_metrics,
                    "metrics_stochastic": evaluation_bundle.get("metrics_stochastic"),
                    "overrides": trial_overrides,
                    "runtime_context": collect_runtime_context(),
                }
                write_json(
                    artifact_store.get_optuna_trial_summary_path(
                        experiment_name,
                        algorithm,
                        phase_name,
                        trial.number,
                    ),
                    to_json_compatible(result_payload),
                )

                trial.set_user_attr(
                    "nested_overrides", to_json_compatible(trial_overrides)
                )
                trial.set_user_attr("phase_name", phase_name)
                trial.set_user_attr(
                    "mean_reward", float(evaluation_bundle["mean_reward"])
                )
                trial.set_user_attr(
                    "task_score", float(evaluation_bundle["task_score"])
                )
                trial.set_user_attr(
                    "selection_score", float(evaluation_bundle["selection_score"])
                )
                trial.set_user_attr(
                    "task_eligible", bool(evaluation_bundle["task_eligible"])
                )
                trial.set_user_attr(
                    "timeout_rate",
                    float(
                        deterministic_metrics["article_metrics"].get(
                            "timeout_rate", 0.0
                        )
                    ),
                )
                trial.set_user_attr(
                    "fall_rate",
                    float(
                        deterministic_metrics["article_metrics"].get("fall_rate", 0.0)
                    ),
                )
                trial.set_user_attr(
                    "mean_arc_progress_speed",
                    float(
                        deterministic_metrics["article_metrics"].get(
                            "mean_arc_progress_speed", 0.0
                        )
                    ),
                )
                trial.set_user_attr(
                    "mean_cross_track_error",
                    float(
                        deterministic_metrics["article_metrics"].get(
                            "mean_cross_track_error", 0.0
                        )
                    ),
                )
                write_trial_status(
                    trial_status_path,
                    {
                        "status": "completed",
                        "phase": phase_name,
                        "experiment_name": experiment_name,
                        "algorithm": algorithm,
                        "base_hyperparams_source": base_hyperparams_source,
                        "trial_number": trial.number,
                        "target_index": target_index,
                        "trial_index": trial.number + 1,
                        "total_trials": total_trials,
                        "trial_seed": trial_seed,
                        "trial_dir": str(trial_dir),
                        "mean_reward": float(evaluation_bundle["mean_reward"]),
                        "task_score": float(evaluation_bundle["task_score"]),
                        "selection_score": float(evaluation_bundle["selection_score"]),
                        "selected_model_path": str(selected_model_path),
                    },
                )
                return float(evaluation_bundle["selection_score"])
            except Exception as exc:
                LOGGER.exception(
                    "Failed trial %s/%s (optuna #%s) | phase=%s experiment=%s algorithm=%s",
                    target_index,
                    total_trials,
                    trial.number,
                    phase_name,
                    experiment_name,
                    algorithm,
                )
                write_trial_status(
                    trial_status_path,
                    {
                        "status": "failed",
                        "phase": phase_name,
                        "experiment_name": experiment_name,
                        "algorithm": algorithm,
                        "base_hyperparams_source": base_hyperparams_source,
                        "trial_number": trial.number,
                        "target_index": target_index,
                        "trial_index": trial.number + 1,
                        "total_trials": total_trials,
                        "trial_seed": trial_seed,
                        "trial_dir": str(trial_dir),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    },
                )
                raise
            finally:
                if training_env is not None:
                    training_env.close()
                if evaluation_env is not None:
                    evaluation_env.close()
                shutdown_genesis()
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

        if remaining_trials > 0:
            LOGGER.info(
                "Resuming Optuna study | phase=%s experiment=%s algorithm=%s base_hyperparams_source=%s completed=%s target=%s remaining=%s",
                phase_name,
                experiment_name,
                algorithm,
                base_hyperparams_source,
                completed_trials,
                total_trials,
                remaining_trials,
            )
            study.optimize(
                objective,
                n_trials=remaining_trials,
                timeout=phase.get("timeout_seconds"),
                gc_after_trial=True,
                show_progress_bar=False,
            )
        else:
            LOGGER.info(
                "Optuna study already satisfies target trial count | phase=%s experiment=%s algorithm=%s completed=%s target=%s",
                phase_name,
                experiment_name,
                algorithm,
                completed_trials,
                total_trials,
            )

        best_trial = study.best_trial
        best_nested_overrides = dict(best_trial.user_attrs.get("nested_overrides", {}))
        phase_algorithm_overrides = dict(best_nested_overrides.get("algorithm", {}))
        phase_experiment_overrides = dict(best_nested_overrides.get("experiment", {}))
        phase_environment_overrides = dict(
            phase_experiment_overrides.get("environment", {})
        )

        cumulative_algorithm_overrides = deep_merge(
            cumulative_algorithm_overrides,
            phase_algorithm_overrides,
        )
        cumulative_experiment_environment_overrides = deep_merge(
            cumulative_experiment_environment_overrides,
            phase_environment_overrides,
        )
        current_algorithm_config = deep_merge(
            current_algorithm_config,
            phase_algorithm_overrides,
        )
        current_experiment_config["environment"] = deep_merge(
            current_experiment_config["environment"],
            phase_environment_overrides,
        )

        final_trial_counts = study_trial_counts(study)
        phase_summary = {
            "phase_name": phase_name,
            "study_name": study_name,
            "best_trial_number": best_trial.number,
            "best_value": best_trial.value,
            "best_task_score": best_trial.user_attrs.get("task_score"),
            "best_selection_score": best_trial.user_attrs.get("selection_score"),
            "best_mean_reward": best_trial.user_attrs.get("mean_reward"),
            "best_trial_timeout_rate": best_trial.user_attrs.get("timeout_rate"),
            "best_trial_fall_rate": best_trial.user_attrs.get("fall_rate"),
            "best_trial_mean_arc_progress_speed": best_trial.user_attrs.get(
                "mean_arc_progress_speed"
            ),
            "best_trial_mean_cross_track_error": best_trial.user_attrs.get(
                "mean_cross_track_error"
            ),
            "phase_overrides": best_nested_overrides,
            "n_trials": len(study.trials),
            "completed_trials": int(
                final_trial_counts.get(TrialState.COMPLETE.name, 0)
            ),
            "failed_trials": int(final_trial_counts.get(TrialState.FAIL.name, 0)),
            "running_trials": int(final_trial_counts.get(TrialState.RUNNING.name, 0)),
            "target_trials": total_trials,
            "trial_timesteps": int(phase["trial_timesteps"]),
            "eval_episodes": int(phase["eval_episodes"]),
            "eval_freq_env_steps": int(phase["eval_freq_env_steps"]),
        }
        phase_summaries.append(phase_summary)
        write_json(
            artifact_store.get_optuna_phase_study_summary_path(
                experiment_name,
                algorithm,
                phase_name,
            ),
            to_json_compatible(phase_summary),
        )

    best_overrides_payload = {
        "base_hyperparams_source": base_hyperparams_source,
        "algorithm": cumulative_algorithm_overrides,
        "experiment": {"environment": cumulative_experiment_environment_overrides},
        "tuning_phases": [phase["name"] for phase in phase_plan],
    }
    best_overrides_path = artifact_store.get_optuna_best_overrides_path(
        experiment_name,
        algorithm,
    )
    write_yaml(best_overrides_path, to_json_compatible(best_overrides_payload))
    best_resolved_config = {
        "experiment": current_experiment_config,
        "algorithm": current_algorithm_config,
        "runtime": {
            "experiment_name": experiment_name,
            "algorithm": algorithm,
            "base_hyperparams_source": base_hyperparams_source,
            "hyperparams_source": "optuna",
            "tuning_phases": [phase["name"] for phase in phase_plan],
        },
    }
    best_resolved_config_path = artifact_store.get_optuna_best_resolved_config_path(
        experiment_name,
        algorithm,
    )
    write_yaml(
        best_resolved_config_path,
        to_json_compatible(best_resolved_config),
    )
    best_trial_summary = {
        "experiment_name": experiment_name,
        "algorithm": algorithm,
        "base_hyperparams_source": base_hyperparams_source,
        "tuning_phases": [phase["name"] for phase in phase_plan],
        "best_phase": phase_summaries[-1]["phase_name"] if phase_summaries else None,
        "best_trial_number": (
            phase_summaries[-1]["best_trial_number"] if phase_summaries else None
        ),
        "best_value": phase_summaries[-1]["best_value"] if phase_summaries else None,
        "best_task_score": (
            phase_summaries[-1]["best_task_score"] if phase_summaries else None
        ),
        "best_selection_score": (
            phase_summaries[-1]["best_selection_score"] if phase_summaries else None
        ),
        "best_mean_reward": (
            phase_summaries[-1]["best_mean_reward"] if phase_summaries else None
        ),
        "phase_summaries": phase_summaries,
        "best_overrides_path": str(best_overrides_path),
        "best_resolved_config_path": str(best_resolved_config_path),
    }
    best_trial_summary_path = artifact_store.get_optuna_best_trial_summary_path(
        experiment_name,
        algorithm,
    )
    write_json(best_trial_summary_path, to_json_compatible(best_trial_summary))

    final_summary = {
        "study_name": phase_summaries[-1]["study_name"] if phase_summaries else None,
        "phase_summaries": phase_summaries,
        "best_trial_number": (
            phase_summaries[-1]["best_trial_number"] if phase_summaries else None
        ),
        "best_value": phase_summaries[-1]["best_value"] if phase_summaries else None,
        "best_task_score": (
            phase_summaries[-1]["best_task_score"] if phase_summaries else None
        ),
        "best_selection_score": (
            phase_summaries[-1]["best_selection_score"] if phase_summaries else None
        ),
        "best_mean_reward": (
            phase_summaries[-1]["best_mean_reward"] if phase_summaries else None
        ),
        "best_trial_summary_path": str(best_trial_summary_path),
        "best_overrides_path": str(best_overrides_path),
        "best_resolved_config_path": str(best_resolved_config_path),
        "base_hyperparams_source": base_hyperparams_source,
        "storage": settings.optuna_storage,
        "n_trials": int(sum(phase["n_trials"] for phase in phase_plan)),
        "completed_trials": int(
            sum(phase_summary["completed_trials"] for phase_summary in phase_summaries)
        ),
        "failed_trials": int(
            sum(phase_summary["failed_trials"] for phase_summary in phase_summaries)
        ),
        "running_trials": int(
            sum(phase_summary["running_trials"] for phase_summary in phase_summaries)
        ),
        "target_trials": int(sum(phase["n_trials"] for phase in phase_plan)),
        "genesis_device": resolved_genesis_device,
        "algorithm_device": resolved_algorithm_device,
        "runtime_device": resolved_genesis_device,
        "runtime_context": collect_runtime_context(),
    }
    summary_path = artifact_store.get_optuna_study_summary_path(
        experiment_name,
        algorithm,
    )
    write_json(summary_path, to_json_compatible(final_summary))
    LOGGER.info(
        "Optuna study finished. Best overrides saved to %s",
        best_overrides_path,
    )

    return final_summary
