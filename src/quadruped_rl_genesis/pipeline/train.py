"""Train SB3 policies from resolved experiment configuration."""

from __future__ import annotations

import gc
import shutil
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import CheckpointCallback

from quadruped_rl_genesis.algorithms.factory import create_sb3_model, load_sb3_model
from quadruped_rl_genesis.config.loader import load_resolved_config
from quadruped_rl_genesis.environments.factory import build_vector_env
from quadruped_rl_genesis.pipeline.callbacks import TaskEvalCallback
from quadruped_rl_genesis.services.artifacts import (
    ArtifactStore,
    find_vecnormalize_path_from_model_path,
)
from quadruped_rl_genesis.services.logger import configure_logging, get_logger
from quadruped_rl_genesis.services.metrics import (
    build_evaluation_protocol,
    evaluate_policy_variants,
    summarize_evaluation_bundle,
)
from quadruped_rl_genesis.services.platform import collect_runtime_context
from quadruped_rl_genesis.services.runtime import initialize_genesis
from quadruped_rl_genesis.settings import AppSettings
from quadruped_rl_genesis.utils.io import read_json, to_json_compatible, write_json
from quadruped_rl_genesis.utils.seed import set_global_seeds

LOGGER = get_logger(__name__)


def _callback_frequency(target_env_steps: int, num_envs: int) -> int:
    """Convert environment-step frequency into SB3 callback frequency.

    Args:
        target_env_steps (int): Target frequency expressed in environment steps.
        num_envs (int): Number of parallel training environments.

    Returns:
        int: Callback frequency in SB3 calls, clamped to at least ``1``.
    """
    return max(target_env_steps // max(num_envs, 1), 1)


def _include_stochastic_metrics(algorithm: str) -> bool:
    """Check whether an algorithm should also be evaluated stochastically.

    Args:
        algorithm (str): Algorithm name.

    Returns:
        bool: ``True`` for off-policy algorithms that benefit from stochastic
            evaluation.
    """
    return algorithm.lower() in {"sac", "td3"}


def _find_completed_training_run(
    *,
    settings: AppSettings,
    experiment_name: str,
    algorithm: str,
    seed: int,
    hyperparams_source: str,
) -> Path | None:
    """Find the most recent completed run matching the requested identifiers.

    A run is considered completed when either ``eval/training_summary.json`` or
    ``models/final_model.zip`` exists.
    """
    algo_dir = settings.artifacts_root / "experiments" / experiment_name / algorithm
    if not algo_dir.exists():
        return None

    for run_dir in sorted(algo_dir.iterdir(), key=lambda path: path.name, reverse=True):
        if not run_dir.is_dir():
            continue

        parsed_run_id = ArtifactStore.parse_run_id(run_dir.name)
        if parsed_run_id is None:
            continue
        if parsed_run_id["seed"] != int(seed):
            continue
        if parsed_run_id["hyperparams_source"] != hyperparams_source:
            continue

        summary_path = run_dir / "eval" / "training_summary.json"
        final_model_path = run_dir / "models" / "final_model.zip"
        if summary_path.exists() or final_model_path.exists():
            return run_dir

    return None


def _build_callbacks(
    *,
    experiment_config: dict[str, Any],
    run_root: Path,
    num_envs: int,
    eval_env,
    algorithm: str,
) -> tuple[list[Any], TaskEvalCallback]:
    """Create the training callbacks used for checkpoints and task-aware eval.

    Args:
        experiment_config (dict[str, Any]): Resolved experiment configuration.
        run_root (Path): Root directory of the current run.
        num_envs (int): Number of parallel training environments.
        eval_env: Evaluation environment shared by the callback.
        algorithm (str): Algorithm name.

    Returns:
        tuple[list[Any], TaskEvalCallback]: Callback list passed to SB3 and the
            task-aware evaluation callback instance.
    """
    training_config = experiment_config["training"]
    callbacks: list[Any] = [
        CheckpointCallback(
            save_freq=_callback_frequency(
                training_config["checkpoint_freq_env_steps"], num_envs
            ),
            save_path=str(run_root / "checkpoints"),
            name_prefix="model",
            save_replay_buffer=False,
            save_vecnormalize=True,
        )
    ]

    early_stopping_config = training_config.get("early_stopping", {})

    eval_callback = TaskEvalCallback(
        eval_env=eval_env,
        experiment_config=experiment_config,
        algorithm=algorithm,
        eval_freq=_callback_frequency(training_config["eval_freq_env_steps"], num_envs),
        n_eval_episodes=int(training_config["n_eval_episodes"]),
        deterministic_eval=bool(training_config["deterministic_eval"]),
        include_stochastic_eval=_include_stochastic_metrics(algorithm),
        best_model_save_path=run_root / "models" / "best_model",
        log_dir=run_root / "eval",
        max_no_improvement_evals=(
            int(early_stopping_config["max_no_improvement_evals"])
            if early_stopping_config.get("enabled", False)
            else None
        ),
        min_evals=(
            int(early_stopping_config["min_evals"])
            if early_stopping_config.get("enabled", False)
            else 0
        ),
        verbose=int(training_config.get("verbose", 1)),
    )
    callbacks.append(eval_callback)

    return callbacks, eval_callback


def _checkpoint_step(checkpoint_path: Path) -> int | None:
    """Extract the environment-step count encoded in a checkpoint filename.

    Args:
        checkpoint_path (Path): Checkpoint path following the SB3 naming
            convention.

    Returns:
        int | None: Parsed step count, or ``None`` when the filename does not
            match the expected pattern.
    """
    name = checkpoint_path.name
    prefix = "model_"
    suffix = "_steps.zip"

    if not (name.startswith(prefix) and name.endswith(suffix)):
        return None

    step_text = name.removeprefix(prefix).removesuffix(suffix)

    try:
        return int(step_text)
    except ValueError:
        return None


def _copy_model(source_path: Path, target_path: Path) -> Path:
    """Copy a saved model into a canonical milestone location.

    Args:
        source_path (Path): Existing model file.
        target_path (Path): Canonical milestone destination.

    Returns:
        Path: Destination path after the copy.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)

    return target_path


def _create_milestone_manifest(
    model,
    milestones_dir: Path,
) -> dict[str, dict[str, Any] | None]:
    """Save the initial model and initialize the milestone manifest.

    Args:
        model: Fresh SB3 model before training.
        milestones_dir (Path): Directory that will store milestone copies.

    Returns:
        dict[str, dict[str, Any] | None]: Milestone manifest with the initial
            checkpoint saved; ``middle`` and ``final`` remain ``None`` until set
            during training.
    """
    milestones_dir.mkdir(parents=True, exist_ok=True)
    initial_model_path = milestones_dir / "initial_model.zip"
    model.save(str(initial_model_path))

    return {
        "initial": {
            "path": str(initial_model_path),
            "timesteps": 0,
        },
        "middle": None,
        "final": None,
    }


def _finalize_milestone_manifest(
    *,
    milestone_models: dict[str, dict[str, Any] | None],
    milestones_dir: Path,
    checkpoints_dir: Path,
    final_model_path: Path,
    total_timesteps: int,
) -> dict[str, dict[str, Any] | None]:
    """Attach the middle and final milestone models after training finishes.

    Args:
        milestone_models (dict[str, dict[str, Any] | None]): Existing milestone
            manifest containing the initial model.
        milestones_dir (Path): Directory used to store milestone copies.
        checkpoints_dir (Path): Directory containing periodic checkpoints.
        final_model_path (Path): Path to the saved final model.
        total_timesteps (int): Total number of timesteps requested for training.

    Returns:
        dict[str, dict[str, Any] | None]: Updated milestone manifest.
    """
    middle_target_step = max(total_timesteps // 2, 1)
    checkpoint_candidates: list[tuple[int, Path]] = []

    for checkpoint_path in checkpoints_dir.glob("model_*_steps.zip"):
        step = _checkpoint_step(checkpoint_path)
        if step is None:
            continue

        checkpoint_candidates.append((step, checkpoint_path))

    if checkpoint_candidates:
        middle_step, middle_source_path = min(
            checkpoint_candidates,
            key=lambda item: (abs(item[0] - middle_target_step), item[0]),
        )
        middle_model_path = _copy_model(
            middle_source_path, milestones_dir / "middle_model.zip"
        )
        milestone_models["middle"] = {
            "path": str(middle_model_path),
            "source_path": str(middle_source_path),
            "timesteps": int(middle_step),
            "target_timesteps": int(middle_target_step),
        }

    final_milestone_path = _copy_model(
        final_model_path, milestones_dir / "final_model.zip"
    )
    milestone_models["final"] = {
        "path": str(final_milestone_path),
        "source_path": str(final_model_path),
        "timesteps": int(total_timesteps),
    }

    return milestone_models


def _evaluate_saved_model(
    *,
    algorithm: str,
    model_path: Path,
    experiment_config: dict[str, Any],
    model_device: str,
    episodes: int,
) -> dict[str, Any]:
    """Evaluate a saved model with deterministic and optional stochastic metrics.

    Args:
        algorithm (str): Algorithm name.
        model_path (Path): Saved model path.
        experiment_config (dict[str, Any]): Resolved experiment configuration.
        model_device (str): Runtime device used to load the model.
        episodes (int): Number of evaluation episodes.

    Returns:
        dict[str, Any]: Evaluation bundle returned by
            ``evaluate_policy_variants``.
    """
    eval_env = build_vector_env(
        experiment_config=experiment_config,
        num_envs=int(experiment_config["training"]["eval_num_envs"]),
        monitor=True,
        vecnormalize_path=find_vecnormalize_path_from_model_path(model_path),
        for_training=False,
        norm_reward=False,
    )
    model = load_sb3_model(
        algorithm=algorithm,
        model_path=model_path,
        env=eval_env,
        device=model_device,
    )

    try:
        return evaluate_policy_variants(
            model=model,
            env=eval_env,
            n_eval_episodes=episodes,
            experiment_config=experiment_config,
            include_stochastic=_include_stochastic_metrics(algorithm),
            deterministic_eval=bool(
                experiment_config["training"]["deterministic_eval"]
            ),
        )
    finally:
        eval_env.close()


def run_training(
    settings: AppSettings,
    experiment_name: str,
    algorithm: str,
    hyperparams_source: str,
    seed: int,
    genesis_device: str,
    algorithm_device: str,
    resume: bool = False,
) -> dict[str, Any]:
    """Run the full training workflow for one experiment and algorithm.

    Args:
        settings (AppSettings): Global application settings.
        experiment_name (str): Experiment profile name.
        algorithm (str): Algorithm name.
        hyperparams_source (str): Hyperparameter source label.
        seed (int): Training seed.
        genesis_device (str): Device for Genesis simulation.
        algorithm_device (str): Device for SB3/RL algorithm.
        resume (bool, optional): When ``True``, skip execution if a completed run
            already exists for the same experiment, algorithm, seed, and
            hyperparameter source.

    Returns:
        dict[str, Any]: Training summary persisted to disk.
    """
    artifact_store = ArtifactStore(settings)
    if resume:
        completed_run = _find_completed_training_run(
            settings=settings,
            experiment_name=experiment_name,
            algorithm=algorithm,
            seed=seed,
            hyperparams_source=hyperparams_source,
        )
        if completed_run is not None:
            summary_path = completed_run / "eval" / "training_summary.json"
            if summary_path.exists():
                LOGGER.info(
                    "Resuming train by skipping completed run | experiment=%s algorithm=%s seed=%s run_id=%s",
                    experiment_name,
                    algorithm,
                    seed,
                    completed_run.name,
                )
                return read_json(summary_path)

            final_model_path = completed_run / "models" / "final_model.zip"
            best_model_path = completed_run / "models" / "best_model" / "best_model.zip"
            LOGGER.info(
                "Resuming train by reusing completed model artifacts | experiment=%s algorithm=%s seed=%s run_id=%s",
                experiment_name,
                algorithm,
                seed,
                completed_run.name,
            )
            return {
                "experiment_name": experiment_name,
                "algorithm": algorithm,
                "hyperparams_source": hyperparams_source,
                "seed": int(seed),
                "run_id": completed_run.name,
                "run_root": str(completed_run),
                "final_model_path": str(final_model_path),
                "best_model_path": (
                    str(best_model_path) if best_model_path.exists() else None
                ),
                "resumed": True,
            }

    resolved_config = load_resolved_config(
        settings=settings,
        artifact_store=artifact_store,
        experiment_name=experiment_name,
        algorithm=algorithm,
        hyperparams_source=hyperparams_source,
    )

    resolved_genesis_device = initialize_genesis(seed=seed, device=genesis_device)
    run_artifacts = artifact_store.create_run(
        experiment_name=experiment_name,
        algorithm=algorithm,
        seed=seed,
        hyperparams_source=hyperparams_source,
    )
    configure_logging(run_artifacts.logs_dir / "train.log", settings.log_level)
    set_global_seeds(seed)

    artifact_store.save_resolved_config(run_artifacts, resolved_config)
    artifact_store.save_runtime_metadata(
        run_artifacts,
        {
            "seed": seed,
            "device": resolved_genesis_device,
            "genesis_device": resolved_genesis_device,
            "algorithm_device": algorithm_device,
            "runtime_device": resolved_genesis_device,
            "experiment_name": experiment_name,
            "algorithm": algorithm,
            "hyperparams_source": hyperparams_source,
            "runtime_context": collect_runtime_context(),
        },
    )

    experiment_config = resolved_config["experiment"]
    training_config = experiment_config["training"]
    LOGGER.info(
        "Starting training | experiment=%s algorithm=%s hyperparams=%s seed=%s genesis=%s algorithm=%s",
        experiment_name,
        algorithm,
        hyperparams_source,
        seed,
        resolved_genesis_device,
        algorithm_device,
    )
    training_env = None
    evaluation_env = None
    summary_path = None

    try:
        training_env = build_vector_env(
            experiment_config=experiment_config,
            num_envs=int(training_config["train_num_envs"]),
            monitor=True,
            for_training=True,
        )
        evaluation_env = build_vector_env(
            experiment_config=experiment_config,
            num_envs=int(training_config["eval_num_envs"]),
            monitor=True,
            for_training=False,
            norm_reward=False,
        )

        model = create_sb3_model(
            algorithm_config_payload=resolved_config["algorithm"],
            env=training_env,
            tensorboard_log=run_artifacts.logs_dir / "tensorboard",
            device=algorithm_device,
        )
        milestones_dir = run_artifacts.models_dir / "milestones"
        milestone_models = _create_milestone_manifest(
            model=model,
            milestones_dir=milestones_dir,
        )
        callbacks, eval_callback = _build_callbacks(
            experiment_config=experiment_config,
            run_root=run_artifacts.run_root,
            num_envs=int(training_config["train_num_envs"]),
            eval_env=evaluation_env,
            algorithm=algorithm,
        )
        initial_eval_bundle = eval_callback.record_initial_evaluation(model)

        model.learn(
            total_timesteps=int(training_config["total_timesteps"]),
            callback=callbacks,
            progress_bar=True,
        )

        final_model_path = run_artifacts.models_dir / "final_model.zip"
        model.save(str(final_model_path))
        vec_normalize_env = model.get_vec_normalize_env()
        if vec_normalize_env is not None:
            vec_normalize_env.save(
                str(artifact_store.get_vecnormalize_path(run_artifacts))
            )
        milestone_models = _finalize_milestone_manifest(
            milestone_models=milestone_models,
            milestones_dir=milestones_dir,
            checkpoints_dir=run_artifacts.checkpoints_dir,
            final_model_path=final_model_path,
            total_timesteps=int(training_config["total_timesteps"]),
        )

        best_model_path = run_artifacts.models_dir / "best_model" / "best_model.zip"
        best_model_exists = best_model_path.exists()

        final_bundle = _evaluate_saved_model(
            algorithm=algorithm,
            model_path=final_model_path,
            experiment_config=experiment_config,
            model_device=algorithm_device,
            episodes=int(training_config["n_eval_episodes"]),
        )

        best_bundle = None
        if best_model_exists:
            best_bundle = _evaluate_saved_model(
                algorithm=algorithm,
                model_path=best_model_path,
                experiment_config=experiment_config,
                model_device=algorithm_device,
                episodes=int(training_config["n_eval_episodes"]),
            )

        summary = {
            "experiment_name": experiment_name,
            "algorithm": algorithm,
            "hyperparams_source": hyperparams_source,
            "seed": seed,
            "device": resolved_genesis_device,
            "genesis_device": resolved_genesis_device,
            "algorithm_device": algorithm_device,
            "runtime_device": resolved_genesis_device,
            "run_id": run_artifacts.run_id,
            "run_root": str(run_artifacts.run_root),
            "final_model_path": str(final_model_path),
            "best_model_path": str(best_model_path) if best_model_exists else None,
            "vecnormalize_path": str(
                artifact_store.get_vecnormalize_path(run_artifacts)
            )
            if artifact_store.get_vecnormalize_path(run_artifacts).exists()
            else None,
            "best_vecnormalize_path": str(
                artifact_store.get_best_vecnormalize_path(run_artifacts)
            )
            if artifact_store.get_best_vecnormalize_path(run_artifacts).exists()
            else None,
            "milestone_models": milestone_models,
            "evaluation_protocol": build_evaluation_protocol(experiment_config),
            "initial_eval": summarize_evaluation_bundle(initial_eval_bundle),
            "final_eval": summarize_evaluation_bundle(final_bundle),
            "best_eval": (
                summarize_evaluation_bundle(best_bundle)
                if best_bundle is not None
                else None
            ),
            "final_metrics_deterministic": final_bundle["metrics_deterministic"],
            "final_metrics_stochastic": final_bundle.get("metrics_stochastic"),
            "best_metrics_deterministic": (
                best_bundle["metrics_deterministic"]
                if best_bundle is not None
                else None
            ),
            "best_metrics_stochastic": (
                best_bundle.get("metrics_stochastic")
                if best_bundle is not None
                else None
            ),
            "final_task_score": float(final_bundle["task_score"]),
            "final_selection_score": float(final_bundle["selection_score"]),
            "best_task_score": (
                float(best_bundle["task_score"]) if best_bundle is not None else None
            ),
            "best_selection_score": (
                float(best_bundle["selection_score"])
                if best_bundle is not None
                else None
            ),
            "evaluation_history_path": str(eval_callback.history_path),
            "runtime_context": collect_runtime_context(),
        }

        summary_path = run_artifacts.eval_dir / "training_summary.json"
        write_json(summary_path, to_json_compatible(summary))
    finally:
        if training_env is not None:
            training_env.close()

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
        LOGGER.info("Training finished. Summary saved to %s", summary_path)

    return summary
