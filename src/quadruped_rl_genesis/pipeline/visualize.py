"""Interactive or recorded rollouts with optional metrics overlay."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from quadruped_rl_genesis.algorithms.factory import load_sb3_model
from quadruped_rl_genesis.config.loader import load_resolved_config
from quadruped_rl_genesis.environments.factory import build_vector_env
from quadruped_rl_genesis.interface.telemetry import (
    LAST_REWARDS_COUNT,
    build_metrics_card,
    build_minimap_state,
    get_genesis_window_geometry,
)
from quadruped_rl_genesis.pipeline.evaluate import (
    apply_eval_experiment_environment_overlay,
)
from quadruped_rl_genesis.services.artifacts import (
    ArtifactStore,
    find_run_root_from_model_path,
    find_vecnormalize_path_from_model_path,
)
from quadruped_rl_genesis.services.logger import configure_logging, get_logger
from quadruped_rl_genesis.services.metrics import RandomPolicy
from quadruped_rl_genesis.services.runtime import initialize_genesis
from quadruped_rl_genesis.settings import AppSettings
from quadruped_rl_genesis.utils.io import read_yaml, to_json_compatible, write_json
from quadruped_rl_genesis.utils.seed import set_global_seeds

LOGGER = get_logger(__name__)


def run_visualization(
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
    episodes: int,
    max_steps: int,
    record_video: bool,
    no_model: bool = False,
    fast_viz: bool = False,
    tag: str | None = None,
    headless: bool = False,
    eval_experiment_name: str | None = None,
) -> dict[str, Any]:
    """Run rollout with optional viewer, metrics overlay, and video capture.

    Builds the env with viewer (and camera if recording), loads the policy unless
    no_model is True (zero actions), runs the requested number of episodes while
    optionally showing a live metrics overlay next to the simulator window.
    Optionally
    records frames to an MP4 and always writes a JSON summary to disk.

    Args:
        settings: Global application settings.
        experiment_name: Experiment profile name.
        algorithm: Algorithm name.
        hyperparams_source: Hyperparameter source when config cannot be
            inferred from the model path.
        seed: Visualization seed.
        genesis_device: Device for Genesis simulation.
        algorithm_device: Device for SB3/RL algorithm.
        model_kind: Requested model kind (e.g. ``"best"`` or ``"final"``).
        model_path: Optional explicit model path; otherwise resolved from
            artifact store.
        run_id: Optional explicit training run identifier.
        train_seed: Optional training seed used to resolve the target run when
            ``model_path`` and ``run_id`` are not provided.
        episodes: Number of rollout episodes.
        max_steps: Maximum steps per episode.
        record_video: If True, save frames as MP4 in the run or artifacts dir.
        no_model: If True, run with zero actions (no trained policy). Incompatible
            with ``--algorithm random`` (enforced in the CLI).
        fast_viz: If True, use lighter viz (e.g. no shadows, lower resolution).
        tag: Optional tag for output filenames (video and summary JSON).
        headless: If True, do not open viewer/overlay windows. Useful for
            recording videos in remote or CI/headless environments.
        eval_experiment_name: If set, replace the training run's ``environment``
            block with that profile (e.g. ``"irregular"`` for procedural irregular
            terrain, or ``"rough"`` for Genesis subterrain grids).

    Returns:
        Visualization summary dict (also written to
        ``{output_dir}/{algorithm}_{tag}_visualization.json``).

    Raises:
        FileNotFoundError: When a model is required but none could be resolved.
    """
    is_random_baseline = algorithm.lower() == "random"
    if no_model and is_random_baseline:
        raise ValueError("no_model is incompatible with algorithm 'random'.")

    artifact_store = ArtifactStore(settings)
    resolved_model_path = None
    selected_run_root: Path | None = None

    if not no_model and not is_random_baseline:
        resolved_model_path = Path(model_path) if model_path else None

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
            raise FileNotFoundError("No model was found for visualization.")

    run_root: Path | None = None
    if resolved_model_path is not None:
        run_root = selected_run_root or find_run_root_from_model_path(
            resolved_model_path
        )
    output_root: Path | None = None
    if run_root is not None:
        resolved_config = read_yaml(run_root / "config" / "resolved_config.yaml")
        log_file = run_root / "logs" / "visualization.log"
        output_dir = run_root / "videos"
    else:
        resolved_config = load_resolved_config(
            settings=settings,
            artifact_store=artifact_store,
            experiment_name=experiment_name,
            algorithm=algorithm,
            hyperparams_source=hyperparams_source,
        )
        label = tag
        if label is None:
            if is_random_baseline:
                label = "random_uniform"
            elif no_model:
                label = "no_model"
            elif resolved_model_path is not None:
                label = resolved_model_path.stem
            else:
                label = model_kind

        output_root = artifact_store.create_ad_hoc_execution_dir(
            "visualize",
            experiment_name,
            algorithm,
            seed,
            label=label,
        )
        log_file = output_root / "visualization.log"
        output_dir = output_root

    resolved_genesis_device = initialize_genesis(seed=seed, device=genesis_device)
    configure_logging(log_file, settings.log_level)
    set_global_seeds(seed)

    if eval_experiment_name:
        apply_eval_experiment_environment_overlay(
            settings,
            resolved_config,
            eval_experiment_name,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_config = resolved_config["experiment"]
    runtime_config = resolved_config.get("runtime", {})
    effective_experiment_name = str(
        runtime_config.get("experiment_name", experiment_name)
    )
    effective_algorithm = str(runtime_config.get("algorithm", algorithm))
    if is_random_baseline:
        effective_algorithm = "random"
    effective_hyperparams_source = str(
        runtime_config.get("hyperparams_source", hyperparams_source)
    )
    parsed_run = (
        ArtifactStore.parse_run_id(run_root.name) if run_root is not None else None
    )

    if no_model:
        LOGGER.info(
            "Visualization will run without a trained model using zero actions."
        )
    if is_random_baseline:
        LOGGER.info(
            "Visualization will use a uniform random policy baseline (no checkpoint)."
        )

    if fast_viz:
        LOGGER.info("Fast viz enabled: shadows off, lower resolution, fewer substeps.")
    if headless:
        LOGGER.info("Headless visualization enabled: no viewer/overlay windows.")

    visualization_env = None
    summary_path = None
    try:
        visualization_env = build_vector_env(
            experiment_config=experiment_config,
            num_envs=1,
            show_viewer=not headless,
            add_camera=record_video,
            monitor=True,
            fast_viz=fast_viz,
            viewer_help_text=not headless,
            disable_reward_curriculum=True,
            vecnormalize_path=(
                None
                if is_random_baseline
                else (
                    find_vecnormalize_path_from_model_path(resolved_model_path)
                    if resolved_model_path is not None
                    else None
                )
            ),
            for_training=False,
            norm_reward=False,
        )
        model = None
        random_actor: RandomPolicy | None = None

        if is_random_baseline:
            random_actor = RandomPolicy(
                visualization_env.action_space,
                np.random.default_rng(seed),
            )
        elif not no_model:
            model = load_sb3_model(
                algorithm=algorithm,
                model_path=resolved_model_path,
                env=visualization_env,
                device=algorithm_device,
            )

        frames = []
        episode_returns = []
        zero_action = np.zeros(
            (visualization_env.num_envs, *visualization_env.action_space.shape),
            dtype=np.float32,
        )

        observations = visualization_env.reset()

        unwrapped = getattr(visualization_env, "venv", visualization_env)
        task = getattr(unwrapped, "task", None)
        last_rewards: list[float] = []
        overlay = None
        if not headless:
            from quadruped_rl_genesis.interface.overlay import MetricsOverlayWindow

            overlay = MetricsOverlayWindow()

        try:
            for episode_index in range(episodes):
                episode_return = 0.0
                last_rewards.clear()
                for step_index in range(max_steps):
                    if overlay is not None and task is not None:
                        pending_goal = overlay.consume_pending_goal_xy()
                        if pending_goal is not None:
                            task.set_goal_position_xy(
                                float(pending_goal[0]),
                                float(pending_goal[1]),
                                env_idx=0,
                                reset_trackers=True,
                            )

                    if random_actor is not None:
                        action, _ = random_actor.predict(
                            observations, deterministic=True
                        )
                    elif model is None:
                        action = zero_action
                    else:
                        action, _ = model.predict(observations, deterministic=True)

                    observations, rewards, dones, infos = visualization_env.step(action)
                    step_reward = float(rewards[0])
                    episode_return += step_reward
                    done = bool(dones[0])
                    last_rewards.append(step_reward)
                    if len(last_rewards) > LAST_REWARDS_COUNT:
                        last_rewards = last_rewards[-LAST_REWARDS_COUNT:]

                    card_text = build_metrics_card(
                        unwrapped,
                        step_reward,
                        episode_return,
                        step_index,
                        episode_index,
                        done,
                        infos[0] if infos else None,
                        last_rewards,
                        actions=action,
                    )
                    if overlay is not None:
                        minimap_state = (
                            build_minimap_state(task) if task is not None else None
                        )
                        overlay.update(card_text, minimap_state=minimap_state)
                        geo = get_genesis_window_geometry(unwrapped)
                        if geo is not None:
                            overlay.reposition_next_to(*geo)
                        overlay.pump_events()

                    if record_video:
                        frame = visualization_env.get_images()[0]
                        if frame is not None:
                            frames.append(frame)

                    if done:
                        break

                episode_returns.append(episode_return)
                observations = visualization_env.reset()
        finally:
            if overlay is not None:
                overlay.destroy()

        output_tag = tag
        if output_tag is None:
            if is_random_baseline:
                output_tag = "random_uniform"
            elif no_model:
                output_tag = "no_model"
            elif resolved_model_path is not None:
                output_tag = resolved_model_path.stem
            else:
                output_tag = model_kind

        safe_tag = (
            "".join(
                character if character.isalnum() or character in ("-", "_") else "_"
                for character in output_tag
            ).strip("_")
            or model_kind
        )

        video_path = None
        if record_video and frames:
            video_algo = algorithm
            video_path = output_dir / f"{video_algo}_{safe_tag}_visualization.mp4"
            imageio.mimsave(video_path, frames, fps=30)
        elif record_video:
            LOGGER.warning(
                "Video recording requested but no frames were captured. "
                "Check camera backend availability and visualization camera settings."
            )

        summary = {
            "experiment_name": effective_experiment_name,
            "eval_environment_experiment": eval_experiment_name,
            "algorithm": effective_algorithm,
            "hyperparams_source": effective_hyperparams_source,
            "model_kind": ("random_uniform" if is_random_baseline else model_kind),
            "tag": safe_tag,
            "model_path": (
                str(resolved_model_path) if resolved_model_path is not None else None
            ),
            "run_id": run_root.name if run_root is not None else None,
            "run_root": str(run_root) if run_root is not None else None,
            "train_seed": (
                int(parsed_run["seed"])
                if parsed_run is not None
                else (int(train_seed) if train_seed is not None else None)
            ),
            "no_model": no_model,
            "episodes": episodes,
            "max_steps": max_steps,
            "record_video": record_video,
            "video_path": str(video_path) if video_path is not None else None,
            "episode_returns": episode_returns,
            "device": resolved_genesis_device,
            "genesis_device": resolved_genesis_device,
            "algorithm_device": algorithm_device,
            "runtime_device": resolved_genesis_device,
            "headless": headless,
        }
        if run_root is not None:
            summary_path = output_dir / f"{safe_tag}_visualization_summary.json"
        else:
            summary_path = output_dir / "visualization_summary.json"
        write_json(summary_path, to_json_compatible(summary))
    finally:
        if visualization_env is not None:
            visualization_env.close()

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    if summary_path is not None:
        LOGGER.info("Visualization finished. Summary saved to %s", summary_path)

    return summary
