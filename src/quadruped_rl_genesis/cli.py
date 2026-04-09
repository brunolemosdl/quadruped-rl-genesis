"""Command-line entrypoint and subcommand parsers."""

from __future__ import annotations

import argparse

from quadruped_rl_genesis.config.benchmark import (
    load_profile,
    merge_benchmark_invocation,
    resolve_profile_path,
)
from quadruped_rl_genesis.services.logger import configure_logging
from quadruped_rl_genesis.settings import load_app_settings


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the top-level CLI parser for all project workflows.

    Returns:
        argparse.ArgumentParser: Fully configured parser with setup, validation,
            tuning, training, evaluation, visualization, and benchmark
            subcommands.
    """
    settings = load_app_settings()
    parser = argparse.ArgumentParser(
        description=(
            "Quadruped RL Genesis CLI for setup, validation, monitoring, training, "
            "tuning, evaluation, visualization and final benchmark execution."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--experiment",
        type=str,
        default=settings.default_experiment,
        help="Experiment profile name under configs/experiments.",
    )
    common.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=["random", "ppo", "sac", "td3"],
        help=(
            "Algorithm to run. Use 'random' only for evaluate/visualize (uniform random baseline)."
        ),
    )
    common.add_argument(
        "--seed",
        type=int,
        default=settings.seed,
        help="Global random seed.",
    )
    common.add_argument(
        "--device",
        type=str,
        default=settings.device,
        help="Default runtime device for both Genesis and algorithm (auto, cpu, cuda).",
    )
    common.add_argument(
        "--genesis-device",
        type=str,
        default=None,
        metavar="DEV",
        help="Device for Genesis simulation. Overrides --device when set.",
    )
    common.add_argument(
        "--algorithm-device",
        type=str,
        default=None,
        metavar="DEV",
        help="Device for RL algorithm (SB3). Overrides --device when set.",
    )
    common.add_argument(
        "--hyperparams-source",
        type=str,
        default=settings.default_hyperparams_source,
        help=(
            "Named hyperparameter source such as 'default', 'optuna', or "
            "'utd_gs2'. For 'tune', this selects the base profile used before "
            "Optuna trial overrides."
        ),
    )

    setup_parser = subparsers.add_parser(
        "setup", help="Create/update the local environment."
    )
    setup_parser.add_argument(
        "--python-bin",
        type=str,
        default=None,
        help="Optional Python executable used to create the virtual environment.",
    )
    setup_parser.add_argument(
        "--venv-dir",
        type=str,
        default=None,
        help="Optional virtual environment directory. Defaults to .venv.",
    )
    setup_parser.set_defaults(action="setup")

    check_parser = subparsers.add_parser("check", help="Validate the local runtime.")
    check_parser.set_defaults(action="check")

    monitor_parser = subparsers.add_parser(
        "monitor",
        help="Inspect recent artifacts or launch TensorBoard for live monitoring.",
    )
    monitor_parser.add_argument(
        "--mode",
        type=str,
        choices=["summary", "tensorboard"],
        default="summary",
        help="Use a terminal summary or launch TensorBoard.",
    )
    monitor_parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Benchmark output root containing artifacts/ and reports/.",
    )
    monitor_parser.add_argument(
        "--artifacts-root",
        type=str,
        default=None,
        help="Explicit artifacts root override. Cannot be combined with --output-root.",
    )
    monitor_parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Optional experiment filter used by summary mode.",
    )
    monitor_parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        choices=["random", "ppo", "sac", "td3"],
        help="Optional algorithm filter used by summary mode.",
    )
    monitor_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="TensorBoard host binding.",
    )
    monitor_parser.add_argument(
        "--port",
        type=int,
        default=6006,
        help="TensorBoard port binding.",
    )
    monitor_parser.add_argument(
        "--reload-interval",
        type=int,
        default=15,
        help="TensorBoard reload interval in seconds.",
    )
    monitor_parser.add_argument(
        "--watch",
        action="store_true",
        help="Refresh summary output continuously until interrupted.",
    )
    monitor_parser.add_argument(
        "--interval",
        type=int,
        default=15,
        help="Refresh cadence in seconds for --watch.",
    )
    monitor_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of experiment/algorithm rows shown in summary mode.",
    )
    monitor_parser.set_defaults(action="monitor")

    device_options = argparse.ArgumentParser(add_help=False)
    device_options.add_argument(
        "--device",
        type=str,
        default=argparse.SUPPRESS,
        help=(
            "Default runtime device for both Genesis and algorithm (auto, cpu, cuda). "
            "Defaults from --profile or application settings."
        ),
    )
    device_options.add_argument(
        "--genesis-device",
        type=str,
        default=argparse.SUPPRESS,
        metavar="DEV",
        help="Device for Genesis simulation. Overrides --device when set.",
    )
    device_options.add_argument(
        "--algorithm-device",
        type=str,
        default=argparse.SUPPRESS,
        metavar="DEV",
        help="Device for RL algorithm. Overrides --device when set.",
    )

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        parents=[device_options],
        help="Run the final benchmark protocol for the article.",
    )
    benchmark_parser.add_argument(
        "--profile",
        type=str,
        default=None,
        metavar="NAME_OR_PATH",
        help=(
            "Optional YAML under configs/benchmarks/ (e.g. default) or a path to a "
            ".yaml file. CLI flags override values from the file."
        ),
    )
    benchmark_parser.add_argument(
        "--output-root",
        type=str,
        default=argparse.SUPPRESS,
        help="Root directory used to store benchmark artifacts, logs and reports.",
    )
    benchmark_parser.add_argument(
        "--experiments",
        nargs="+",
        default=argparse.SUPPRESS,
        help="Experiment profiles included in the benchmark.",
    )
    benchmark_parser.add_argument(
        "--algorithms",
        nargs="+",
        default=argparse.SUPPRESS,
        choices=["ppo", "sac", "td3"],
        help="Algorithms included in the benchmark.",
    )
    benchmark_parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=argparse.SUPPRESS,
        help="Training seeds used for the final benchmark.",
    )
    benchmark_parser.add_argument(
        "--hyperparams-source",
        type=str,
        default=argparse.SUPPRESS,
        help=(
            "Hyperparameter source used for training and evaluation. Defaults from "
            "--profile or QUADRUPED_RL_GENESIS_DEFAULT_HPARAMS_SOURCE."
        ),
    )
    benchmark_parser.add_argument(
        "--tune-base-source",
        type=str,
        default=argparse.SUPPRESS,
        help="Named base hyperparameter source used during Optuna tuning.",
    )
    benchmark_parser.add_argument(
        "--seed",
        type=int,
        default=argparse.SUPPRESS,
        help=(
            "Default seed for Optuna and final evaluation unless --tune-seed or "
            "--eval-seed are set. Defaults from --profile or application settings."
        ),
    )
    benchmark_parser.add_argument(
        "--tune-seed",
        type=int,
        default=argparse.SUPPRESS,
        help="Seed for Optuna studies (defaults to --seed).",
    )
    benchmark_parser.add_argument(
        "--eval-seed",
        type=int,
        default=argparse.SUPPRESS,
        help="Seed for final deterministic evaluation (defaults to --seed).",
    )
    benchmark_parser.add_argument(
        "--final-eval-episodes",
        type=int,
        default=argparse.SUPPRESS,
        help="Number of episodes in the final deterministic evaluation.",
    )
    benchmark_parser.add_argument(
        "--skip-tune",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip Optuna and use the requested hyperparameter source as-is.",
    )
    benchmark_parser.add_argument(
        "--skip-train",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip training runs.",
    )
    benchmark_parser.add_argument(
        "--skip-eval",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip final evaluation runs.",
    )
    benchmark_parser.add_argument(
        "--record-videos",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Record benchmark videos for the best checkpoint of each algorithm/scenario.",
    )
    benchmark_parser.add_argument(
        "--video-episodes",
        type=int,
        default=argparse.SUPPRESS,
        help="Number of visualization episodes recorded per benchmark video.",
    )
    benchmark_parser.add_argument(
        "--video-max-steps",
        type=int,
        default=argparse.SUPPRESS,
        help="Maximum steps per benchmark video episode.",
    )
    benchmark_parser.add_argument(
        "--video-seed",
        type=int,
        default=argparse.SUPPRESS,
        help="Optional seed for benchmark video rendering (defaults to resolved eval seed).",
    )
    benchmark_parser.add_argument(
        "--video-fast-viz",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use lighter visualization settings while recording videos.",
    )
    benchmark_parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Skip training runs that already have a completed run (same experiment, "
            "algorithm, seed). Use after a pod restart to continue the benchmark."
        ),
    )
    benchmark_parser.set_defaults(action="benchmark")

    train_parser = subparsers.add_parser(
        "train", parents=[common], help="Train a policy."
    )
    train_parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip execution when a completed run already exists for the same "
            "experiment, algorithm, seed, and hyperparameter source."
        ),
    )
    train_parser.set_defaults(action="train")

    tune_parser = subparsers.add_parser(
        "tune", parents=[common], help="Optimize hyperparameters with Optuna."
    )
    tune_parser.set_defaults(action="tune")

    evaluate_parser = subparsers.add_parser(
        "evaluate", parents=[common], help="Evaluate a saved model."
    )
    evaluate_parser.add_argument(
        "--model-kind",
        type=str,
        choices=["best", "final"],
        default="best",
        help="Use the latest best or final model when --model-path is not provided.",
    )
    evaluate_parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional explicit path to a model zip file.",
    )
    evaluate_parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional explicit training run identifier used to resolve the model.",
    )
    evaluate_parser.add_argument(
        "--train-seed",
        type=int,
        default=None,
        help="Optional training seed used to resolve the latest matching run.",
    )
    evaluate_parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override the number of evaluation episodes.",
    )
    evaluate_parser.add_argument(
        "--eval-experiment",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Override the simulation environment (YAML environment: block) for this evaluation "
            "only — e.g. train on flat but evaluate with configs/experiments/irregular.yaml or "
            "configs/experiments/rough.yaml terrain and rewards. The policy and VecNormalize "
            "stats still come from the loaded model."
        ),
    )
    evaluate_parser.set_defaults(action="evaluate")

    visualize_parser = subparsers.add_parser(
        "visualize", parents=[common], help="Visualize a trained model in Genesis."
    )
    visualize_parser.add_argument(
        "--model-kind",
        type=str,
        choices=["best", "final"],
        default="best",
        help="Use the latest best or final model when --model-path is not provided.",
    )
    visualize_parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional explicit path to a model zip file.",
    )
    visualize_parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional explicit training run identifier used to resolve the model.",
    )
    visualize_parser.add_argument(
        "--train-seed",
        type=int,
        default=None,
        help="Optional training seed used to resolve the latest matching run.",
    )
    visualize_parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of visualization episodes.",
    )
    visualize_parser.add_argument(
        "--max-steps",
        type=int,
        default=2000,
        help="Maximum steps per visualization episode.",
    )
    visualize_parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record an mp4 if the Genesis camera backend is available.",
    )
    visualize_parser.add_argument(
        "--no-model",
        action="store_true",
        help="Open the experiment without loading a trained policy and use zero actions instead.",
    )
    visualize_parser.add_argument(
        "--fast-viz",
        action="store_true",
        help="Use lighter visualization for higher FPS.",
    )
    visualize_parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional output tag used to name visualization artifacts.",
    )
    visualize_parser.add_argument(
        "--headless",
        action="store_true",
        help="Run visualization without opening viewer/overlay windows.",
    )
    visualize_parser.add_argument(
        "--eval-experiment",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Override the simulation environment (YAML environment: block) for this rollout "
            "only — e.g. train on flat but visualize with configs/experiments/irregular.yaml or "
            "configs/experiments/rough.yaml. Policy and VecNormalize stats still come from the "
            "loaded model."
        ),
    )
    visualize_parser.set_defaults(action="visualize")

    return parser


def main() -> None:
    """Parse CLI arguments and dispatch execution to the requested workflow.

    Raises:
        ValueError: If the parsed action is not recognized by the dispatcher.
    """
    parser = build_argument_parser()
    args = parser.parse_args()
    settings = load_app_settings()
    configure_logging(level=settings.log_level)

    if args.action == "evaluate" and args.algorithm == "random":
        if args.hyperparams_source != "default":
            parser.error("Algorithm 'random' requires --hyperparams-source default.")
        if args.model_path or args.run_id or args.train_seed is not None:
            parser.error(
                "Algorithm 'random' cannot be used with --model-path, --run-id, or --train-seed."
            )

    if args.action == "visualize" and args.algorithm == "random":
        if args.hyperparams_source != "default":
            parser.error("Algorithm 'random' requires --hyperparams-source default.")
        if args.no_model:
            parser.error("Algorithm 'random' cannot be combined with --no-model.")
        if args.model_path or args.run_id or args.train_seed is not None:
            parser.error(
                "Algorithm 'random' cannot be used with --model-path, --run-id, or --train-seed."
            )

    if args.action == "setup":
        from quadruped_rl_genesis.operations import run_setup

        run_setup(
            settings=settings,
            python_bin=args.python_bin,
            venv_dir=args.venv_dir,
        )
        return

    if args.action == "check":
        from quadruped_rl_genesis.operations import run_check

        run_check(settings=settings)
        return

    if args.action == "monitor":
        from quadruped_rl_genesis.operations import run_monitor

        run_monitor(
            settings=settings,
            mode=args.mode,
            output_root=args.output_root,
            artifacts_root=args.artifacts_root,
            experiment_name=args.experiment,
            algorithm=args.algorithm,
            host=args.host,
            port=args.port,
            reload_interval=args.reload_interval,
            watch=args.watch,
            interval=args.interval,
            limit=args.limit,
        )
        return

    def _genesis_device(a) -> str:
        return a.genesis_device if getattr(a, "genesis_device", None) else a.device

    def _algorithm_device(a) -> str:
        return a.algorithm_device if getattr(a, "algorithm_device", None) else a.device

    if args.action == "benchmark":
        profile_payload: dict = {}
        if args.profile:
            profile_payload = load_profile(resolve_profile_path(settings, args.profile))
        resolved = merge_benchmark_invocation(args, settings, profile_payload)
        from quadruped_rl_genesis.operations import run_benchmark

        run_benchmark(
            settings=settings,
            output_root=resolved.output_root,
            experiments=resolved.experiments,
            algorithms=resolved.algorithms,
            seeds=resolved.seeds,
            device=resolved.device,
            genesis_device=resolved.genesis_device,
            algorithm_device=resolved.algorithm_device,
            hyperparams_source=resolved.hyperparams_source,
            tune_base_source=resolved.tune_base_source,
            tune_seed=resolved.tune_seed,
            eval_seed=resolved.eval_seed,
            final_eval_episodes=resolved.final_eval_episodes,
            skip_tune=resolved.skip_tune,
            skip_train=resolved.skip_train,
            skip_eval=resolved.skip_eval,
            record_videos=resolved.record_videos,
            video_episodes=resolved.video_episodes,
            video_max_steps=resolved.video_max_steps,
            video_seed=resolved.video_seed,
            video_fast_viz=resolved.video_fast_viz,
            resume=resolved.resume,
        )
        return

    if args.action == "train":
        if args.algorithm == "random":
            parser.error("Algorithm 'random' is only valid for evaluate and visualize.")
        from quadruped_rl_genesis.pipeline.train import run_training

        run_training(
            settings=settings,
            experiment_name=args.experiment,
            algorithm=args.algorithm,
            hyperparams_source=args.hyperparams_source,
            seed=args.seed,
            genesis_device=_genesis_device(args),
            algorithm_device=_algorithm_device(args),
            resume=bool(getattr(args, "resume", False)),
        )
        return

    if args.action == "tune":
        if args.algorithm == "random":
            parser.error("Algorithm 'random' is only valid for evaluate and visualize.")
        from quadruped_rl_genesis.pipeline.tune import run_tuning

        run_tuning(
            settings=settings,
            experiment_name=args.experiment,
            algorithm=args.algorithm,
            seed=args.seed,
            genesis_device=_genesis_device(args),
            algorithm_device=_algorithm_device(args),
            base_hyperparams_source=args.hyperparams_source,
        )
        return

    if args.action == "evaluate":
        from quadruped_rl_genesis.pipeline.evaluate import run_evaluation

        run_evaluation(
            settings=settings,
            experiment_name=args.experiment,
            algorithm=args.algorithm,
            hyperparams_source=args.hyperparams_source,
            seed=args.seed,
            genesis_device=_genesis_device(args),
            algorithm_device=_algorithm_device(args),
            model_kind=args.model_kind,
            model_path=args.model_path,
            run_id=args.run_id,
            train_seed=args.train_seed,
            episodes=args.episodes,
            eval_experiment_name=args.eval_experiment,
        )
        return

    if args.action == "visualize":
        from quadruped_rl_genesis.pipeline.visualize import run_visualization

        run_visualization(
            settings=settings,
            experiment_name=args.experiment,
            algorithm=args.algorithm,
            hyperparams_source=args.hyperparams_source,
            seed=args.seed,
            genesis_device=_genesis_device(args),
            algorithm_device=_algorithm_device(args),
            model_kind=args.model_kind,
            model_path=args.model_path,
            run_id=args.run_id,
            train_seed=args.train_seed,
            episodes=args.episodes,
            max_steps=args.max_steps,
            record_video=args.record_video,
            no_model=args.no_model,
            fast_viz=args.fast_viz,
            tag=args.tag,
            headless=bool(getattr(args, "headless", False)),
            eval_experiment_name=args.eval_experiment,
        )
        return

    raise ValueError(f"Unsupported action: {args.action}")
