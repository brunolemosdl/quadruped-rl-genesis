"""Live metrics monitor subprocess and snapshot formatting."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from quadruped_rl_genesis.services.logger import get_logger
from quadruped_rl_genesis.services.monitor import (
    build_monitor_snapshot,
    format_monitor_snapshot,
    resolve_monitor_layout,
)
from quadruped_rl_genesis.settings import AppSettings

LOGGER = get_logger(__name__)


def run_monitor(
    *,
    settings: AppSettings,
    mode: str,
    output_root: str | Path | None,
    artifacts_root: str | Path | None,
    experiment_name: str | None,
    algorithm: str | None,
    host: str,
    port: int,
    reload_interval: int,
    watch: bool,
    interval: int,
    limit: int,
) -> dict[str, Any]:
    """Monitor training/tuning progress from TensorBoard or generated artifacts.

    Args:
        settings (AppSettings): Global application settings.
        mode (str): Monitoring mode: ``summary`` or ``tensorboard``.
        output_root (str | Path | None): Benchmark output root override.
        artifacts_root (str | Path | None): Explicit artifacts root override.
        experiment_name (str | None): Optional experiment filter.
        algorithm (str | None): Optional algorithm filter.
        host (str): TensorBoard host binding.
        port (int): TensorBoard port binding.
        reload_interval (int): TensorBoard reload interval in seconds.
        watch (bool): Refresh summary continuously when enabled.
        interval (int): Refresh cadence for ``summary --watch``.
        limit (int): Maximum number of experiment rows shown in summary mode.

    Returns:
        dict[str, Any]: Summary snapshot or TensorBoard launch metadata.
    """
    if mode == "summary":
        return _run_summary_monitor(
            settings=settings,
            output_root=output_root,
            artifacts_root=artifacts_root,
            experiment_name=experiment_name,
            algorithm=algorithm,
            watch=watch,
            interval=interval,
            limit=limit,
        )

    if mode == "tensorboard":
        return _run_tensorboard_monitor(
            settings=settings,
            output_root=output_root,
            artifacts_root=artifacts_root,
            host=host,
            port=port,
            reload_interval=reload_interval,
        )

    raise ValueError(f"Unsupported monitor mode: {mode}")


def _run_summary_monitor(
    *,
    settings: AppSettings,
    output_root: str | Path | None,
    artifacts_root: str | Path | None,
    experiment_name: str | None,
    algorithm: str | None,
    watch: bool,
    interval: int,
    limit: int,
) -> dict[str, Any]:
    last_snapshot: dict[str, Any] | None = None
    last_text = ""

    try:
        while True:
            last_snapshot = build_monitor_snapshot(
                settings=settings,
                output_root=output_root,
                artifacts_root=artifacts_root,
                experiment_name=experiment_name,
                algorithm=algorithm,
                limit=limit,
            )
            last_text = format_monitor_snapshot(last_snapshot)

            if watch:
                print("\033[2J\033[H", end="")

            print(last_text, flush=True)

            if not watch:
                break

            time.sleep(max(interval, 1))
    except KeyboardInterrupt:
        LOGGER.info("Stopped summary monitor.")

    snapshot = last_snapshot or {
        "artifacts_root": None,
        "benchmark_root": None,
        "benchmark_summary": None,
        "runs": [],
    }
    snapshot["mode"] = "summary"
    snapshot["text"] = last_text

    return snapshot


def _run_tensorboard_monitor(
    *,
    settings: AppSettings,
    output_root: str | Path | None,
    artifacts_root: str | Path | None,
    host: str,
    port: int,
    reload_interval: int,
) -> dict[str, Any]:
    layout = resolve_monitor_layout(
        settings,
        output_root=output_root,
        artifacts_root=artifacts_root,
    )
    resolved_artifacts_root = layout["artifacts_root"]
    assert resolved_artifacts_root is not None

    url = f"http://{host}:{int(port)}"
    command = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(resolved_artifacts_root),
        "--host",
        host,
        "--port",
        str(int(port)),
        "--reload_interval",
        str(max(int(reload_interval), 1)),
    ]

    LOGGER.info(
        "Starting TensorBoard | logdir=%s url=%s",
        resolved_artifacts_root,
        url,
    )
    print(
        (
            f"Starting TensorBoard at {url}\n"
            f"Logdir: {resolved_artifacts_root}\n"
            "Press Ctrl+C to stop."
        ),
        flush=True,
    )
    subprocess.run(command, check=True)

    return {
        "mode": "tensorboard",
        "artifacts_root": str(resolved_artifacts_root),
        "url": url,
        "command": command,
    }
