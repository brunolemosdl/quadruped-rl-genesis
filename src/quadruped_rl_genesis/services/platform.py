"""Host platform, GPU, and runtime context introspection."""

from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any


def _collect_git_context() -> dict[str, str]:
    """Collect lightweight Git metadata when the current workspace is versioned.

    Returns:
        dict[str, str]: Mapping with keys such as ``commit`` and ``branch`` when
            Git is available and the commands succeed. Returns an empty dict
            otherwise.
    """
    if shutil.which("git") is None:
        return {}

    payload: dict[str, str] = {}
    commands = {
        "commit": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "rev-parse", "--abbrev-ref", "HEAD"],
    }

    for name, command in commands.items():
        result = subprocess.run(command, capture_output=True, text=True, check=False)

        if result.returncode == 0:
            payload[name] = result.stdout.strip()

    return payload


def _collect_gpu_context() -> list[dict[str, str]]:
    """Collect GPU metadata from ``nvidia-smi`` when it is available.

    Returns:
        list[dict[str, str]]: One dictionary per GPU containing index, model
            name, total memory, and driver version. Returns an empty list when
            ``nvidia-smi`` is unavailable or fails.
    """
    if shutil.which("nvidia-smi") is None:
        return []

    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        return []

    rows: list[dict[str, str]] = []

    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]

        if len(parts) != 4:
            continue

        rows.append(
            {
                "index": parts[0],
                "name": parts[1],
                "memory_total_mb": parts[2],
                "driver_version": parts[3],
            }
        )

    return rows


def _collect_torch_context() -> dict[str, Any]:
    """Collect minimal Torch runtime metadata when Torch can be imported.

    Returns:
        dict[str, Any]: Mapping with Torch version and CUDA availability data.
            Returns an empty dict when Torch cannot be imported.
    """
    try:
        import torch
    except ImportError:
        return {}

    payload: dict[str, Any] = {
        "version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }

    if torch.cuda.is_available():
        payload["cuda_device_count"] = int(torch.cuda.device_count())

    return payload


def collect_runtime_context() -> dict[str, Any]:
    """Collect reproducibility-oriented metadata about the current runtime.

    Returns:
        dict[str, Any]: Summary with host, platform, Python, Torch, GPU, and
            optional Git metadata.
    """
    payload: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "cwd": str(Path.cwd()),
        "cpu_count": os.cpu_count(),
        "gpu": _collect_gpu_context(),
        "torch": _collect_torch_context(),
    }

    git_context = _collect_git_context()

    if git_context:
        payload["git"] = git_context

    return payload
