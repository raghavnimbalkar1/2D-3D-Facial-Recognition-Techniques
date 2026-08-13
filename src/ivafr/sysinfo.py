"""System / environment fingerprinting.

Written into every run directory as ``sysinfo.json`` so any number in a
metrics file is traceable to the machine and environment that produced it
(required by the "Experimental Setup" section of the report).
"""

from __future__ import annotations

import os
import platform
import socket
import subprocess
from pathlib import Path
from typing import Any

import psutil


def git_sha(repo_root: Path | None = None) -> str | None:
    """Return the short SHA of the current git HEAD, if inside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root or Path.cwd(),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def cpu_model() -> str:
    """Return a human-readable CPU model string."""
    system = platform.system()
    if system == "Darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
        return platform.processor() or "unknown"
    if system == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
        return platform.processor() or "unknown"
    return platform.processor() or "unknown"


def sysinfo(repo_root: Path | None = None) -> dict[str, Any]:
    """Collect an environment fingerprint dict.

    Args:
        repo_root: repository root for the git SHA lookup.

    Returns:
        Flat dict with hardware, OS, python and process info.
    """
    import platform as _p

    return {
        "git_sha": git_sha(repo_root),
        "python": _p.python_version(),
        "python_impl": _p.python_implementation(),
        "os": _p.system(),
        "os_release": _p.release(),
        "os_version": _p.version(),
        "machine": _p.machine(),
        "hostname": socket.gethostname(),
        "cpu": cpu_model(),
        "cpu_count": os.cpu_count(),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "pid": os.getpid(),
    }
