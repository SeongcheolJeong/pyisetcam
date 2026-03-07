"""GNU Octave command discovery and execution helpers."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Sequence

from .exceptions import OctaveExecutionError

OCTAVE_BIN_ENV = "PYISETCAM_OCTAVE_BIN"
_CRASH_TIMESTAMP = re.compile(r"-(\d{4}-\d{2}-\d{2}-\d{6}(?:\.\d{3})?)\.ips$")


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve() if path.exists() else path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def octave_search_paths(*, search_paths: Sequence[Path] | None = None) -> list[Path]:
    """Return the preferred directory list for locating an Octave binary."""

    if search_paths is not None:
        return _dedupe_paths(Path(path) for path in search_paths)

    path_entries = [Path(entry) for entry in os.environ.get("PATH", "").split(os.pathsep) if entry]
    extra_entries = [
        Path.home() / "miniforge3" / "envs" / "isetcam-py" / "bin",
        Path.home() / "miniforge3" / "bin",
    ]
    return _dedupe_paths([*path_entries, *extra_entries])


def _find_versioned_octave_cli(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    matches = sorted(directory.glob("octave-cli-*"), reverse=True)
    return [path for path in matches if path.is_file() and os.access(path, os.X_OK)]


def find_octave_binary(
    *,
    preferred: str | Path | None = None,
    search_paths: Sequence[Path] | None = None,
) -> Path:
    """Resolve the best available Octave executable for parity export."""

    explicit = preferred or os.environ.get(OCTAVE_BIN_ENV)
    if explicit:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Configured Octave binary does not exist: {path}")
        if not path.is_file() or not os.access(path, os.X_OK):
            raise FileNotFoundError(f"Configured Octave binary is not executable: {path}")
        return path

    directories = octave_search_paths(search_paths=search_paths)
    for directory in directories:
        versioned = _find_versioned_octave_cli(directory)
        if versioned:
            return versioned[0]

    for name in ("octave-cli", "octave"):
        resolved = shutil.which(name)
        if resolved:
            path = Path(resolved)
            if path.is_file() and os.access(path, os.X_OK):
                return path

    raise FileNotFoundError(
        "Unable to locate GNU Octave. Set PYISETCAM_OCTAVE_BIN to an executable octave-cli binary."
    )


def latest_octave_crash_log(
    *,
    diagnostics_dir: str | Path | None = None,
    prefix: str = "octave",
) -> Path | None:
    """Return the newest macOS Octave crash report if one exists."""

    default_dir = Path.home() / "Library" / "Logs" / "DiagnosticReports"
    root = Path(diagnostics_dir) if diagnostics_dir is not None else default_dir
    if not root.exists():
        return None

    def sort_key(path: Path) -> tuple[str, float]:
        match = _CRASH_TIMESTAMP.search(path.name)
        timestamp = match.group(1) if match is not None else ""
        return (timestamp, path.stat().st_mtime)

    candidates = sorted(
        root.glob(f"{prefix}*.ips"),
        key=sort_key,
        reverse=True,
    )
    return candidates[0] if candidates else None


def run_octave(
    arguments: Sequence[str],
    *,
    cwd: str | Path,
    octave_bin: str | Path | None = None,
    search_paths: Sequence[Path] | None = None,
    timeout_s: float = 300.0,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Execute Octave and raise a rich error if the command crashes."""

    binary = find_octave_binary(preferred=octave_bin, search_paths=search_paths)
    command = [str(binary), *arguments]
    effective_env = os.environ.copy()
    if env:
        effective_env.update(env)

    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=effective_env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=True,
        )
    except subprocess.TimeoutExpired as error:
        crash_log = latest_octave_crash_log(prefix=binary.name)
        raise OctaveExecutionError(
            f"Octave timed out after {timeout_s:.0f}s.",
            command=command,
            stdout=error.stdout or "",
            stderr=error.stderr or "",
            crash_log=str(crash_log) if crash_log is not None else None,
        ) from error
    except subprocess.CalledProcessError as error:
        crash_log = latest_octave_crash_log(prefix=binary.name)
        message = f"Octave command failed with exit code {error.returncode}."
        if crash_log is not None:
            message = f"{message} Latest crash log: {crash_log}"
        raise OctaveExecutionError(
            message,
            command=command,
            returncode=error.returncode,
            stdout=error.stdout or "",
            stderr=error.stderr or "",
            crash_log=str(crash_log) if crash_log is not None else None,
        ) from error

    return completed
