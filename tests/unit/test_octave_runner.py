from __future__ import annotations

import os
from pathlib import Path

from pyisetcam.octave_runner import find_octave_binary, latest_octave_crash_log


def _make_executable(path: Path) -> None:
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)


def test_find_octave_binary_prefers_explicit_path(tmp_path: Path) -> None:
    explicit = tmp_path / "octave-cli-10.3.0"
    _make_executable(explicit)

    resolved = find_octave_binary(preferred=explicit)

    assert resolved == explicit


def test_find_octave_binary_prefers_versioned_cli_over_wrapper(tmp_path: Path) -> None:
    wrapper = tmp_path / "octave"
    versioned = tmp_path / "octave-cli-10.3.0"
    _make_executable(wrapper)
    _make_executable(versioned)

    resolved = find_octave_binary(search_paths=[tmp_path])

    assert resolved == versioned


def test_latest_octave_crash_log_returns_newest_match(tmp_path: Path) -> None:
    older = tmp_path / "octave-cli-10.3.0-older.ips"
    newer = tmp_path / "octave-cli-10.3.0-newer.ips"
    older.write_text("older")
    newer.write_text("newer")
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    resolved = latest_octave_crash_log(diagnostics_dir=tmp_path, prefix="octave-cli")

    assert resolved == newer
