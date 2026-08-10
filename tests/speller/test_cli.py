"""Speller CLI smoke."""

import subprocess
import sys


def test_speller_cli_help():
    root = subprocess.run(
        [sys.executable, "-m", "pattern_recognition.speller", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert root.returncode == 0
    assert "run" in root.stdout

    run_help = subprocess.run(
        [sys.executable, "-m", "pattern_recognition.speller", "run", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_help.returncode == 0
    assert "--config" in run_help.stdout
    assert "--run-dir" in run_help.stdout
