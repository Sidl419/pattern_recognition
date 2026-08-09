import subprocess
import sys


def test_speller_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "pattern_recognition.speller", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "run" in result.stdout


def test_speller_run_help():
    result = subprocess.run(
        [sys.executable, "-m", "pattern_recognition.speller", "run", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--config" in result.stdout
    assert "--run-dir" in result.stdout
