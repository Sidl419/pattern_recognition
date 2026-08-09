"""CLI: python -m pattern_recognition.speller run --config ... [--run-dir ...]"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pattern_recognition.speller.benchmark import run_speller_benchmark


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pattern_recognition.speller")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run", help="Run speller benchmark from a JSON config")
    run_p.add_argument(
        "--config",
        required=True,
        help="Path to speller benchmark JSON config",
    )
    run_p.add_argument(
        "--run-dir",
        help="Override run_dir in the speller config (binary experiment directory)",
    )
    args = parser.parse_args(argv)
    if args.cmd == "run":
        cfg_path = Path(args.config)
        payload = json.loads(cfg_path.read_text())
        if args.run_dir is not None:
            payload["run_dir"] = args.run_dir
        out = run_speller_benchmark(payload)
        print(out)


if __name__ == "__main__":
    main()
