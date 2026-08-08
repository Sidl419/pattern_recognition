"""CLI: python -m pattern_recognition.experiment run <config.json>"""

from __future__ import annotations

import argparse

from pattern_recognition.experiment.runner import run_experiment


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pattern_recognition.experiment")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run", help="Run an experiment from a JSON config")
    run_p.add_argument("config", help="Path to experiment JSON config")
    args = parser.parse_args(argv)
    if args.cmd == "run":
        path = run_experiment(args.config)
        print(path)


if __name__ == "__main__":
    main()
