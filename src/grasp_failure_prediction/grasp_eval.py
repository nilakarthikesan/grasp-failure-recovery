"""One command for validating, running, resuming, and viewing grasp evaluations."""

from __future__ import annotations

import argparse
import json

from grasp_failure_prediction.evaluation.result_viewer import serve_results
from grasp_failure_prediction.proposals.workflow import (
    experiment_output_root,
    experiment_status,
    load_experiment,
    run_experiment,
    validate_experiment,
)


def _add_experiment_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("experiment", help="versioned experiment YAML or JSON")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", help="override the experiment output directory")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grasp-eval",
        description="Generate proposals and run reproducible MuJoCo grasp evaluations.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate an experiment without running it")
    _add_experiment_argument(validate)

    status = commands.add_parser("status", help="show completed and remaining cases")
    _add_experiment_argument(status)

    for name, help_text in (
        ("run", "generate proposals and execute every evaluation"),
        ("resume", "continue an interrupted experiment"),
    ):
        command = commands.add_parser(name, help=help_text)
        _add_experiment_argument(command)
        command.add_argument("--hug-root")
        command.add_argument("--checkpoint")
        command.add_argument("--no-video", action="store_true")

    view = commands.add_parser("view", help="open the experiment results viewer")
    _add_experiment_argument(view)
    view.add_argument("--host", default="127.0.0.1")
    view.add_argument("--port", type=int, default=8765)
    view.add_argument("--no-open", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        result = validate_experiment(args.experiment, project_root=args.project_root)
    elif args.command == "status":
        result = experiment_status(
            args.experiment,
            project_root=args.project_root,
            output_override=args.output,
        )
    elif args.command in {"run", "resume"}:
        result = run_experiment(
            args.experiment,
            project_root=args.project_root,
            output_override=args.output,
            hug_root=args.hug_root,
            checkpoint=args.checkpoint,
            record_video=False if args.no_video else None,
            resume=args.command == "resume",
        )
    else:
        experiment = load_experiment(args.experiment)
        output = experiment_output_root(
            experiment, args.project_root, args.output
        )
        serve_results(
            output / "runs",
            args.host,
            args.port,
            open_browser=not args.no_open,
        )
        return 0
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
