"""The initial mlef command line: initialize a project and run a script."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

from typing_extensions import override

from . import __version__
from .comparison import compare
from .evidence import read_result
from .runner import run
from .storage import QueryError, artifacts, inspect, list_runs, retrieve


def project_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise ValueError("Run mlef from the root of a Git repository (use git init first).")
    root = Path(result.stdout.strip()).resolve()
    if root != Path.cwd().resolve():
        raise ValueError(f"Run mlef from the project root: {root}")
    return root


def init(root: Path) -> None:
    (root / ".mlef" / "runs").mkdir(parents=True, exist_ok=True)
    ignore = root / ".gitignore"
    content = ignore.read_text(encoding="utf-8") if ignore.exists() else ""
    if "/.mlef/" not in content.splitlines() and ".mlef/" not in content.splitlines():
        with ignore.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(("\n" if content and not content.endswith("\n") else "") + "/.mlef/\n")
    print(f"Initialized mlef storage: {root / '.mlef'}")


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    json_requested = (
        "--json" in arguments[: arguments.index("--")]
        if "--" in arguments
        else "--json" in arguments
    )

    class Parser(argparse.ArgumentParser):
        @override
        def error(self, message: str) -> NoReturn:
            if json_requested:
                print(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "error": {"category": "arguments", "message": message},
                        }
                    )
                )
            super().error(message)

    parser = Parser(prog="mlef", description="Run Python experiments in Docker.")
    parser.add_argument("--version", action="version", version=f"mlef {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="Initialize local run storage in a Git project.")
    execute = commands.add_parser("run", help="Run a script in the standard Python container.")
    execute.add_argument("script", type=Path)
    execute.add_argument("--include", action="append", default=[], type=Path, metavar="PATH")
    execute.add_argument("--output", action="append", default=[], type=Path, metavar="DIRECTORY")
    execute.add_argument("--experiment", type=Path, metavar="JSON_FILE")
    execute.add_argument("--timeout", type=float, metavar="SECONDS")
    for name in ["list", "status", "get", "compare"]:
        query = commands.add_parser(name, help=f"{name.capitalize()} saved local runs/artifacts.")
        if name == "compare":
            query.add_argument("baseline")
            query.add_argument("candidate")
        elif name != "list":
            query.add_argument("run_id")
        if name == "get":
            query.add_argument("artifact", nargs="?")
            query.add_argument("--output", type=Path)
        else:
            query.add_argument("--json", action="store_true")

    # Explicitly split here so script flags can never be interpreted as mlef options.
    script_arguments = []
    if "--" in arguments:
        boundary = arguments.index("--")
        script_arguments = arguments[boundary + 1 :]
        arguments = arguments[:boundary]
    args = parser.parse_args(arguments)
    if args.command != "run" and script_arguments:
        parser.error("Only mlef run accepts script arguments after --.")
    try:
        root = project_root()
        if args.command == "init":
            init(root)
            return 0
        if args.command == "list":
            records, diagnostics = list_runs(root)
            for diagnostic in diagnostics:
                print(f"mlef: {diagnostic}", file=sys.stderr)
            if args.json:
                print(json.dumps([record.model_dump(mode="json") for record in records]))
            else:
                for record in records:
                    print(f"{record.id}  {record.state}  {record.created_at}  {record.command}")
            return 0
        if args.command == "status":
            record = inspect(root, args.run_id)
            evidence = read_result(root, args.run_id)
            if args.json:
                print(
                    json.dumps(
                        {
                            **record.model_dump(mode="json"),
                            "result": evidence.model_dump(mode="json"),
                        }
                    )
                )
            else:
                print(f"{record.id}: {record.state}\nCommand: {record.command}")
                print(f"Created: {record.created_at}\nStarted: {record.started_at}")
                print(f"Finished: {record.finished_at}")
                print(f"Execution exit code: {record.execution_exit_code}")
                if record.error is not None:
                    print(f"Error ({record.error.category}): {record.error.message}")
                for warning in record.warnings:
                    print(f"Warning: {warning}")
                print(f"Evidence: {evidence.evidence_status}")
                for name, value in list(evidence.metrics.items())[:20]:
                    print(f"  {name}: {value}")
                if len(evidence.metrics) > 20:
                    print(f"  {len(evidence.metrics) - 20} metrics omitted; use --json.")
                for diagnostic in evidence.diagnostics:
                    print(f"{diagnostic.code}: {diagnostic.message}")
                print(f"Artifacts: .mlef/runs/{record.id}/artifacts/")
            return 0
        if args.command == "compare":
            comparison = compare(root, args.baseline, args.candidate)
            if args.json:
                print(comparison.model_dump_json())
            else:
                print(
                    f"{comparison.baseline_id} -> {comparison.candidate_id}: "
                    f"{comparison.compatibility}"
                )
                print(
                    f"Execution seconds: {comparison.baseline_seconds} -> "
                    f"{comparison.candidate_seconds}"
                )
                for metric in comparison.metrics[:20]:
                    print(
                        f"{metric.name}: {metric.baseline} -> {metric.candidate}; "
                        f"delta={metric.delta}; relative={metric.relative_delta}; "
                        f"{metric.compatibility}; {metric.observation}"
                    )
                    for reason in metric.reasons:
                        print(f"  {reason}")
                if len(comparison.metrics) > 20:
                    print(f"{len(comparison.metrics) - 20} metrics omitted; use --json.")
                print(
                    f"Changed parameters: {len(comparison.parameters)}; "
                    f"provenance fields: {list(comparison.provenance)}"
                )
                for name, change in list(comparison.parameters.items())[:20]:
                    print(f"  {name}: {change.baseline!r} -> {change.candidate!r}")
                if len(comparison.parameters) > 20:
                    print(
                        f"{len(comparison.parameters) - 20} parameter changes omitted; use --json."
                    )
                print(
                    f"Artifacts: .mlef/runs/{args.baseline}/artifacts/ "
                    f"and .mlef/runs/{args.candidate}/artifacts/"
                )
                for limitation in comparison.limitations:
                    print(limitation)
            return 0
        if args.command == "get":
            if args.artifact is None:
                if args.output is not None:
                    parser.error("--output requires an artifact path.")
                print("\n".join(artifacts(root, args.run_id)))
            else:
                print(retrieve(root, args.run_id, args.artifact, args.output))
            return 0
        if args.timeout is not None:
            return run(
                root,
                args.script,
                script_arguments,
                args.include,
                args.output,
                args.experiment,
                timeout=args.timeout,
            )
        if args.experiment is not None:
            return run(
                root, args.script, script_arguments, args.include, args.output, args.experiment
            )
        return run(root, args.script, script_arguments, args.include, args.output)
    except QueryError as error:
        if getattr(args, "json", False):
            print(
                json.dumps(
                    {
                        "schema_version": 1,
                        "error": {"category": error.category, "message": str(error)},
                    }
                )
            )
        print(f"mlef: {error}", file=sys.stderr)
        return 1
    except ValueError as error:
        if getattr(args, "json", False):
            print(
                json.dumps(
                    {
                        "schema_version": 1,
                        "error": {"category": "configuration", "message": str(error)},
                    }
                )
            )
        print(f"mlef: {error}", file=sys.stderr)
        return 2
    except FileNotFoundError as error:
        if getattr(args, "json", False):
            print(
                json.dumps(
                    {"schema_version": 1, "error": {"category": "storage", "message": str(error)}}
                )
            )
        print(f"mlef: Required executable or file is missing: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        if getattr(args, "json", False):
            print(
                json.dumps(
                    {"schema_version": 1, "error": {"category": "storage", "message": str(error)}}
                )
            )
        print(f"mlef: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("mlef: Cancelled.", file=sys.stderr)
        return 130
