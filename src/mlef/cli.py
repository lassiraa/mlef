"""The initial mlef command line: initialize a project and run a script."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

from typing_extensions import override

from . import __version__
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
    for name in ["list", "status", "get"]:
        query = commands.add_parser(name, help=f"{name.capitalize()} saved local runs/artifacts.")
        if name != "list":
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
            if args.json:
                print(record.model_dump_json())
            else:
                print(f"{record.id}: {record.state}\nCommand: {record.command}")
                print(f"Created: {record.created_at}\nStarted: {record.started_at}")
                print(f"Finished: {record.finished_at}")
                print(f"Execution exit code: {record.execution_exit_code}")
                if record.error is not None:
                    print(f"Error ({record.error.category}): {record.error.message}")
                for warning in record.warnings:
                    print(f"Warning: {warning}")
            return 0
        if args.command == "get":
            if args.artifact is None:
                if args.output is not None:
                    parser.error("--output requires an artifact path.")
                print("\n".join(artifacts(root, args.run_id)))
            else:
                print(retrieve(root, args.run_id, args.artifact, args.output))
            return 0
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
