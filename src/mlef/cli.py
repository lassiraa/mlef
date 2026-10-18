"""The initial mlef command line: initialize a project and run a script."""

import argparse
import subprocess
import sys
from pathlib import Path

from . import __version__
from .runner import run


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
    parser = argparse.ArgumentParser(prog="mlef", description="Run Python experiments in Docker.")
    parser.add_argument("--version", action="version", version=f"mlef {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="Initialize local run storage in a Git project.")
    execute = commands.add_parser("run", help="Run a script in the standard Python container.")
    execute.add_argument("script", type=Path)
    execute.add_argument("--include", action="append", default=[], type=Path, metavar="PATH")
    execute.add_argument("--output", action="append", default=[], type=Path, metavar="DIRECTORY")

    # Explicitly split here so script flags can never be interpreted as mlef options.
    arguments = list(sys.argv[1:] if argv is None else argv)
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
        return run(root, args.script, script_arguments, args.include, args.output)
    except ValueError as error:
        print(f"mlef: {error}", file=sys.stderr)
        return 2
    except FileNotFoundError as error:
        print(f"mlef: Required executable or file is missing: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"mlef: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("mlef: Cancelled.", file=sys.stderr)
        return 130
