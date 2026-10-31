"""Shared Git-root and initialization behavior without terminal rendering."""

import subprocess
from importlib.resources import files
from pathlib import Path


def project_root(root: Path | str | None = None) -> Path:
    candidate = Path.cwd() if root is None else Path(root).resolve()
    try:
        result = subprocess.run(
            ["git", "-C", str(candidate), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except subprocess.SubprocessError as error:
        raise ValueError("Cannot resolve the Git project root within ten seconds.") from error
    if result.returncode:
        raise ValueError("Use a Git project root (initialize Git first).")
    resolved = Path(result.stdout.strip()).resolve()
    if resolved != candidate.resolve():
        raise ValueError(f"Use the project root: {resolved}")
    return resolved


def initialize(root: Path, *, scaffold: bool = True) -> None:
    (root / ".mlef/runs").mkdir(parents=True, exist_ok=True)
    ignore = root / ".gitignore"
    content = ignore.read_text(encoding="utf-8") if ignore.exists() else ""
    if "/.mlef/" not in content.splitlines() and ".mlef/" not in content.splitlines():
        with ignore.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(("\n" if content and not content.endswith("\n") else "") + "/.mlef/\n")
    if not scaffold:
        return
    definition = root / "mlef.toml"
    steps = root / "steps"
    if any(path.exists() or path.is_symlink() for path in (definition, steps)):
        return
    templates = files("mlef").joinpath("templates")
    steps.mkdir()
    for name in ("ingest", "split", "train", "evaluate"):
        with (steps / f"{name}.py").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(templates.joinpath(f"{name}.py.txt").read_text(encoding="utf-8"))
    with definition.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(templates.joinpath("mlef.toml").read_text(encoding="utf-8"))
