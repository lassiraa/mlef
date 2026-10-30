"""Shared Git-root and initialization behavior without terminal rendering."""

import subprocess
from pathlib import Path


def project_root(root: Path | str | None = None) -> Path:
    candidate = Path.cwd() if root is None else Path(root).resolve()
    result = subprocess.run(
        ["git", "-C", str(candidate), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if result.returncode:
        raise ValueError("Use a Git project root (initialize Git first).")
    resolved = Path(result.stdout.strip()).resolve()
    if resolved != candidate.resolve():
        raise ValueError(f"Use the project root: {resolved}")
    return resolved


def initialize(root: Path) -> None:
    (root / ".mlef/runs").mkdir(parents=True, exist_ok=True)
    ignore = root / ".gitignore"
    content = ignore.read_text(encoding="utf-8") if ignore.exists() else ""
    if "/.mlef/" not in content.splitlines() and ".mlef/" not in content.splitlines():
        with ignore.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(("\n" if content and not content.endswith("\n") else "") + "/.mlef/\n")
