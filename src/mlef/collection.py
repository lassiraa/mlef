"""One destination policy for normal collection, timeout cleanup and recovery."""

from pathlib import Path


def output_destination(run_dir: Path, relative: str) -> Path:
    base = run_dir / "artifacts"
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path == Path(".") or ":" in relative:
        raise RuntimeError(f"Invalid declared output path: {relative}")
    target = base / path
    if base.is_symlink() or not target.resolve().is_relative_to(base.resolve()):
        raise RuntimeError(f"Output destination escapes artifact storage: {relative}")
    if target.exists() or target.is_symlink():
        raise RuntimeError(f"Output destination already exists: {relative}")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target
