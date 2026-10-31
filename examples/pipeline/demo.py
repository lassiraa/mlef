"""Run the generated four-step pipeline in a new Git project using the host SDK."""

import argparse
import shutil
import subprocess
from pathlib import Path

from mlef import api


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", type=Path, help="New project directory; must not already exist."
    )
    args = parser.parse_args()
    root = args.directory.absolute()
    root.mkdir(parents=True, exist_ok=False)
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    api.init(root=root)
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(Path(__file__).parent / name, root / name)
    result = api.run(root=root, options=api.RunOptions(timeout_seconds=180))
    print(f"Run {result.id}: {result.state}")
    if result.state != "succeeded":
        raise SystemExit(result.error.message if result.error is not None else "Pipeline failed.")
    print(api.pipeline_status(result.id, root=root, verify=True).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
