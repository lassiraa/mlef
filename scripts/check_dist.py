"""Install and exercise wheel and sdist in separate environments outside the checkout."""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def check(distribution: Path, temporary: Path) -> None:
    environment = temporary / "environment"
    subprocess.run(["uv", "venv", "--python", sys.executable, str(environment)], check=True)
    binaries = environment / ("Scripts" if os.name == "nt" else "bin")
    python = binaries / ("python.exe" if os.name == "nt" else "python")
    executable = binaries / ("mlef.exe" if os.name == "nt" else "mlef")
    subprocess.run(["uv", "pip", "install", "--python", str(python), str(distribution)], check=True)
    subprocess.run([str(executable), "--version"], cwd=temporary, check=True)
    subprocess.run(
        [
            str(python),
            "-c",
            "from importlib.metadata import distribution; import mlef; "
            "d=distribution('mlef'); assert d.version == mlef.__version__; "
            "assert any(str(p).endswith('LICENSE') for p in d.files); "
            "assert 'site-packages' in mlef.__file__",
        ],
        cwd=temporary,
        check=True,
    )
    project = temporary / "project"
    project.mkdir()
    subprocess.run(["git", "init", "--quiet", str(project)], check=True)
    subprocess.run([str(executable), "init"], cwd=project, check=True)
    result = subprocess.run(
        [str(executable), "list", "--json"], cwd=project, capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]", result.stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    distributions = sorted([*args.directory.glob("*.whl"), *args.directory.glob("*.tar.gz")])
    if len(distributions) != 2:
        raise ValueError("Expected exactly one wheel and one sdist in the distribution directory.")
    for distribution in distributions:
        with tempfile.TemporaryDirectory(prefix="mlef-install-") as temporary:
            check(distribution.resolve(), Path(temporary))
        print(f"Verified {distribution.name}")


if __name__ == "__main__":
    main()
