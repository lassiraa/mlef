"""A foreground Docker runner using a durable source snapshot."""

import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
from datetime import UTC, datetime
from io import BufferedReader
from pathlib import Path
from typing import BinaryIO, TextIO
from uuid import uuid4

from .provenance import capture, extract_source

BASE_IMAGE = "python:3.12-slim"


def now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def save(path: Path, data: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def relative_path(root: Path, path: Path) -> Path:
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ValueError(f"Expected a project-relative path without '..': {path}")
    if not (root / path).resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes the project: {path}")
    if any(part.lower() in {".git", ".mlef"} for part in path.parts):
        raise ValueError(f"Cannot include Git internals or run storage: {path}")
    return path


def source_files(root: Path, script: Path, includes: list[Path]) -> list[Path]:
    tracked = (
        subprocess.run(
            ["git", "ls-files", "--cached", "-z"], cwd=root, capture_output=True, check=True
        )
        .stdout.decode("utf-8")
        .split("\0")
    )
    files = {Path(name) for name in tracked if name}
    for path in [script, *includes]:
        relative_path(root, path)
        source = root / path
        if source.is_symlink():
            raise ValueError(f"Symbolic links are not supported in source inputs yet: {path}")
        if not source.exists():
            raise ValueError(f"Source path does not exist: {path}")
        if source.is_dir():
            files.update(item.relative_to(root) for item in source.rglob("*") if item.is_file())
        else:
            files.add(path)
    selected = []
    for path in sorted(files):
        relative_path(root, path)
        if any(
            part.lower()
            in {
                ".venv",
                "venv",
                "__pycache__",
                ".pytest_cache",
                ".ruff_cache",
                ".mypy_cache",
                ".tox",
                ".ssh",
                ".aws",
                ".azure",
            }
            or part.lower() == ".env"
            or part.lower().startswith(".env.")
            or part.lower().endswith((".pem", ".key"))
            for part in path.parts
        ):
            raise ValueError(f"Excluded credential or generated path in source selection: {path}")
        source = root / path
        if any(parent.is_symlink() for parent in [source, *source.parents] if parent != root):
            raise ValueError(f"Symbolic links are not supported in source inputs yet: {path}")
        if source.is_file():
            selected.append(path)
    return selected


def docker(
    arguments: list[str], log: BinaryIO, *, check: bool = True
) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(
            ["docker", *arguments],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=600 if arguments[0] == "build" else 30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        if check:
            raise
        log.write((str(error) + "\n").encode("utf-8"))
        result = subprocess.CompletedProcess[bytes](["docker", *arguments], 1)
    log.flush()
    if check and result.returncode:
        raise RuntimeError(
            f"docker {arguments[0]} failed (exit {result.returncode}); see runner.log"
        )
    return result


def stream_process(
    process: subprocess.Popen[bytes], run_dir: Path, name: str, log: BinaryIO
) -> int:
    errors: list[OSError] = []

    def copy(source: BufferedReader, path: Path, terminal: TextIO) -> None:
        try:
            with path.open("wb") as destination:
                while chunk := source.read1(65536):
                    destination.write(chunk)
                    destination.flush()
                    try:
                        if hasattr(terminal, "buffer"):
                            terminal.buffer.write(chunk)
                            terminal.buffer.flush()
                        else:
                            terminal.write(chunk.decode("utf-8", errors="replace"))
                            terminal.flush()
                    except (OSError, ValueError):
                        pass  # Saved logs remain available if a terminal consumer disconnects.
        except OSError as error:
            errors.append(error)
        finally:
            source.close()

    threads = [
        threading.Thread(target=copy, args=(process.stdout, run_dir / "stdout.log", sys.stdout)),
        threading.Thread(target=copy, args=(process.stderr, run_dir / "stderr.log", sys.stderr)),
    ]
    for thread in threads:
        thread.start()
    try:
        return_code = process.wait()
    except BaseException:
        docker(["stop", "--time", "5", name], log, check=False)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    finally:
        for thread in threads:
            thread.join()
    if errors:
        raise OSError(f"Cannot save experiment logs: {errors[0]}")
    return return_code


def run(
    root: Path, script: Path, arguments: list[str], includes: list[Path], outputs: list[Path]
) -> int:
    if not (root / ".mlef" / "runs").is_dir():
        raise ValueError("Project is not initialized; run mlef init first.")
    relative_path(root, script)
    if not (root / script).is_file():
        raise ValueError(f"Script does not exist or is not a file: {script}")
    for output in outputs:
        relative_path(root, output)
    files = source_files(root, script, includes)
    run_id = str(uuid4())
    run_dir = root / ".mlef" / "runs" / run_id
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True)
    for filename in ["stdout.log", "stderr.log", "runner.log"]:
        (run_dir / filename).touch()
    command = ["python", script.as_posix(), *arguments]
    warnings = ["Environment provenance and locked dependencies are not captured yet."]
    metadata: dict[str, object] = {
        "schema_version": 1,
        "id": run_id,
        "state": "preparing",
        "created_at": now(),
        "started_at": None,
        "finished_at": None,
        "command": command,
        "execution_exit_code": None,
        "error": None,
        "warnings": warnings,
        "outputs": [path.as_posix() for path in outputs],
        "git": None,
        "inputs": None,
    }
    save(run_dir / "run.json", metadata)
    print(f"Run ID: {run_id}\nResults: {run_dir}", flush=True)
    name = f"mlef-{run_id}"
    image = f"mlef-run:{run_id}"
    process = None
    created = False
    cancelled = False
    result = 1
    with (run_dir / "runner.log").open("ab") as log:
        try:
            git_provenance, inputs = capture(root, run_dir, files)
            metadata.update(git=git_provenance.model_dump(), inputs=inputs.model_dump())
            save(run_dir / "run.json", metadata)
            if shutil.which("docker") is None:
                raise RuntimeError(
                    "Docker executable not found. Install Docker and start its engine."
                )
            docker(["info"], log)
            with tempfile.TemporaryDirectory(prefix="build-", dir=run_dir) as temporary:
                context = Path(temporary)
                workspace = context / "workspace"
                workspace.mkdir()
                extract_source(run_dir / "source.tar.gz", workspace)
                user = None
                if hasattr(os, "getuid") and hasattr(os, "getgid"):
                    user = f"{os.getuid()}:{os.getgid()}"
                ownership = f"--chown={user} " if user is not None else ""
                workspace_owner = f"RUN chown {user} /workspace\n" if user is not None else ""
                (context / "Dockerfile").write_text(
                    f"FROM {BASE_IMAGE}\nWORKDIR /workspace\n"
                    f"COPY {ownership}workspace/ /workspace/\n{workspace_owner}",
                    encoding="utf-8",
                )
                shutil.copy2(context / "Dockerfile", run_dir / "Dockerfile")
                docker(["build", "--tag", image, str(context)], log)
            create = [
                "create",
                "--name",
                name,
                "--init",
                "--workdir",
                "/workspace",
                "--env",
                "MLEF_OUTPUT_DIR=/outputs",
                "--env",
                "PYTHONUNBUFFERED=1",
                "--volume",
                f"{artifacts}:/outputs",
            ]
            if user is not None:
                create.extend(["--user", user])
            docker([*create, image, *command], log)
            created = True
            # Do not allocate a TTY: Docker then preserves separate stdout/stderr streams.
            # Handle Ctrl+C in mlef; do not also deliver it to the attached Docker client.
            process = subprocess.Popen(
                ["docker", "start", "--attach", name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
                start_new_session=sys.platform != "win32",
            )
            metadata.update(state="running", started_at=now())
            save(run_dir / "run.json", metadata)
            attach_exit_code = stream_process(process, run_dir, name, log)
            inspection = subprocess.run(
                ["docker", "inspect", "--format", "{{json .State}}", name],
                capture_output=True,
                text=True,
                check=True,
            )
            state = json.loads(inspection.stdout)
            if state["Running"] or not state.get("StartedAt", "").startswith("20"):
                raise RuntimeError("Docker did not report a completed script execution.")
            if state.get("Error"):
                raise RuntimeError(f"Container execution failed: {state['Error']}")
            metadata["execution_exit_code"] = state["ExitCode"]
            if state["ExitCode"]:
                raise RuntimeError(f"Script exited with code {state['ExitCode']}; see stderr.log.")
            if attach_exit_code:
                raise RuntimeError("Docker attach failed; experiment logs may be incomplete.")
            result = 0
        except KeyboardInterrupt:
            cancelled = True
            metadata["error"] = {"category": "cancelled", "message": "Cancelled by user."}
            if created:
                docker(["stop", "--time", "5", name], log, check=False)
            result = 130
        except (
            OSError,
            RuntimeError,
            subprocess.SubprocessError,
            ValueError,
            tarfile.TarError,
        ) as error:
            metadata["error"] = {"category": "execution", "message": str(error)}
            log.write((str(error) + "\n").encode("utf-8"))
            print(f"mlef: {error}", file=sys.stderr)
        finally:
            if process is not None and process.poll() is None:
                docker(["stop", "--time", "5", name], log, check=False)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if created:
                for output in outputs:
                    try:
                        destination = artifacts / output
                        if not destination.resolve().is_relative_to(artifacts.resolve()):
                            raise RuntimeError(
                                f"Output destination escapes artifact storage: {output}"
                            )
                        if destination.exists() or destination.is_symlink():
                            raise RuntimeError(f"Output destination already exists: {output}")
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        docker(
                            ["cp", f"{name}:/workspace/{output.as_posix()}", str(destination)], log
                        )
                    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                        result = 130 if cancelled else 1
                        message = f"Could not collect output {output}: {error}"
                        metadata["error"] = {
                            "category": "collection",
                            "message": message,
                        }
                        print(f"mlef: {message}", file=sys.stderr)
                if docker(["rm", "--force", name], log, check=False).returncode:
                    warnings.append(f"Container cleanup failed: {name}")
            if (
                shutil.which("docker") is not None
                and docker(["image", "rm", image], log, check=False).returncode
            ):
                warnings.append(f"Image cleanup failed or image was not built: {image}")
            metadata.update(
                state="cancelled" if cancelled else "succeeded" if result == 0 else "failed",
                finished_at=now(),
            )
            save(run_dir / "run.json", metadata)
    print(f"Run {run_id}: {metadata['state']}", flush=True)
    return result
