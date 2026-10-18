"""A foreground Docker runner. Provenance and locked dependencies follow in step 3."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

BASE_IMAGE = "python:3.12-slim"


def now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def save(path: Path, data: dict) -> None:
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
            part.lower() in {".venv", "__pycache__", ".ssh"}
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


def docker(arguments: list[str], log, *, check: bool = True) -> subprocess.CompletedProcess:
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
        result = subprocess.CompletedProcess(["docker", *arguments], 1)
    log.flush()
    if check and result.returncode:
        raise RuntimeError(
            f"docker {arguments[0]} failed (exit {result.returncode}); see runner.log"
        )
    return result


def stream_process(process: subprocess.Popen, run_dir: Path, name: str, log) -> int:
    errors = []

    def copy(source, path: Path, terminal) -> None:
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
    metadata = {
        "schema_version": 1,
        "id": run_id,
        "state": "preparing",
        "created_at": now(),
        "started_at": None,
        "finished_at": None,
        "command": ["python", script.as_posix(), *arguments],
        "execution_exit_code": None,
        "error": None,
        "warnings": [
            "Step 2: Git/environment provenance and locked dependencies are not captured yet."
        ],
        "outputs": [path.as_posix() for path in outputs],
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
            if not shutil.which("docker"):
                raise RuntimeError(
                    "Docker executable not found. Install Docker and start its engine."
                )
            docker(["info"], log)
            with tempfile.TemporaryDirectory(prefix="build-", dir=run_dir) as temporary:
                context = Path(temporary)
                workspace = context / "workspace"
                workspace.mkdir()
                for path in files:
                    destination = workspace / path
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(root / path, destination)
                ownership = f"--chown={os.getuid()}:{os.getgid()} " if hasattr(os, "getuid") else ""
                workspace_owner = (
                    f"RUN chown {os.getuid()}:{os.getgid()} /workspace\n"
                    if hasattr(os, "getuid")
                    else ""
                )
                (context / "Dockerfile").write_text(
                    f"FROM {BASE_IMAGE}\nWORKDIR /workspace\n"
                    f"COPY {ownership}workspace/ /workspace/\n{workspace_owner}",
                    encoding="utf-8",
                )
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
            if hasattr(os, "getuid"):
                create.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
            docker([*create, image, *metadata["command"]], log)
            created = True
            # Do not allocate a TTY: Docker then preserves separate stdout/stderr streams.
            # Handle Ctrl+C in mlef; do not also deliver it to the attached Docker client.
            process_options = (
                {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                if sys.platform == "win32"
                else {"start_new_session": True}
            )
            process = subprocess.Popen(
                ["docker", "start", "--attach", name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                **process_options,
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
        except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
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
                        metadata["error"] = {
                            "category": "collection",
                            "message": f"Could not collect output {output}: {error}",
                        }
                        print(f"mlef: {metadata['error']['message']}", file=sys.stderr)
                if docker(["rm", "--force", name], log, check=False).returncode:
                    metadata["warnings"].append(f"Container cleanup failed: {name}")
            if shutil.which("docker"):
                if docker(["image", "rm", image], log, check=False).returncode:
                    metadata["warnings"].append(
                        f"Image cleanup failed or image was not built: {image}"
                    )
            metadata.update(
                state="cancelled" if cancelled else "succeeded" if result == 0 else "failed",
                finished_at=now(),
            )
            save(run_dir / "run.json", metadata)
    print(f"Run {run_id}: {metadata['state']}", flush=True)
    return result
