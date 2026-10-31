"""A foreground Docker runner using a durable source snapshot."""

import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from contextlib import suppress
from datetime import UTC, datetime
from io import BufferedReader
from pathlib import Path
from typing import BinaryIO, TextIO
from uuid import uuid4

from pydantic import BaseModel, Field, JsonValue, TypeAdapter

from .collection import output_destination
from .environment import (
    UV_IMAGE,
    Environment,
    build_recipe,
    container_info,
    dependency_inputs,
    file_hash,
    image_identity,
    query,
)
from .evidence import experiment_spec, normalize
from .execution import interrupt_pipeline, prepare_pipeline, read_pipeline
from .limits import Limits, memory_bytes
from .locking import run_lock
from .pipeline import validate
from .processes import BoundedLog, Budget, LogLimitError
from .processes import command as run_command
from .provenance import capture, extract_source
from .records import Controls, RunError, RunRecord
from .sdk_distribution import capture_sdk

BASE_IMAGE = "python:3.12-slim"


class CompletedState(BaseModel):
    running: bool = Field(alias="Running")
    started_at: str = Field(alias="StartedAt")
    finished_at: str = Field(alias="FinishedAt")
    exit_code: int = Field(alias="ExitCode")
    oom_killed: bool = Field(default=False, alias="OOMKilled")
    error: str = Field(default="", alias="Error")


def completed_state(name: str) -> CompletedState:
    """Docker's OOM event can arrive just after attach closes; require affirmative evidence."""
    deadline = time.monotonic() + 2
    while True:
        inspection = subprocess.run(
            ["docker", "inspect", "--format", "{{json .State}}", name],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        state = CompletedState.model_validate_json(inspection.stdout)
        if state.exit_code != 137 or state.oom_killed or time.monotonic() >= deadline:
            return state
        time.sleep(0.05)


def now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def save(path: Path, data: dict[str, object] | BaseModel) -> None:
    temporary = path.with_suffix(".tmp")
    values = data.model_dump(mode="json") if isinstance(data, BaseModel) else data
    temporary.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n", encoding="utf-8")
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
    dependency_files = [
        path.relative_to(root)
        for path in root.iterdir()
        if path.is_file()
        and (
            path.name
            in {
                "pyproject.toml",
                "uv.lock",
                "uv.toml",
                "Pipfile",
                "Pipfile.lock",
                "poetry.lock",
                "pdm.lock",
                "setup.py",
                "setup.cfg",
            }
            or (path.name.startswith("requirements") and path.suffix in {".txt", ".in"})
        )
    ]
    for path in [script, *includes, *dependency_files]:
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
        result = run_command(
            ["docker", *arguments],
            log if check else None,
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
    process: subprocess.Popen[bytes],
    run_dir: Path,
    name: str,
    log: BinaryIO,
    budget: Budget | None = None,
) -> int:
    errors: list[OSError | LogLimitError] = []
    budget = budget if budget is not None else Budget(None)

    def copy(source: BufferedReader, path: Path, terminal: TextIO) -> None:
        try:
            with path.open("wb") as destination:
                while chunk := source.read1(65536):
                    chunk = budget.take(chunk)
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
                    if budget.exceeded.is_set():
                        raise LogLimitError("Combined stdout/stderr byte budget exceeded.")
        except (OSError, LogLimitError) as error:
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
        while True:
            if errors:
                raise errors[0]
            try:
                return_code = process.wait(timeout=0.05)
                break
            except subprocess.TimeoutExpired:
                pass
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
        raise errors[0]
    return return_code


def run(
    root: Path,
    script: Path,
    arguments: list[str],
    includes: list[Path],
    outputs: list[Path],
    experiment: Path | None = None,
    timeout: float | None = None,
    cpus: float | None = None,
    memory: str | int | None = None,
    network: str = "default",
    max_log_bytes: int | None = None,
    max_runner_log_bytes: int | None = None,
    *,
    reuse_from: str | None = None,
    from_step: str | None = None,
    resume: bool = False,
    cache_from: str | None = None,
    force: list[str] | None = None,
    _identifier: str | None = None,
    _token: str | None = None,
) -> int:
    from .supervision import Request, identity, supervise

    limits = Limits.model_validate(
        {
            "cpus": cpus,
            "memory_bytes": memory_bytes(memory) if isinstance(memory, str) else memory,
            "network": network,
        }
    )
    script_budget = Budget(max_log_bytes)
    Budget(max_runner_log_bytes)
    pipeline = script.suffix == ".toml"
    force = force or []
    if (cache_from is not None and (not pipeline or reuse_from is not None or resume)) or (
        force and cache_from is None
    ):
        raise ValueError(
            "Cache/force options require a pipeline cache candidate, not explicit reuse."
        )
    if (reuse_from is None) != (from_step is None) or (resume and reuse_from is None):
        raise ValueError("Use --reuse-from and --from-step together; --resume requires both.")
    if cache_from is not None:
        from .records import run_id as validate_run_id

        validate_run_id(cache_from)
    if reuse_from is not None and not pipeline:
        raise ValueError("Artifact reuse is only supported for pipelines.")
    if resume and includes:
        raise ValueError("Resume uses the captured snapshot and does not accept --include.")
    if pipeline and (arguments or outputs or experiment is not None):
        raise ValueError("Pipelines do not accept script arguments, --output or --experiment.")
    if _identifier is None and (
        pipeline
        or any(value is not None for value in [timeout, max_log_bytes, max_runner_log_bytes])
    ):
        return supervise(
            Request(
                root=str(root.resolve()),
                script=str(script),
                arguments=arguments,
                includes=[str(path) for path in includes],
                outputs=[str(path) for path in outputs],
                experiment=str(experiment.resolve()) if experiment is not None else None,
                timeout=timeout,
                identifier=str(uuid4()),
                token=str(uuid4()),
                limits=limits,
                max_log_bytes=max_log_bytes,
                max_runner_log_bytes=max_runner_log_bytes,
                reuse_from=reuse_from,
                from_step=from_step,
                resume=resume,
                cache_from=cache_from,
                force=force,
            )
        )
    if not (root / ".mlef" / "runs").is_dir():
        raise ValueError("Project is not initialized; run mlef init first.")
    if resume:
        from .reuse import verify_source

        assert reuse_from is not None
        script = Path(verify_source(root, reuse_from).definition_path)
    relative_path(root, script)
    if not resume and not (root / script).is_file():
        raise ValueError(f"Script does not exist or is not a file: {script}")
    for output in outputs:
        relative_path(root, output)
    if pipeline and not resume:
        report = validate(root, script)
        if report.status != "verified" or report.definition is None:
            raise ValueError("Pipeline must validate before execution: " + report.model_dump_json())
        includes = [
            *includes,
            *(Path(entry.script) for entry in report.definition.steps),
            *(
                Path(entry.contract)
                for entry in report.definition.steps
                if entry.contract is not None
            ),
        ]
    files = source_files(root, script, includes) if not resume else []
    specification = experiment_spec(root, experiment) if experiment is not None else None
    run_id = _identifier if _identifier is not None else str(uuid4())
    run_dir = root / ".mlef" / "runs" / run_id
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True)
    for filename in ["stdout.log", "stderr.log", "runner.log"]:
        (run_dir / filename).touch()
    command = (
        ["python", "-I", "-m", "mlef.execution", "/outputs/pipeline"]
        if pipeline
        else ["python", script.as_posix(), *arguments]
    )
    warnings = ["Environment capture is incomplete."]
    metadata = RunRecord(
        id=run_id,
        state="preparing",
        created_at=now(),
        command=command,
        warnings=warnings,
        outputs=[path.as_posix() for path in outputs],
        experiment=specification,
        runtime=identity(root, run_id, _token if _token is not None else str(uuid4())),
        controls=Controls(
            requested={
                "timeout_seconds": timeout,
                **limits.model_dump(mode="json"),
                "max_log_bytes": max_log_bytes,
                "max_runner_log_bytes": max_runner_log_bytes,
            },
            applied={
                "supervised_deadline": timeout is not None,
                "script_log_budget": max_log_bytes is not None,
                "runner_log_budget": max_runner_log_bytes is not None,
            },
        ),
        stage="source_selection",
    )
    with run_lock(run_dir):
        warnings = metadata.warnings
        save(run_dir / "run.json", metadata)
        print(f"Run ID: {run_id}\nResults: {run_dir}", flush=True)
        name = f"mlef-{run_id}"
        image = f"mlef-run:{run_id}"
        process = None
        created = False
        cancelled = False
        result = 1
        environment = Environment()
        save(run_dir / "environment.json", environment.model_dump())
        with BoundedLog(str(run_dir / "runner.log"), max_runner_log_bytes) as log:

            def stage(value: str) -> None:
                metadata.stage = value
                save(run_dir / "run.json", metadata)

            try:
                stage("source_capture")
                if resume:
                    from . import storage
                    from .reuse import copy_snapshot

                    assert reuse_from is not None
                    copy_snapshot(root, reuse_from, run_dir)
                    original = storage.inspect(root, reuse_from)
                    assert original.inputs is not None
                    git_provenance = original.git
                    inputs = original.inputs.model_copy(
                        deep=True, update={"build_recipe_sha256": None, "dependency_hashes": {}}
                    )
                else:
                    git_provenance, inputs = capture(root, run_dir, files)
                metadata.git, metadata.inputs = git_provenance, inputs
                inputs.outputs = metadata.outputs
                save(run_dir / "run.json", metadata)
                with tempfile.TemporaryDirectory(prefix="build-", dir=run_dir) as temporary:
                    context = Path(temporary)
                    workspace = context / "workspace"
                    workspace.mkdir()
                    extract_source(run_dir / "source.tar.gz", workspace)
                    stage("dependencies")
                    dependencies = dependency_inputs(workspace)
                    environment.dependencies = dependencies
                    if pipeline:
                        if dependencies.install_command is None:
                            raise ValueError(
                                "Pipelines require pyproject.toml and uv.lock with the SDK runtime "
                                "dependencies. Use uv add 'pydantic>=2,<3' "
                                "'typing-extensions>=4.12'."
                            )
                        stage("sdk_capture")
                        if resume:
                            from .reuse import verify_source

                            assert reuse_from is not None
                            environment.sdk_wheel = verify_source(root, reuse_from).sdk_wheel
                        else:
                            environment.sdk_wheel = capture_sdk(run_dir / "sdk")
                        shutil.copytree(run_dir / "sdk", context / "sdk")
                        prepared = prepare_pipeline(
                            workspace,
                            script,
                            artifacts / "pipeline",
                            run_id,
                            inputs.source_sha256,
                            environment.sdk_wheel,
                        )
                        from .external_inputs import stage as stage_external_inputs

                        stage_external_inputs(
                            root,
                            artifacts / "pipeline",
                            resume_from=reuse_from if resume else None,
                        )
                        if set(force) - set(prepared.order):
                            raise ValueError("Forced steps must name known pipeline steps.")
                        if reuse_from is not None:
                            from .reuse import reuse_steps

                            assert from_step is not None
                            reuse_steps(
                                root, reuse_from, artifacts / "pipeline", from_step, resume=resume
                            )
                        stage("dependencies")
                    inputs.dependency_hashes = dependencies.hashes
                    save(run_dir / "run.json", metadata)
                    save(run_dir / "environment.json", environment.model_dump())
                    if shutil.which("docker") is None:
                        raise RuntimeError(
                            "Docker executable not found. Install Docker and start its engine."
                        )
                    docker(["info"], log)
                    if limits.cpus is not None or limits.memory_bytes is not None:
                        limits.capabilities(
                            json.loads(query(["info", "--format", "{{json .}}"], log))
                        )
                    stage("image_pull")
                    docker(["pull", BASE_IMAGE], log)
                    environment.base_image = image_identity(BASE_IMAGE, log)
                    save(run_dir / "environment.json", environment.model_dump())
                    if not environment.base_image.digests:
                        raise RuntimeError("The base image has no immutable repository digest.")
                    uv_digest = None
                    if dependencies.install_command is not None:
                        docker(["pull", UV_IMAGE], log)
                        environment.uv_image = image_identity(UV_IMAGE, log)
                        save(run_dir / "environment.json", environment.model_dump())
                        if not environment.uv_image.digests:
                            raise RuntimeError("The uv image has no immutable repository digest.")
                        uv_digest = environment.uv_image.digests[0]
                    user = None
                    if hasattr(os, "getuid") and hasattr(os, "getgid"):
                        user = f"{os.getuid()}:{os.getgid()}"
                    (context / "Dockerfile").write_text(
                        build_recipe(
                            environment.base_image.digests[0],
                            uv_digest,
                            user,
                            environment.sdk_wheel,
                        ),
                        encoding="utf-8",
                    )
                    shutil.copy2(context / "Dockerfile", run_dir / "Dockerfile")
                    inputs.build_recipe_sha256 = file_hash(run_dir / "Dockerfile")
                    save(run_dir / "run.json", metadata)
                    stage("image_build")
                    docker(["build", "--tag", image, str(context)], log)
                environment.execution_image = image_identity(image, log)
                save(run_dir / "environment.json", environment.model_dump())
                stage("environment_probe")
                runtime = metadata.runtime
                assert runtime is not None
                if timeout is not None or limits.docker_arguments():
                    environment.container = container_info(
                        environment.execution_image.id,
                        log,
                        user,
                        name=runtime.probe_name,
                        labels={"mlef.owner": runtime.token, "mlef.project": runtime.project},
                        resource_arguments=limits.docker_arguments(),
                    )
                else:
                    environment.container = container_info(
                        environment.execution_image.id, log, user
                    )
                if environment.sdk_wheel is not None and (
                    environment.container.packages.get("mlef") != environment.sdk_wheel.version
                ):
                    raise RuntimeError("Execution environment does not contain the captured SDK.")
                environment.warnings = []
                warnings.clear()
                save(run_dir / "environment.json", environment.model_dump())
                if pipeline:
                    from .cache import configure_cache

                    stage("cache_selection")
                    configure_cache(root, artifacts / "pipeline", candidate=cache_from, force=force)
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
                create.extend(
                    [
                        "--label",
                        f"mlef.owner={runtime.token}",
                        "--label",
                        f"mlef.project={runtime.project}",
                    ]
                )
                create.extend(limits.docker_arguments())
                stage("container_create")
                runtime.cleanup_pending = True
                save(run_dir / "run.json", metadata)
                docker([*create, environment.execution_image.id, *command], log)
                created = True
                runtime.container_id = query(["inspect", "--format", "{{.Id}}", name], log).strip()
                host = json.loads(
                    query(
                        ["inspect", "--format", "{{json .HostConfig}}", runtime.container_id], log
                    )
                )
                applied = limits.applied(host)
                assert metadata.controls is not None
                metadata.controls.applied.update(
                    TypeAdapter(dict[str, JsonValue]).validate_python(applied)
                )
                stage("execution")
                # Do not allocate a TTY: Docker then preserves separate stdout/stderr streams.
                # Handle Ctrl+C in mlef; do not also deliver it to the attached Docker client.
                # Supervised clients inherit the worker's group so deadline cleanup reaps them.
                process = subprocess.Popen(
                    ["docker", "start", "--attach", name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
                    start_new_session=sys.platform != "win32" and _identifier is None,
                )
                metadata.state, metadata.started_at = "running", now()
                save(run_dir / "run.json", metadata)
                if max_log_bytes is None:
                    attach_exit_code = stream_process(process, run_dir, name, log)
                else:
                    attach_exit_code = stream_process(
                        process, run_dir, name, log, budget=script_budget
                    )
                state = completed_state(name)
                if state.running or not state.started_at.startswith("20"):
                    raise RuntimeError("Docker did not report a completed script execution.")
                if state.error:
                    raise RuntimeError(f"Container execution failed: {state.error}")
                metadata.execution_exit_code = state.exit_code
                assert metadata.controls is not None
                metadata.controls.applied["oom_killed"] = state.oom_killed
                metadata.execution_seconds = max(
                    0,
                    (
                        datetime.fromisoformat(state.finished_at)
                        - datetime.fromisoformat(state.started_at)
                    ).total_seconds(),
                )
                if state.exit_code:
                    if state.oom_killed:
                        metadata.error = RunError(
                            category="oom", message="Docker confirmed an OOM kill."
                        )
                    raise RuntimeError(
                        f"Script exited with code {state.exit_code}; see stderr.log."
                    )
                if attach_exit_code:
                    raise RuntimeError("Docker attach failed; experiment logs may be incomplete.")
                if pipeline:
                    pipeline_record = read_pipeline(artifacts / "pipeline", verify=True)
                    if pipeline_record.state != "complete":
                        raise RuntimeError("Pipeline exited without completing every step.")
                result = 0
            except KeyboardInterrupt:
                cancelled = True
                metadata.error = RunError(category="cancelled", message="Cancelled by user.")
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
                if metadata.error is None:
                    metadata.error = RunError(
                        category="output_limit"
                        if isinstance(error, LogLimitError)
                        else "preparation"
                        if process is None
                        else "execution",
                        message=str(error),
                    )
                with suppress(LogLimitError):
                    log.write((str(error) + "\n").encode("utf-8"))
                print(f"mlef: {error}", file=sys.stderr)
            finally:
                stage("collection")
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
                            destination = output_destination(run_dir, output.as_posix())
                            docker(
                                ["cp", f"{name}:/workspace/{output.as_posix()}", str(destination)],
                                log,
                            )
                        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                            result = 130 if cancelled else 1
                            message = f"Could not collect output {output}: {error}"
                            metadata.error = RunError(category="collection", message=message)
                            print(f"mlef: {message}", file=sys.stderr)
                    if docker(["rm", "--force", name], log, check=False).returncode:
                        warnings.append(f"Container cleanup failed: {name}")
                    elif metadata.runtime is not None:
                        metadata.runtime.cleanup_pending = False
                if (
                    shutil.which("docker") is not None
                    and docker(["image", "rm", image], log, check=False).returncode
                ):
                    warnings.append(f"Image cleanup failed or image was not built: {image}")
                metadata.state = (
                    "cancelled" if cancelled else "succeeded" if result == 0 else "failed"
                )
                metadata.finished_at = now()
                if pipeline and result != 0:
                    try:
                        interrupt_pipeline(
                            artifacts / "pipeline",
                            metadata.error.message if metadata.error else "Pipeline failed.",
                            cancelled=cancelled,
                        )
                    except (OSError, ValueError) as error:
                        warnings.append(
                            f"Pipeline interruption evidence could not be saved: {error}"
                        )
                try:
                    save(run_dir / "result.json", normalize(root, run_id))
                except (OSError, ValueError) as error:
                    result = 130 if cancelled else 1
                    metadata.state = "cancelled" if cancelled else "failed"
                    metadata.error = RunError(
                        category="storage", message=f"Cannot save result evidence: {error}"
                    )
                    print(f"mlef: {metadata.error.message}", file=sys.stderr)
                save(run_dir / "run.json", metadata)
        print(f"Run {run_id}: {metadata.state}", flush=True)
        return result
