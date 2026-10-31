"""Bound the complete worker lifetime, including stalled preparation calls."""

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

from pydantic import Field

from .events import Delivery
from .evidence import Diagnostic, Result
from .limits import Limits
from .locking import run_lock, supervisor_lock
from .records import Contract, Controls, RunError, RunRecord, Runtime, read_run

CLEANUP_SECONDS = 15.0


class Request(Contract):
    root: str
    script: str
    arguments: list[str]
    includes: list[str]
    outputs: list[str]
    experiment: str | None
    timeout: float | None = Field(default=None, gt=0)
    identifier: str
    token: str
    limits: Limits = Field(default_factory=Limits)
    max_log_bytes: int | None = Field(default=None, gt=0)
    max_runner_log_bytes: int | None = Field(default=None, gt=0)
    quiet: bool = False
    reuse_from: str | None = None
    from_step: str | None = None
    resume: bool = False


def identity(root: Path, identifier: str, token: str) -> Runtime:
    project = hashlib.sha256(os.path.normcase(str(root.resolve())).encode()).hexdigest()
    return Runtime(
        token=token,
        project=project,
        container_name=f"mlef-{identifier}",
        probe_name=f"mlef-probe-{identifier}",
    )


def kill_tree(process: subprocess.Popen[bytes]) -> None:
    if hasattr(os, "killpg") and hasattr(signal, "SIGKILL"):
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    else:
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            check=False,
            timeout=5,
        )
    process.wait(timeout=5)


def cleanup(root: Path, record: RunRecord, run_dir: Path) -> bool:
    """Collect and remove only exact owned resources; retain them on collection failure."""
    from .collection import output_destination

    runtime = record.runtime
    if runtime is None or runtime.project != identity(root, record.id, runtime.token).project:
        record.warnings.append("Cleanup refused: project ownership is unknown or mismatched.")
        return False
    deadline = time.monotonic() + CLEANUP_SECONDS
    complete = True

    def docker(*arguments: str) -> subprocess.CompletedProcess[str]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Cleanup grace period expired.")
        return subprocess.run(
            ["docker", *arguments],
            capture_output=True,
            text=True,
            check=False,
            timeout=min(5, remaining),
        )

    for name in [runtime.container_name, runtime.probe_name]:
        try:
            inspection = docker(
                "inspect",
                runtime.container_id
                if name == runtime.container_name and runtime.container_id is not None
                else name,
            )
            if inspection.returncode:
                if (
                    "no such" in inspection.stderr.casefold()
                    and docker("info", "--format", "{{.ServerVersion}}").returncode == 0
                ):
                    continue
                raise RuntimeError(f"Cannot inspect owned resource {name}.")
            container = json.loads(inspection.stdout)[0]
            labels = container["Config"].get("Labels") or {}
            if (
                labels.get("mlef.owner") != runtime.token
                or labels.get("mlef.project") != runtime.project
            ):
                raise RuntimeError(f"Refused resource with different ownership: {name}")
            identifier = container["Id"]
            if container["State"]["Running"] and docker("kill", identifier).returncode:
                raise RuntimeError(f"Cannot stop owned resource: {identifier}")
            if name == runtime.container_name:
                for output in record.outputs:
                    target = output_destination(run_dir, output)
                    if docker("cp", f"{identifier}:/workspace/{output}", str(target)).returncode:
                        raise RuntimeError(f"Collection incomplete: {output}; container retained.")
            if docker("rm", "--force", identifier).returncode:
                raise RuntimeError(f"Container cleanup incomplete: {identifier}")
        except (
            OSError,
            ValueError,
            RuntimeError,
            KeyError,
            IndexError,
            subprocess.SubprocessError,
        ) as error:
            complete = False
            record.warnings.append(f"Cleanup incomplete for {name}: {error}")
    return complete


def start_worker(path: Path) -> subprocess.Popen[bytes]:
    request = Request.model_validate_json(path.read_text(encoding="utf-8"))
    return subprocess.Popen(
        [sys.executable, "-u", "-m", "mlef.supervision", str(path)],
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        start_new_session=os.name != "nt",
        stdout=subprocess.PIPE if request.quiet else None,
        stderr=subprocess.PIPE if request.quiet else None,
    )


def supervise(request: Request, delivery: Delivery | None = None) -> int:
    from .runner import now, save

    root = Path(request.root)
    storage = root / ".mlef"
    if not (storage / "runs").is_dir():
        raise ValueError("Project is not initialized; run mlef init first.")
    requests = storage / "requests"
    requests.mkdir(exist_ok=True)
    with supervisor_lock(root, request.identifier):
        path = requests / f"{request.identifier}.json"
        path.write_text(request.model_dump_json(), encoding="utf-8")
        began = time.monotonic()
        process = start_worker(path)
        if request.quiet:
            delivery = delivery if delivery is not None else Delivery(None, 65536)
            delivery.start(process)
        cancelled = False
        callback_failed = False
        try:
            try:
                while True:
                    if delivery is not None and delivery.errors:
                        callback_failed = True
                        raise RuntimeError("Output callback failed.")
                    remaining = (
                        request.timeout - (time.monotonic() - began)
                        if request.timeout is not None
                        else None
                    )
                    if remaining is not None and remaining <= 0:
                        raise subprocess.TimeoutExpired(process.args, remaining)
                    try:
                        result = process.wait(
                            timeout=min(0.05, remaining) if remaining is not None else 0.05
                        )
                        break
                    except subprocess.TimeoutExpired:
                        pass
                if delivery is not None:
                    delivery.finish()
                    if delivery.errors:
                        callback_failed = True
                        raise RuntimeError("Output callback failed.")
                if hasattr(os, "killpg") and hasattr(signal, "SIGKILL"):
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                return result
            except subprocess.TimeoutExpired:
                pass
            except RuntimeError:
                if not callback_failed:
                    raise
            except KeyboardInterrupt:
                cancelled = True
            if process.poll() is None:
                kill_tree(process)
            run_dir = storage / "runs" / request.identifier
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "artifacts").mkdir(exist_ok=True)
            with run_lock(run_dir):
                record_path = run_dir / "run.json"
                record = (
                    read_run(record_path)
                    if record_path.exists()
                    else RunRecord(
                        id=request.identifier,
                        state="preparing",
                        created_at=now(),
                        command=(
                            ["python", "-I", "-m", "mlef.execution", "/outputs/pipeline"]
                            if Path(request.script).suffix == ".toml"
                            else ["python", request.script, *request.arguments]
                        ),
                        outputs=request.outputs,
                        runtime=identity(root, request.identifier, request.token),
                        stage="configuration",
                        controls=Controls(
                            requested={"timeout_seconds": request.timeout},
                            applied={"supervised_deadline": True},
                        ),
                    )
                )
                for name in ["stdout.log", "stderr.log", "runner.log"]:
                    (run_dir / name).touch(exist_ok=True)
                completed_cleanup = cleanup(root, record, run_dir)
                if record.runtime is not None:
                    record.runtime.cleanup_pending = not completed_cleanup
                record.state = "cancelled" if cancelled else "failed"
                record.finished_at = now()
                category = (
                    "cancelled" if cancelled else "callback" if callback_failed else "timeout"
                )
                record.error = RunError(
                    category=category, message=f"{category.capitalize()} during {record.stage}."
                )
                from .execution import interrupt_pipeline

                try:
                    interrupt_pipeline(
                        run_dir / "artifacts/pipeline", record.error.message, cancelled=cancelled
                    )
                except (OSError, ValueError) as error:
                    record.warnings.append(f"Pipeline interruption evidence was not saved: {error}")
                save(
                    run_dir / "result.json",
                    Result(
                        run_id=record.id,
                        evidence_status="missing",
                        diagnostics=[
                            Diagnostic(
                                code="evidence_interrupted",
                                severity="warning",
                                message="Evidence interrupted; partial artifacts were kept.",
                            )
                        ],
                    ),
                )
                save(record_path, record)
                if not request.quiet:
                    print(f"mlef: {record.error.message} Results: {run_dir}", file=sys.stderr)
                return 130 if cancelled else 1
        finally:
            if delivery is not None:
                delivery.finish()
            path.unlink(missing_ok=True)


def main() -> None:
    from .runner import run

    request = Request.model_validate_json(Path(sys.argv[1]).read_text(encoding="utf-8"))
    try:
        result = run(
            Path(request.root),
            Path(request.script),
            request.arguments,
            [Path(path) for path in request.includes],
            [Path(path) for path in request.outputs],
            Path(request.experiment) if request.experiment is not None else None,
            timeout=request.timeout,
            _identifier=request.identifier,
            _token=request.token,
            reuse_from=request.reuse_from,
            from_step=request.from_step,
            resume=request.resume,
            cpus=request.limits.cpus,
            memory=request.limits.memory_bytes,
            network=request.limits.network,
            max_log_bytes=request.max_log_bytes,
            max_runner_log_bytes=request.max_runner_log_bytes,
        )
    except (ValueError, OSError) as error:
        print(f"mlef: {error}", file=sys.stderr)
        result = 2 if isinstance(error, ValueError) else 1
    raise SystemExit(result)


if __name__ == "__main__":
    main()
