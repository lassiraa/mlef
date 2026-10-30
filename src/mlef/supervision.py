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

from .evidence import Diagnostic, Result
from .limits import Limits
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


def cleanup(root: Path, record: RunRecord, run_dir: Path) -> None:
    """Only touch resources whose exact recorded ownership labels match."""
    runtime = record.runtime
    if runtime is None:
        return
    deadline = time.monotonic() + CLEANUP_SECONDS

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
            inspection = docker("inspect", name)
            if inspection.returncode:
                record.warnings.append(
                    f"Cleanup could not inspect {name}; it may require recovery."
                )
                continue
            objects = json.loads(inspection.stdout)
            container = objects[0]
            labels = container["Config"].get("Labels") or {}
            if (
                labels.get("mlef.owner") != runtime.token
                or labels.get("mlef.project") != runtime.project
            ):
                record.warnings.append(
                    f"Refused cleanup of a resource with different ownership: {name}"
                )
                continue
            identifier = container["Id"]
            docker("kill", identifier)
            if name == runtime.container_name:
                for output in record.outputs:
                    target = run_dir / "artifacts" / output
                    if target.exists() or not target.resolve().is_relative_to(
                        (run_dir / "artifacts").resolve()
                    ):
                        record.warnings.append(
                            f"Output not overwritten during deadline cleanup: {output}"
                        )
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    copied = docker("cp", f"{identifier}:/workspace/{output}", str(target))
                    if copied.returncode:
                        record.warnings.append(f"Output collection incomplete: {output}")
            removed = docker("rm", "--force", identifier)
            if removed.returncode:
                record.warnings.append(f"Container cleanup incomplete: {identifier}")
        except (OSError, ValueError, subprocess.SubprocessError, TimeoutError) as error:
            record.warnings.append(f"Bounded cleanup incomplete for {name}: {error}")


def start_worker(path: Path) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-u", "-m", "mlef.supervision", str(path)],
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        start_new_session=os.name != "nt",
    )


def supervise(request: Request) -> int:
    from .runner import now, save

    root = Path(request.root)
    storage = root / ".mlef"
    if not (storage / "runs").is_dir():
        raise ValueError("Project is not initialized; run mlef init first.")
    requests = storage / "requests"
    requests.mkdir(exist_ok=True)
    path = requests / f"{request.identifier}.json"
    path.write_text(request.model_dump_json(), encoding="utf-8")
    began = time.monotonic()
    process = start_worker(path)
    cancelled = False
    try:
        try:
            remaining = (
                max(0, request.timeout - (time.monotonic() - began))
                if request.timeout is not None
                else None
            )
            result = process.wait(timeout=remaining)
            if hasattr(os, "killpg") and hasattr(signal, "SIGKILL"):
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            return result
        except subprocess.TimeoutExpired:
            pass
        except KeyboardInterrupt:
            cancelled = True
        kill_tree(process)
        run_dir = storage / "runs" / request.identifier
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "artifacts").mkdir(exist_ok=True)
        record_path = run_dir / "run.json"
        record = (
            read_run(record_path)
            if record_path.exists()
            else RunRecord(
                id=request.identifier,
                state="preparing",
                created_at=now(),
                command=["python", request.script, *request.arguments],
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
        cleanup(root, record, run_dir)
        record.state = "cancelled" if cancelled else "failed"
        record.finished_at = now()
        category = "cancelled" if cancelled else "timeout"
        record.error = RunError(
            category=category, message=f"{category.capitalize()} during {record.stage}."
        )
        save(
            run_dir / "result.json",
            Result(
                run_id=record.id,
                evidence_status="missing",
                diagnostics=[
                    Diagnostic(
                        code="evidence_interrupted",
                        severity="warning",
                        message="Partial artifacts remain retrievable; evidence was interrupted.",
                    )
                ],
            ),
        )
        save(record_path, record)
        print(f"mlef: {record.error.message} Results: {run_dir}", file=sys.stderr)
        return 130 if cancelled else 1
    finally:
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
