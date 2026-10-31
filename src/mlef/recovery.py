"""Explicit recovery of abandoned owned runs; ordinary inspection remains read-only."""

import subprocess
from pathlib import Path

from .environment import query
from .evidence import normalize
from .locking import run_lock, supervisor_lock
from .processes import BoundedLog
from .records import RunError, RunRecord, read_run
from .storage import QueryError, directory
from .supervision import cleanup, identity


def recover(root: Path, identifier: str) -> RunRecord:
    from .runner import now, save

    run_dir = directory(root, identifier)
    with supervisor_lock(root, identifier), run_lock(run_dir):
        record = read_run(run_dir / "run.json")
        if record.id != identifier:
            raise QueryError("invalid_record", "Run ID does not match its storage directory.")
        runtime = record.runtime
        terminal = record.state in {"succeeded", "failed", "cancelled"}
        if terminal and (runtime is None or not runtime.cleanup_pending):
            return record
        if runtime is None or runtime.project != identity(root, identifier, runtime.token).project:
            raise QueryError(
                "unsafe_recovery",
                "Reliable ownership evidence is absent or belongs to another project.",
            )
        with BoundedLog(str(run_dir / "recovery.log"), 65536) as log:
            try:
                query(["info", "--format", "{{.ServerVersion}}"], log)
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                raise QueryError(
                    "engine_unavailable", "Docker is unavailable; recorded liveness is unchanged."
                ) from error
            complete = cleanup(root, record, run_dir)
        runtime.cleanup_pending = not complete
        if not terminal:
            record.state = "failed"
            record.finished_at = now()
            record.error = RunError(
                category="abandoned",
                message="Original runner disappeared; recovery does not establish script success.",
            )
        record.stage = "recovered" if complete else "recovery_incomplete"
        if record.state != "succeeded":
            from .execution import interrupt_pipeline

            try:
                interrupt_pipeline(
                    run_dir / "artifacts/pipeline",
                    record.error.message if record.error else "Run was abandoned.",
                    cancelled=record.state == "cancelled",
                )
            except (OSError, ValueError) as error:
                record.warnings.append(f"Pipeline interruption evidence was not saved: {error}")
        save(run_dir / "result.json", normalize(root, identifier))
        save(run_dir / "run.json", record)
        if not complete:
            raise QueryError(
                "recovery_incomplete",
                "Some resources or outputs need attention; originals were retained.",
            )
        return record
