"""Supported synchronous Python entry points; execution failures are typed run outcomes."""

import subprocess
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import ParamSpec, TypeVar
from uuid import uuid4

from pydantic import Field

from . import comparison, recovery, storage
from .events import Delivery, LogEvent
from .evidence import Result, read_result
from .limits import Limits
from .project import initialize, project_root
from .records import Contract, RunRecord
from .supervision import Request, supervise
from .tabular_comparison import TabularComparison
from .tabular_comparison import compare_tabular as _compare_tabular
from .tabular_diagnostics import Diagnostics, diagnose, read_profile
from .tabular_runtime import Profile

__all__ = [
    "LogEvent",
    "MlefError",
    "RunListing",
    "RunOptions",
    "RunStatus",
    "artifacts",
    "compare",
    "compare_tabular",
    "get",
    "init",
    "list_runs",
    "recover",
    "run",
    "status",
    "tabular_diagnostics",
    "tabular_profile",
]

P = ParamSpec("P")
T = TypeVar("T")


class MlefError(RuntimeError):
    def __init__(self, category: str, message: str) -> None:
        self.category = category
        super().__init__(message)


def _translated(function: Callable[P, T]) -> Callable[P, T]:
    @wraps(function)
    def call(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return function(*args, **kwargs)
        except storage.QueryError as error:
            raise MlefError(error.category, str(error)) from error
        except ValueError as error:
            raise MlefError("configuration", str(error)) from error
        except OSError as error:
            raise MlefError("storage", str(error)) from error
        except subprocess.SubprocessError as error:
            raise MlefError("process", str(error)) from error

    return call


class RunOptions(Limits):
    timeout_seconds: float | None = Field(default=None, gt=0)
    max_log_bytes: int | None = Field(default=None, gt=0)
    max_runner_log_bytes: int | None = Field(default=None, gt=0)


class RunStatus(RunRecord):
    result: Result


class RunListing(Contract):
    runs: list[RunRecord]
    diagnostics: list[str]


@_translated
def init(*, root: Path | str | None = None) -> None:
    initialize(project_root(root))


def _status(root: Path, identifier: str) -> RunStatus:
    record = storage.inspect(root, identifier)
    return RunStatus.model_validate(
        {**record.model_dump(mode="json"), "result": read_result(root, identifier)}
    )


@_translated
def status(identifier: str, *, root: Path | str | None = None) -> RunStatus:
    return _status(project_root(root), identifier)


@_translated
def list_runs(*, root: Path | str | None = None) -> RunListing:
    records, diagnostics = storage.list_runs(project_root(root))
    return RunListing(runs=records, diagnostics=diagnostics)


@_translated
def artifacts(identifier: str, *, root: Path | str | None = None) -> list[str]:
    return storage.artifacts(project_root(root), identifier)


@_translated
def get(
    identifier: str,
    artifact: str,
    *,
    output: Path | str | None = None,
    root: Path | str | None = None,
) -> Path:
    return storage.retrieve(
        project_root(root), identifier, artifact, Path(output) if output is not None else None
    )


@_translated
def compare(
    baseline: str, candidate: str, *, root: Path | str | None = None
) -> comparison.Comparison:
    return comparison.compare(project_root(root), baseline, candidate)


@_translated
def compare_tabular(
    baseline: str, candidate: str, *, root: Path | str | None = None, limit: int = 20
) -> TabularComparison:
    return _compare_tabular(project_root(root), baseline, candidate, limit=limit)


@_translated
def tabular_profile(identifier: str, *, root: Path | str | None = None) -> Profile:
    return read_profile(project_root(root), identifier)


@_translated
def tabular_diagnostics(
    baseline: str,
    candidate: str,
    *,
    root: Path | str | None = None,
    minimum_count: int = 20,
    limit: int = 20,
) -> Diagnostics:
    return diagnose(
        project_root(root), baseline, candidate, minimum_count=minimum_count, limit=limit
    )


@_translated
def recover(identifier: str, *, root: Path | str | None = None) -> RunStatus:
    resolved = project_root(root)
    recovery.recover(resolved, identifier)
    return _status(resolved, identifier)


@_translated
def run(
    script: Path | str,
    *,
    root: Path | str | None = None,
    arguments: list[str] | None = None,
    includes: list[Path | str] | None = None,
    outputs: list[Path | str] | None = None,
    experiment: Path | str | None = None,
    options: RunOptions | None = None,
    on_output: Callable[[LogEvent], None] | None = None,
    callback_bytes: int = 65536,
) -> RunStatus:
    resolved = project_root(root)
    options = options if options is not None else RunOptions()
    delivery = Delivery(on_output, callback_bytes)
    identifier = str(uuid4())
    request = Request(
        root=str(resolved),
        script=str(script),
        arguments=arguments or [],
        includes=[str(path) for path in (includes or [])],
        outputs=[str(path) for path in (outputs or [])],
        experiment=str((resolved / experiment).resolve()) if experiment is not None else None,
        timeout=options.timeout_seconds,
        identifier=identifier,
        token=str(uuid4()),
        limits=Limits(
            cpus=options.cpus, memory_bytes=options.memory_bytes, network=options.network
        ),
        max_log_bytes=options.max_log_bytes,
        max_runner_log_bytes=options.max_runner_log_bytes,
        quiet=True,
    )
    result = supervise(request, delivery)
    if result == 2:
        raise MlefError(
            "configuration",
            delivery.diagnostics.decode("utf-8", errors="replace").strip()
            or "Worker rejected configuration.",
        )
    if not (resolved / ".mlef/runs" / identifier / "run.json").exists():
        raise MlefError("worker_failed", "Worker exited without a saved run record.")
    record = _status(resolved, identifier)
    if record.error is not None and record.error.category == "callback":
        raise MlefError("callback", "Output callback failed; partial run evidence is preserved.")
    if delivery.budget.exceeded.is_set():
        record.warnings.append(
            "Callback byte budget exhausted; further callback output was discarded."
        )
    if any(thread.is_alive() for thread in delivery.threads):
        record.warnings.append("A callback is still blocked; callbacks must return promptly.")
    return record
