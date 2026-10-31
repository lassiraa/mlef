"""Supported synchronous Python entry points; execution failures are typed run outcomes."""

import subprocess
from collections.abc import Callable
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Literal, ParamSpec, TypeVar
from uuid import uuid4

from pydantic import ConfigDict, Field

from . import annotations as organization
from . import comparison, models, recovery, storage
from .annotations import ExperimentGroup, ExperimentListing, RunAnnotations, RunLabels
from .environment import file_hash
from .evaluation_comparison import EvaluationComparison
from .evaluations import EvaluationBatch, EvaluationListing
from .events import Delivery, LogEvent
from .evidence import Result, read_result
from .execution import PipelineRecord, read_pipeline, verify_source_archive
from .inspection import RunMetrics
from .inspection import run_metrics as _run_metrics
from .limits import Limits
from .models import ModelListing, ModelVersion
from .pipeline import ValidationReport
from .pipeline import validate as _validate
from .project import initialize, project_root
from .records import Contract, RunRecord
from .sdk import _checked
from .supervision import Request, supervise
from .tabular_artifacts import TabularBundle, export_artifacts
from .tabular_comparison import TabularComparison
from .tabular_comparison import compare_tabular as _compare_tabular
from .tabular_diagnostics import Diagnostics, diagnose, read_profile
from .tabular_runtime import Profile
from .transfers import RunTransfer, backend, fetch, publish

__all__ = [
    "EvaluationBatch",
    "EvaluationComparison",
    "EvaluationListing",
    "ExperimentDetails",
    "ExperimentGroup",
    "ExperimentListing",
    "LogEvent",
    "MlefError",
    "ModelListing",
    "ModelVersion",
    "RunAnnotations",
    "RunHistory",
    "RunListing",
    "RunMetrics",
    "RunOptions",
    "RunStatus",
    "annotate_run",
    "artifacts",
    "compare",
    "compare_evaluations",
    "compare_tabular",
    "create_experiment",
    "evaluate",
    "evaluation_batch",
    "evaluation_leaderboard",
    "experiment_details",
    "experiment_group",
    "export_tabular_artifacts",
    "fetch_run",
    "get",
    "get_model",
    "init",
    "list_evaluations",
    "list_experiments",
    "list_models",
    "list_runs",
    "model_version",
    "pipeline_status",
    "publish_run",
    "recover",
    "register_model",
    "run",
    "run_annotations",
    "run_history",
    "run_metrics",
    "status",
    "tabular_diagnostics",
    "tabular_profile",
    "validate",
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


class RunHistoryV1(RunListing):
    model_config = ConfigDict(title="RunHistory")
    schema_version: Literal[1] = 1
    total: int
    next_offset: int | None


class RunHistory(RunListing):
    schema_version: Literal[2] = 2
    total: int
    next_offset: int | None
    annotations: dict[str, RunAnnotations]


class ExperimentDetails(Contract):
    schema_version: Literal[1] = 1
    group: ExperimentGroup
    history: RunHistory


@_translated
def experiment_details(
    name: str,
    *,
    limit: int = 50,
    offset: int = 0,
    include_archived: bool = False,
    root: Path | str | None = None,
) -> ExperimentDetails:
    resolved = project_root(root)
    return ExperimentDetails(
        group=organization.group(resolved, name),
        history=run_history(
            root=resolved,
            experiment=name,
            limit=limit,
            offset=offset,
            include_archived=include_archived,
        ),
    )


@_translated
def run_annotations(identifier: str, *, root: Path | str | None = None) -> RunAnnotations:
    resolved = project_root(root)
    storage.inspect(resolved, identifier)
    return organization.read(resolved, identifier)


@_translated
def annotate_run(
    identifier: str,
    *,
    name: str | None = None,
    experiment: str | None = None,
    note: str | None = None,
    tags: dict[str, str] | None = None,
    remove_tags: list[str] | None = None,
    archived: bool | None = None,
    clear_name: bool = False,
    clear_note: bool = False,
    clear_experiment: bool = False,
    root: Path | str | None = None,
) -> RunAnnotations:
    return organization.update(
        project_root(root),
        identifier,
        name=name,
        experiment=experiment,
        note=note,
        tags=tags,
        remove_tags=remove_tags,
        archived=archived,
        clear_name=clear_name,
        clear_note=clear_note,
        clear_experiment=clear_experiment,
    )


@_translated
def create_experiment(
    name: str, *, description: str | None = None, root: Path | str | None = None
) -> ExperimentGroup:
    return organization.create_group(project_root(root), name, description)


@_translated
def list_experiments(*, root: Path | str | None = None) -> ExperimentListing:
    return organization.groups(project_root(root))


@_translated
def experiment_group(name: str, *, root: Path | str | None = None) -> ExperimentGroup:
    return organization.group(project_root(root), name)


@_translated
def evaluate(
    definition: Path | str,
    models: list[str],
    *,
    options: RunOptions | None = None,
    root: Path | str | None = None,
) -> EvaluationBatch:
    from .evaluations import evaluate as execute

    return execute(
        project_root(root),
        Path(definition),
        models,
        options if options is not None else RunOptions(),
    )


@_translated
def evaluation_batch(identifier: str, *, root: Path | str | None = None) -> EvaluationBatch:
    from .evaluations import show

    return show(project_root(root), identifier)


@_translated
def compare_evaluations(
    identifiers: list[str],
    *,
    baseline: str,
    metric: str,
    scope: str | None = None,
    output: str | None = None,
    root: Path | str | None = None,
) -> EvaluationComparison:
    from .evaluation_comparison import compare

    return compare(
        project_root(root),
        identifiers,
        baseline=baseline,
        metric=metric,
        scope=scope,
        output=output,
    )


@_translated
def evaluation_leaderboard(
    batch_id: str,
    *,
    baseline: str,
    metric: str,
    scope: str | None = None,
    output: str | None = None,
    root: Path | str | None = None,
) -> EvaluationComparison:
    from .evaluation_comparison import leaderboard

    return leaderboard(
        project_root(root), batch_id, baseline=baseline, metric=metric, scope=scope, output=output
    )


@_translated
def list_evaluations(
    *, model: str | None = None, limit: int = 50, offset: int = 0, root: Path | str | None = None
) -> EvaluationListing:
    from .evaluations import listing

    return listing(project_root(root), model, limit=limit, offset=offset)


@_translated
def list_models(*, run_id: str | None = None, root: Path | str | None = None) -> ModelListing:
    return models.list_models(project_root(root), run_id)


@_translated
def register_model(
    identifier: str,
    step: str,
    output: str,
    *,
    name: str,
    root: Path | str | None = None,
) -> ModelVersion:
    return models.register(project_root(root), identifier, step, output, name)


@_translated
def model_version(selector: str, *, root: Path | str | None = None) -> ModelVersion:
    return models.show(project_root(root), selector)


@_translated
def get_model(
    selector: str,
    *,
    output: Path | str,
    root: Path | str | None = None,
) -> Path:
    return models.get(project_root(root), selector, Path(output))


@_translated
def run_metrics(
    identifier: str,
    *,
    step: str | None = None,
    scope: str | None = None,
    metric: str | None = None,
    max_bytes: int = 32 * 1024 * 1024,
    root: Path | str | None = None,
) -> RunMetrics:
    return _run_metrics(
        project_root(root),
        identifier,
        step=step,
        scope=scope,
        metric=metric,
        max_bytes=max_bytes,
    )


@_translated
def run_history(
    *,
    state: Literal["preparing", "running", "succeeded", "failed", "cancelled"] | None = None,
    since: str | None = None,
    limit: int = 50,
    offset: int = 0,
    experiment: str | None = None,
    name: str | None = None,
    tags: dict[str, str] | None = None,
    include_archived: bool = False,
    root: Path | str | None = None,
) -> RunHistory:
    """Read newest-first history; offsets apply after filtering saved records."""
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("limit must be an integer between 1 and 1000.")
    if type(offset) is not int or offset < 0:
        raise ValueError("offset must be a nonnegative integer.")
    if state not in {None, "preparing", "running", "succeeded", "failed", "cancelled"}:
        raise ValueError("Unknown run state.")
    threshold = datetime.fromisoformat(since) if since is not None else None
    if threshold is not None and threshold.utcoffset() is None:
        raise ValueError("since must include a timezone offset.")
    RunLabels(name=name, experiment=experiment, tags=tags if tags is not None else {})
    resolved = project_root(root)
    group_id = organization.group(resolved, experiment).id if experiment is not None else None
    records, diagnostics = storage.list_runs(resolved)
    selected = []
    annotations = {}
    for record in records:
        if (state is not None and record.state != state) or (
            threshold is not None and datetime.fromisoformat(record.created_at) < threshold
        ):
            continue
        try:
            labels = organization.read(resolved, record.id)
        except storage.QueryError as error:
            diagnostics.append(f"{record.id}: {error}")
            continue
        if (
            (labels.archived and not include_archived)
            or (group_id is not None and labels.experiment_id != group_id)
            or (name is not None and labels.name != name)
            or any(labels.tags.get(key) != value for key, value in (tags or {}).items())
        ):
            continue
        selected.append(record)
        annotations[record.id] = labels
    end = offset + limit
    return RunHistory(
        runs=selected[offset:end],
        diagnostics=diagnostics,
        total=len(selected),
        next_offset=end if end < len(selected) else None,
        annotations={record.id: annotations[record.id] for record in selected[offset:end]},
    )


@_translated
def publish_run(
    identifier: str, destination: str, *, root: Path | str | None = None
) -> RunTransfer:
    return publish(project_root(root), identifier, backend(destination))


@_translated
def fetch_run(identifier: str, source: str, *, root: Path | str | None = None) -> RunTransfer:
    return fetch(project_root(root), identifier, backend(source))


@_translated
def export_tabular_artifacts(
    identifier: str,
    data: Path | str,
    destination: Path | str,
    *,
    root: Path | str | None = None,
    profile: bool = False,
) -> TabularBundle:
    resolved = project_root(root)
    return export_artifacts(
        resolved, identifier, resolved / data, resolved / destination, profile=profile
    )


@_translated
def validate(
    definition: Path | str = "mlef.toml", *, root: Path | str | None = None
) -> ValidationReport:
    return _validate(project_root(root), Path(definition))


@_translated
def init(*, root: Path | str | None = None, scaffold: bool = True) -> None:
    initialize(project_root(root), scaffold=scaffold)


@_translated
def pipeline_status(
    identifier: str, *, root: Path | str | None = None, verify: bool = False
) -> PipelineRecord:
    resolved = project_root(root)
    parent = storage.inspect(resolved, identifier)
    path = storage.artifact_path(resolved, identifier, "pipeline/pipeline.json")
    try:
        record = read_pipeline(path.parent, verify=verify)
    except (OSError, ValueError, KeyError) as error:
        raise storage.QueryError(
            "invalid_record", f"Cannot read pipeline evidence: {error}"
        ) from error
    if (
        record.run_id != identifier
        or parent.inputs is None
        or record.source_sha256 != parent.inputs.source_sha256
    ):
        raise storage.QueryError("invalid_record", "Pipeline does not match its captured run.")
    if verify:
        if parent.runtime is not None and parent.runtime.cleanup_pending:
            raise storage.QueryError("cleanup_pending", "Owned execution resources need recovery.")
        run_dir = storage.directory(resolved, identifier)
        try:
            if (
                file_hash(_checked(run_dir / "sdk" / record.sdk_wheel.filename))
                != record.sdk_wheel.sha256
            ):
                raise ValueError("Captured SDK wheel bytes changed.")
            verify_source_archive(run_dir / "source.tar.gz", record)
            if record.cache_environment is not None:
                from .cache import environment_identity

                if record.cache_environment != environment_identity(run_dir):
                    raise ValueError("Cache environment identity disagrees with saved evidence.")
        except (OSError, ValueError) as error:
            raise storage.QueryError("invalid_record", str(error)) from error
    return record


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
    script: Path | str = "mlef.toml",
    *,
    root: Path | str | None = None,
    arguments: list[str] | None = None,
    includes: list[Path | str] | None = None,
    outputs: list[Path | str] | None = None,
    experiment: Path | str | None = None,
    options: RunOptions | None = None,
    on_output: Callable[[LogEvent], None] | None = None,
    callback_bytes: int = 65536,
    reuse_from: str | None = None,
    from_step: str | None = None,
    resume: bool = False,
    cache_from: str | None = None,
    force: list[str] | None = None,
    name: str | None = None,
    experiment_name: str | None = None,
    tags: dict[str, str] | None = None,
    note: str | None = None,
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
        reuse_from=reuse_from,
        from_step=from_step,
        resume=resume,
        cache_from=cache_from,
        force=force or [],
        labels=RunLabels(
            name=name, experiment=experiment_name, tags=tags if tags is not None else {}, note=note
        )
        if any(value is not None for value in [name, experiment_name, tags, note])
        else None,
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
