Python API reference
====================

:doc:`Documentation home <index>`

Python callers can use the supported synchronous ``mlef.api`` entry points.
The API covers execution, run history and annotations, experiment groups, model
versions and aliases, evaluation batches, metrics and comparisons. Inspection
works offline without Docker or ML dependencies; execution uses Docker.
For non-executing script contract and wiring checks, see ``api.validate`` in
:doc:`pipelines`. It returns a versioned validation report without requiring
initialized run storage or Docker.

The root ``mlef`` package also exports ``Step``, ``Input``, ``Output``, ``Artifact``
and the local persistence helpers documented in :doc:`script-artifacts`. Those
explicit contexts are independent of the supervised ``api.run`` operations.

``api.publish_run(identifier, destination, *, root=None)`` and
``api.fetch_run(identifier, source, *, root=None)`` explicitly transfer whole
terminal runs through local directories or ``s3://`` stores. They return a
``RunTransfer`` manifest and use the same typed errors as other API operations.
See :doc:`storage` for integrity, credentials, retries and privacy limitations.

For supervised execution:

.. code-block:: python

    from mlef import api

    # From the checkout used in the README quick start:
    api.init()  # Or pass root="/path/to/project" to every operation.
    result = api.run(
        "examples/train.py",
        arguments=["--accuracy", "0.95"],
        options=api.RunOptions(
            timeout_seconds=120,
            cpus=0.5,
            memory_bytes=268435456,
            network="none",
            max_log_bytes=65536,
            max_runner_log_bytes=1048576,
        ),
    )
    print(result.id, result.state, result.result.metrics)


``init``, ``run``, ``list_runs``, ``status``, ``pipeline_status``, ``artifacts``, ``get``,
``compare`` and ``recover``
reuse the CLI's execution, storage, comparison and recovery services. ``run``,
``status`` and ``recover`` return a Pydantic ``RunStatus`` containing run metadata plus
normalized ``result`` evidence; ``list_runs`` returns ``RunListing(runs, diagnostics)``.
Artifact listing returns relative names; ``get`` returns the retrieved ``Path``;
``compare`` returns the existing ``Comparison`` contract. Use ``model_dump(mode="json")``
for JSON-compatible output.
Control options are validated by ``RunOptions`` before execution.

``run()`` defaults to the captured ``mlef.toml`` pipeline; pass a script path for
single-script execution. ``pipeline_status`` returns a ``PipelineRecord``
from ``mlef.execution`` without executing code or requiring Docker. Use
``verify=True`` to check saved source/SDK bytes, completed output inventories and
input producer wiring, and to reject runs with resource cleanup pending. See
:doc:`pipelines` for the locked environment requirements and managed script context.
For explicit pipeline reuse, pass ``reuse_from=RUN`` and ``from_step=STEP``
together. ``resume=True`` retains the original captured source and SDK instead
of capturing the current checkout. Each operation creates a new self-contained
run; it never overwrites or silently repairs the source run.

Calls block until execution and finalization finish, and produce no console output.
Script failures, timeouts and Ctrl+C return their saved failed/cancelled status;
invalid project/run configuration and query failures raise ``MlefError``, whose
``category`` matches the service error (for example ``unknown_run``, ``unsafe_path`` or
``configuration``). Failed execution is data, so check ``state`` before claiming success.
Optional invalid/missing metrics do not change the execution outcome. Pydantic
rejects invalid option values when constructing ``RunOptions``.

Every operation accepts optional ``root``: the exact Git root, defaulting to the
current directory. Script, include and output paths must stay within that root. Experiment paths
resolve against that root and may also be absolute; artifact names are relative to the saved artifact directory. Retrieval
destinations are relative to the caller's current directory and never overwritten.
No global working-directory change or shell command parsing is performed.

For live output, pass ``on_output=lambda event: ...`` to ``run``. ``LogEvent.stream`` is
``stdout`` or ``stderr``; ``data`` contains raw bytes from the worker, including run
progress messages. Calls are serialized on background drain threads; callbacks
must return promptly and must decode partial UTF-8 chunks incrementally if needed.
``callback_bytes`` defaults to 65536 bytes total per call; further delivery is
discarded and a warning is added to the returned status. Saved logs still follow
their separate limits. A callback exception stops the owned run, preserves partial
evidence, and raises ``MlefError(category="callback")``. Call-specific delivery
warnings are not persisted. This API supplies foreground execution, without a
scheduler or asynchronous job protocol.

Function signatures
-------------------

All parameters after ``*`` are keyword-only. ``PathLike`` below abbreviates
``Path | str``; ``Root`` abbreviates ``Path | str | None``. These are documentation
shorthands, not exported types. Every ``root`` defaults to ``None``.

.. code-block:: python

    init(*, root: Root = None, scaffold: bool = True) -> None
    pipeline_status(
        identifier: str, *, root: Root = None, verify: bool = False,
    ) -> PipelineRecord
    run(
        script: PathLike = "mlef.toml", *, root: Root = None,
        arguments: list[str] | None = None,
        includes: list[PathLike] | None = None,
        outputs: list[PathLike] | None = None,
        experiment: PathLike | None = None,
        options: RunOptions | None = None,
        on_output: Callable[[LogEvent], None] | None = None,
        callback_bytes: int = 65536,
        reuse_from: str | None = None, from_step: str | None = None,
        resume: bool = False, cache_from: str | None = None,
        force: list[str] | None = None,
        name: str | None = None, experiment_name: str | None = None,
        tags: dict[str, str] | None = None, note: str | None = None,
    ) -> RunStatus
    status(identifier: str, *, root: Root = None) -> RunStatus
    list_runs(*, root: Root = None) -> RunListing
    artifacts(identifier: str, *, root: Root = None) -> list[str]
    get(identifier: str, artifact: str, *, output: PathLike | None = None,
        root: Root = None) -> Path
    compare(baseline: str, candidate: str, *, root: Root = None) -> Comparison
    recover(identifier: str, *, root: Root = None) -> RunStatus
    compare_tabular(baseline: str, candidate: str, *, root: Root = None,
                    limit: int = 20) -> TabularComparison
    tabular_profile(identifier: str, *, root: Root = None) -> Profile
    tabular_diagnostics(baseline: str, candidate: str, *, root: Root = None,
                        minimum_count: int = 20, limit: int = 20) -> Diagnostics

``RunOptions`` defaults to no deadline, CPU/memory cap or saved-log cap,
and ``network="default"``. Optional ``timeout_seconds`` and ``cpus`` are positive
floats; ``memory_bytes``, ``max_log_bytes`` and ``max_runner_log_bytes`` are positive
integers. ``network`` accepts ``"default"`` or ``"none"``. See the
:doc:`run reference <runs>` for enforcement scope and cleanup grace periods.

``compare_tabular`` checks actual fold/prediction evidence; ``compare`` checks
declared experiment compatibility. ``limit`` ranges from 1 to 100;
``minimum_count`` ranges from 2 to 1,000,000. Tabular inspection is offline.
See the :doc:`tabular guide <tabular>` for score definitions and interpretation.

Run tracking and experiment groups
----------------------------------

``run_history`` returns a version-2 ``RunHistory`` envelope with runs, annotations,
diagnostics, total matching records and the next offset. Filters apply before
pagination; archived runs are hidden by default. ``run_metrics`` returns root
result evidence and qualified pipeline metric artifacts without implicitly
ranking or reducing their payloads. See :doc:`runs` for filter semantics,
annotation limits and verification boundaries.

.. code-block:: python

    run_history(
        *, state: Literal["preparing", "running", "succeeded", "failed", "cancelled"] | None = None,
        since: str | None = None, limit: int = 50, offset: int = 0,
        experiment: str | None = None, name: str | None = None,
        tags: dict[str, str] | None = None, include_archived: bool = False,
        root: Root = None,
    ) -> RunHistory
    run_metrics(
        identifier: str, *, step: str | None = None, scope: str | None = None,
        metric: str | None = None, max_bytes: int = 33554432, root: Root = None,
    ) -> RunMetrics
    run_annotations(identifier: str, *, root: Root = None) -> RunAnnotations
    annotate_run(
        identifier: str, *, name: str | None = None, experiment: str | None = None,
        note: str | None = None, tags: dict[str, str] | None = None,
        remove_tags: list[str] | None = None, archived: bool | None = None,
        clear_name: bool = False, clear_note: bool = False,
        clear_experiment: bool = False, root: Root = None,
    ) -> RunAnnotations
    create_experiment(
        name: str, *, description: str | None = None, root: Root = None,
    ) -> ExperimentGroup
    list_experiments(*, root: Root = None) -> ExperimentListing
    experiment_group(name: str, *, root: Root = None) -> ExperimentGroup
    experiment_details(
        name: str, *, limit: int = 50, offset: int = 0,
        include_archived: bool = False, root: Root = None,
    ) -> ExperimentDetails

``experiment_group`` reads group metadata; ``experiment_details`` also returns
paged run history. ``run(experiment_name=...)`` assigns a group, while
``run(experiment=...)`` captures the separate immutable experiment-intent file.
Annotations live outside captured run evidence. Omitted annotation fields are
preserved, tags merge by key, and explicit clear/remove arguments delete labels.

.. code-block:: python

    from mlef import api

    page = api.run_history(experiment="Churn models", tags={"family": "linear"})
    for record in page.runs:
        print(record.id, record.state, page.annotations[record.id].name)
        metrics = api.run_metrics(record.id)
    api.annotate_run(run_id, note="Reviewed", tags={"reviewed": "yes"})

Models and aliases
------------------

``list_models(run_id=...)`` discovers typed completed model outputs, including
unregistered artifacts; omitting ``run_id`` lists registered versions.
Registration is idempotent for the same name and original producer. Version
descriptors are immutable, and retrieval verifies and copies saved model bytes
without deserializing them. See :doc:`models` for contracts and storage details.

.. code-block:: python

    list_models(*, run_id: str | None = None, root: Root = None) -> ModelListing
    register_model(
        identifier: str, step: str, output: str, *, name: str, root: Root = None,
    ) -> ModelVersion
    model_version(selector: str, *, root: Root = None) -> ModelVersion
    get_model(selector: str, *, output: PathLike, root: Root = None) -> Path
    set_model_alias(
        selector: str, *, alias: str, note: str | None = None, root: Root = None,
    ) -> ModelAlias
    model_alias(selector: str, *, root: Root = None) -> ModelAlias
    list_model_aliases(name: str, *, root: Root = None) -> AliasListing

Selectors use ``MODEL@VERSION`` for exact versions or ``MODEL:ALIAS`` for mutable
pointers. ``set_model_alias`` requires an exact version; ``model_alias`` reads an
alias and its complete change history. ``model_version`` and ``get_model`` accept
either selector. Existing retrieval destinations are refused.

Evaluation batches and comparisons
----------------------------------

``evaluate`` resolves model selectors to exact versions, freezes the evaluator
source/SDK and creates a sequential batch of normal child runs. Its returned
``EvaluationBatch`` retains each candidate's state, run ID and metric summaries.
Inspect the batch state for partial failures or cancellation. See
:doc:`evaluations` for definition syntax, outcomes and comparison requirements.

.. code-block:: python

    evaluate(
        definition: PathLike, models: list[str], *, options: RunOptions | None = None,
        root: Root = None,
    ) -> EvaluationBatch
    evaluation_batch(identifier: str, *, root: Root = None) -> EvaluationBatch
    list_evaluations(
        *, model: str | None = None, limit: int = 50, offset: int = 0,
        root: Root = None,
    ) -> EvaluationListing
    compare_evaluations(
        identifiers: list[str], *, baseline: str, metric: str,
        scope: str | None = None, output: str | None = None, root: Root = None,
    ) -> EvaluationComparison
    evaluation_leaderboard(
        batch_id: str, *, baseline: str, metric: str,
        scope: str | None = None, output: str | None = None, root: Root = None,
    ) -> EvaluationComparison

``evaluation_batch`` takes a batch UUID; comparison identifiers and ``baseline``
are child run UUIDs. ``output`` qualifies the metric artifact's output name.
``list_evaluations(model=...)`` filters batches by exact version or the current
target of an alias. Comparisons reverify child evidence; only compatible,
successful scalar results with a declared direction are eligible for ranking.
Inspect row compatibility and eligibility, since a valid query can contain
unranked results.

Use ``compare`` for legacy root metrics and declared experiment intent,
``compare_evaluations`` for frozen saved-model evaluations, and
``compare_tabular`` for the specialized saved-fold training workflow. These
operations retain their distinct evidence and compatibility rules.

Portable tabular helper and contracts
-------------------------------------

``mlef.tabular.export_helper(destination: Path | str) -> Path`` copies the exact
installed evaluator into a new file, refusing to overwrite an existing path.
Capture this file with experiment inputs so the evaluator is part of provenance.
Run its ``prepare``, ``evaluate`` and ``predict`` commands inside the locked
experiment environment; the host does not need sklearn or XGBoost.

``mlef.tabular`` exports the Pydantic contracts ``Protocol``, ``Split``, ``Fold``,
``Evaluation``, ``Prediction`` and ``Profile``. Note that tabular ``Evaluation``
describes actual fold evidence; ``mlef.records.Evaluation`` describes experiment
intent. Public JSON contracts live in `schemas <https://github.com/lassiraa/mlef/tree/main/schemas>`_; generate JSON with
``model_dump(mode="json")`` or ``model_dump_json()``. Importing the API and
tabular contracts requires no ML dependencies.

For host installation and running copied demos, see :doc:`setup <setup>`.

``api.export_tabular_artifacts(identifier, data, destination, *, root=None,
profile=False)`` returns a ``TabularBundle`` after exporting verified public
evaluation evidence to a fresh destination. ``mlef.tabular.read_artifacts(Path)``
verifies a moved or standalone bundle without the original run or ML dependencies.
See :doc:`tabular` for payload formats, CV recipe limitations and privacy boundaries.
