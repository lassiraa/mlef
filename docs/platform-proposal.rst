ML platform functionality proposal
==================================

Status: draft requirements. Commands and contracts below are proposed unless
documented in :doc:`runs`. Filtered/paginated history, run and step details, and
root/pipeline metric inspection are implemented initial slices. Experiment
grouping, model registration and multi-model evaluation remain planned.
This proposal extends the local, file-backed execution model.

The intended workflow is: find previous experiments, inspect their models and
metrics, evaluate selected model versions under one saved protocol, compare the
results, and retain the evidence behind a selection.

Existing foundation
---------------------------

MLEF already has captured runs, offline list/status/artifact inspection, experiment
parameters and provenance, typed model artifacts, rich metric payloads, pipeline
steps, explicit reuse/cache, and verified local/S3 run transfers. Generic run
comparison and specialized tabular evaluation/comparison already exist.

The missing layer is discovery and organization across these records. In
particular, pipeline metric artifacts are not exposed through generic run
comparison, models have no user-facing version catalog, and steps have no
declared training/evaluation role. The existing artifact codec registry is not
a model registry and should keep its current purpose.

1. Searchable run history
---------------------------------

Required behavior:

* Group runs into named experiments, with a run name, tags and notes. Keep the
  existing per-run ``Experiment`` intent (hypothesis, parameters, seeds, etc.);
  add a separate experiment-group identity rather than changing its meaning.
* Filter history by experiment, state, time range, tags, source revision, model,
  dataset and parameter values. Select columns and sort by time, duration or a
  fully qualified scalar metric. Support bounded results and pagination.
* Show a run's recorded state, timestamps, duration, parameters, source and
  environment, parent/reuse relationships, steps, models, evaluations, metrics,
  artifact references and diagnostics from one detail command.
* Inspect step attempts and saved logs directly. Distinguish execution time from
  reused output timestamps, and last-recorded state from confirmed liveness.
* Preserve failed/cancelled runs and partial evidence in history. Corrupt records
  produce visible diagnostics; missing metrics never become zero.
* Keep annotations editable without rewriting immutable execution evidence.
  Archiving hides records by default without deleting their artifacts.

Proposed CLI:

.. code-block:: text

    mlef experiments list
    mlef experiments show churn
    mlef run --experiment-name churn --name baseline --tag family=linear
    mlef runs list --experiment churn --state succeeded --limit 50
    mlef runs list --tag family=linear --since 2026-10-01 --json
    mlef runs show RUN
    mlef runs steps RUN
    mlef runs logs RUN --step evaluate --tail 100
    mlef runs annotate RUN --tag reviewed=true --note "Validation baseline"

Keep ``mlef list``, ``status``, ``get`` and ``compare`` working with their current
output contracts. New namespaced commands may provide richer versioned responses.
Keep ``--experiment FILE`` for captured experiment intent; do not overload it
with the new group name.

2. Model discovery and version tracking
-----------------------------------------------

Separate three concepts: a logical model name, an immutable model version, and
the exact model artifact produced by a step. A run may produce multiple models;
a model version may have many evaluations from different runs.

Required behavior:

* Discover completed typed model outputs automatically without requiring
  registration. Register selected outputs under a logical name such as
  ``churn-classifier`` and assign an immutable version ID.
* Record the exact artifact reference and checksum, producing run/step,
  training parameters, available dataset/split references, source/environment
  identity, task, framework/format and available input/output signature. Unknown
  metadata remains explicitly unknown; never infer lineage from filenames.
* Registration references verified saved bytes and never deserializes a model.
  Re-registering the same producer artifact under the same name is idempotent.
  Identical bytes from distinct producers retain their distinct provenance.
* Inspect all evaluations of a version, including protocol, dataset, scope,
  metrics and evaluation failures. There is no single mutable accuracy field
  on the model itself.
* Retrieve model bytes through existing verified artifact retrieval. An external
  model can be imported explicitly through a captured import step; mark absent
  training provenance as unknown.
* Support optional mutable aliases such as ``baseline`` or ``champion`` after
  basic version tracking. Resolve aliases to exact versions before execution
  and retain alias-change history. An alias is not a deployment operation.

Proposed CLI (``RUN/train/model`` denotes a producer selector):

.. code-block:: text

    mlef models list
    mlef models list --run RUN
    mlef models register RUN/train/model --name churn-classifier
    mlef models show churn-classifier@VERSION
    mlef models evaluations churn-classifier@VERSION
    mlef models get churn-classifier@VERSION --output model
    mlef models alias churn-classifier@VERSION --alias champion

Selectors must resolve unambiguously to exact saved identities; saved records
contain full IDs and checksums, never unresolved aliases or implicit latest runs.

3. Typed step roles
---------------------------

Add a semantic ``kind`` to the existing ``Step`` contract rather than a separate
execution framework or subclass hierarchy. Existing declarations default to
``generic``. Initial validated kinds are ``generic``, ``train``, ``predict`` and
``evaluate``; ingestion and transformation can remain generic initially.

* ``train`` declares training inputs and at least one model output. Preserve
  training parameters and seed information in its execution evidence.
* ``predict`` consumes an exact model and dataset and produces predictions that
  reference both, including row identity and evaluation scope.
* ``evaluate`` produces at least one structured metrics artifact. It either
  consumes a model plus evaluation data, or consumes existing predictions plus
  their ground-truth data. Both modes must identify the evaluated model,
  dataset/split and protocol. It must not silently fit a model on evaluation data.
* Names remain arbitrary: a step named ``evaluate`` is not automatically an
  evaluation step. Validate declarations statically and output references at
  publication, using the current trusted-script execution boundary.

Model formats remain explicit. The evaluator script supplies loading and scoring
logic in its locked environment; MLEF does not promise a universal model loader.
Declared roles and checks do not prove that trusted user code avoided leakage.

4. Reusable evaluation protocols and multi-model execution
------------------------------------------------------------------

Make evaluation a recorded relationship between an exact model, dataset/split,
protocol and producing run/step. A protocol captures evaluator source/config,
environment, task, preprocessing, seeds, scope, metric definitions and aggregation
rules. Give it a content identity; a friendly protocol name alone is insufficient.

Required behavior:

* Evaluate one or more existing model versions without repeating training.
  Bind exact cross-run artifacts through verified input copies and retain them
  in the new run, extending today's within-pipeline binding rules explicitly.
* Freeze the candidate list, protocol, dataset and split before launching work.
  Save an evaluation-batch record with one child run per model. Each child uses
  the normal runner, ownership, timeout, cancellation and collection mechanisms.
* Initially execute child runs sequentially. Keep a fresh process/container and
  output namespace per model; reuse prepared environment images where supported.
  Concurrency is a later scheduling feature, not a prerequisite for comparison.
* Use one compatible locked evaluator environment per batch initially. Reject
  unsupported model formats/dependencies during validation where detectable;
  execution-time loader errors remain per-model failures. Separate environments
  can produce separate evaluations, with compatibility assessed explicitly.
* Continue independent candidates after a model failure, retaining their results.
  Batch states distinguish complete, partial, failed and cancelled; the command
  returns a failure if any requested evaluation fails. Ctrl+C stops the active
  child and leaves remaining candidates explicitly unstarted.
* Retry failures explicitly as new runs linked to the original batch; never
  overwrite evidence or silently count retries as independent measurements.
* Record predictions when the protocol requires them, timing with measurement
  method/environment, and structured metrics with exact evidence references.

Proposed CLI:

.. code-block:: text

    mlef evaluate evaluation.toml --model churn-classifier@V1 --model churn-classifier@V2
    mlef evaluations list --model churn-classifier@V1
    mlef evaluations show EVALUATION
    mlef evaluations compare EVAL_A EVAL_B --baseline EVAL_A
    mlef evaluations leaderboard --batch BATCH --metric validation.roc_auc

The evaluation definition declares an evaluator script, verified data/split
selectors and protocol parameters. The script's ``Step(kind="evaluate", ...)``
contract remains the source of its input/output declarations. Exact syntax and
schema versions should be finalized during implementation.

Evaluating a frozen trained model is distinct from comparing training recipes
with cross-validation. Preserve the existing tabular saved-fold workflow; adapt
its trial, fold-model and aggregate evidence into the common query layer without
pretending that an out-of-fold score evaluates a final refitted model. General
training matrices and hyperparameter sweeps can follow later.

5. Unified metric inspection and trustworthy comparison
---------------------------------------------------------------

Expose existing root metrics, pipeline ``Metrics`` artifacts and tabular results
through one query API while retaining their original evidence and semantics.

* Address metrics by run, step, model/evaluation, scope, name and dimensions.
  Ambiguous names require qualification; never silently choose one step's score.
* List scalar summaries and available series, curves, matrices and tables.
  Retrieve a selected payload or export JSON/CSV; external payloads are resolved
  only explicitly, under the existing verification and size limits.
* Compare multiple evaluations with an explicit baseline. Show values, deltas,
  direction, sample counts and available uncertainty, alongside compatibility
  reasons. Preserve ``Undefined`` values and failure states.
* Check dataset/split identity, evaluated rows, scope, protocol/evaluator,
  metric definitions/versions, units, parameters, aggregation and seeds. Extend
  existing comparators rather than creating a second conflicting ruleset.
* Permit side-by-side inspection of incompatible or unknown evidence, but rank
  only compatible successful evaluations for an explicitly selected scalar.
  Handle maximize, minimize and target directions; metrics without an ordering
  cannot produce a winner. Never rank curves or average folds implicitly.
* Keep validation selection separate from held-out testing. Report actual
  aggregation and uncertainty methods; a score difference alone is not a
  significance claim. Runtime comparisons expose hardware/control differences.
* Legacy flat metrics lacking evaluation identities remain inspectable with
  compatibility marked unknown. Reused results retain original producer identity
  and are not counted as independent repeated measurements.

Proposed CLI:

.. code-block:: text

    mlef runs metrics RUN
    mlef runs metrics RUN --step evaluate --scope validation --json
    mlef runs metrics RUN --step train --metric loss --resolve --output loss.json

6. Storage, API and compatibility requirements
------------------------------------------------------

Run directories and artifact manifests remain authoritative. Add versioned
file-backed experiment-group, model-version, evaluation and batch records with
atomic publication and ownership-aware updates. Mutable notes/tags/aliases live
separately from captured evidence. Use locking for concurrent registration and
alias changes; partial writes must not create valid catalog entries.

Start queries by scanning compact records. Add a rebuildable SQLite index only
if measured query latency warrants it; no mandatory service or database server.
Inspection must work offline without Docker, ML packages or importing user code.

Expose each capability through the Python API as well as the CLI. New query
responses include schema versions, bounded results, diagnostic information and
pagination metadata. Keep existing JSON shapes and exit-code behavior stable.

Version changed strict contracts deliberately and retain readers for existing
runs, artifacts and captured SDKs. Missing new fields mean unknown or generic,
not inferred successful evaluations. Querying never migrates saved evidence.
Extend schema exports and reference verification alongside the new records.

Evaluation runs must carry the model/data evidence needed for self-contained
inspection and transfer. Extend explicit publish/fetch to include immutable model
and evaluation descriptors belonging to the transferred run; importing them must
not overwrite local aliases or annotations. Do not implicitly traverse arbitrary
remote references or claim that a descriptor contains its target bytes.

Model deletion, retention and garbage collection need reference protection for
registered versions and saved evaluations. Defer physical deletion; use archival
first. Existing local SDK attempts remain outside captured run history unless
explicitly imported, with their missing capture evidence clearly identified.

Delivery order and acceptance criteria
----------------------------------------------

1. **Run and metric discovery.** Add experiment grouping, annotations, history
   filters, run/step detail and unified metric queries. Acceptance: inspect both
   a legacy script run and a pipeline's evaluation metrics offline; filters and
   pagination are deterministic, and corrupt evidence produces diagnostics.
2. **Model catalog.** Discover typed model outputs, register immutable versions,
   show lineage and retrieve bytes. Acceptance: a run with two models exposes
   both; registration is idempotent, concurrent writes are safe, and corrupt or
   incomplete producer artifacts cannot be registered.
3. **Evaluation workflow.** Add step roles, protocol/evaluation records, exact
   cross-run input binding and sequential multi-model batches. Acceptance: score
   two saved models on identical saved data without retraining; one failed model
   does not discard another's result; cancellation and retry preserve history.
4. **Comparison and selection.** Add multi-evaluation comparison, scoped
   leaderboards, tabular integration and optional model aliases. Acceptance:
   compatible scores rank correctly; mismatched splits, missing evidence and
   reused duplicates cannot create misleading winners or replicate counts;
   changing an alias cannot alter an already captured evaluation.

Each phase includes API/CLI parity, JSON schemas, compatibility tests and updated
user documentation. Validate publish/fetch round trips when adding portable
records. No new execution engine is required for the first release.

Later platform features
-------------------------------

Consider parameter sweeps and repeated-seed studies once model/evaluation
identity is stable; background execution and bounded parallel scheduling once
foreground batches work; quality gates and selection reports once comparison
semantics are established. Live metric streaming, a web UI, shared catalogs,
permissions, deployment/serving, drift monitoring and feature stores are separate
projects and are not required for this draft's initial workflow.
