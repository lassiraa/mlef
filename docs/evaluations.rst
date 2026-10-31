Evaluation batches
====================

Evaluate saved model versions with one frozen evaluator and explicit saved data:

.. code-block:: text

    mlef evaluate evaluation.toml --model classifier@VERSION_A --model classifier@VERSION_B
    mlef evaluations list --limit 50 --json
    mlef evaluations show BATCH --json
    mlef models evaluations classifier@VERSION_A --json
    mlef runs metrics EVALUATION_RUN --step evaluate --json

Replace model selectors with complete version hashes from :doc:`models` and
BATCH/EVALUATION_RUN with the UUIDs in the saved batch response. Execution uses
Docker; history, descriptors and metric queries work offline.

Define an evaluator
-------------------

The evaluation definition is distinct from a pipeline definition. For example:

.. code-block:: toml

    schema_version = 1
    script = "evaluate.py"
    step = "evaluate"
    model_input = "model"
    parameters = { seed = 42 }

    [bindings.data]
    run_id = "LOCATION-RUN-UUID"
    [bindings.data.artifact]
    run_id = "ORIGINAL-PRODUCER-RUN-UUID"
    step = "data"
    output = "dataset"
    sha256 = "FULL-ARTIFACT-CONTENT-SHA256"

Use actual canonical UUID4 identifiers and a full 64-character artifact content
hash. Bind any required split or auxiliary artifacts the same way. These are the
explicit cross-run bindings described in :doc:`pipelines`. ``model_input`` is
assigned by candidate selection and cannot also appear in ``bindings``.

The evaluator must declare a matching literal version-2 ``Step`` with
``kind="evaluate"``, required model/dataset inputs and a JSON metrics output.
``step`` and ``model_input`` default to ``evaluate`` and ``model``. An optional
``contract`` names an explicit TOML script contract. ``includes`` lists additional
project-relative source paths to capture, including untracked helper code.

The script uses ordinary ``step.context()``, verified input paths and
``step.save``. It loads the model using its own trusted code and locked framework
dependencies and emits structured ``Metrics`` with exact model/data references.
Publication follows :doc:`pipelines` and its typed role lineage checks.
MLEF supplies orchestration and evidence rather
than a universal model loader or scoring library.

Frozen source and recipe identity
---------------------------------

Before starting candidates, MLEF resolves 1 through 100 distinct exact model
versions, captures evaluator source, the definition, dependency files and explicit
includes, and captures the loaded SDK wheel. Each candidate receives its own
generated evaluation-only pipeline inside a frozen source archive. Editing live
source after preparation does not change later candidates. Training steps are
not executed by these generated pipelines.

Each batch records a ``protocol_id`` derived from captured source content/modes,
SDK bytes and the evaluation specification, including parameters and saved
data/split references. This identifies the shared evaluation recipe. Actual
resolved images, Python/packages and resource controls remain part of each child
run's normal environment evidence. A shared recipe ID alone does not establish
statistical significance, actual row usage, leakage freedom or hardware-equivalent
runtime measurements. The comparisons below verify these declarations and
environment evidence separately.

Source/SDK bundles and their checksums are verified again when each child worker
restores them. Child runs retain their generated definition, source archive,
SDK, logs, metrics and self-contained imported inputs. A root ``evaluation.json``
descriptor records batch/recipe identity, the selected model version and the
specification; ordinary run publication includes that descriptor. Each candidate
has its own container, output directory and normal run UUID.

Outcomes and inspection
-----------------------

Candidates run sequentially. A failed candidate keeps its evidence and does not
prevent independent candidates from running. Batches finish as ``complete`` when
all candidates succeed, ``partial`` when some succeed, ``failed`` when none do,
or ``cancelled`` on cancellation. Intermediate states are ``preparing`` and
``running``. Per-candidate states also distinguish ``pending`` from active work.

Ctrl+C cancels the active supervised run and stops subsequent candidates; their
reserved UUIDs remain pending without claiming that a run was executed. Query
commands show last-saved states, not confirmed process liveness. After abrupt
process death, recover abandoned child runs explicitly with ``mlef recover RUN``;
automatic batch reconciliation and retry commands are not implemented. Rerunning
``evaluate`` creates a fresh batch and never overwrites earlier results.

Batch records retain model versions, child IDs, diagnostics, exact metric artifact
references and compact scalar/undefined summaries. Series, curves, matrices,
tables, distributions and external descriptors stay in child metric artifacts;
use ``runs metrics`` for their full structured envelopes. A valid evaluation can
have no scalar summaries. No metric is implicitly selected as a winner.

``evaluations list`` and ``models evaluations`` return a versioned ``batches``
envelope with diagnostics, total matching batches and ``next_offset``. Results
are newest first with batch UUID as tie breaker. Limits default to 50 and range
from 1 through 1000; offsets apply after exact model-selector filtering. Concurrent
new batches can shift pages. ``evaluations show`` takes a batch UUID; inspect an
individual evaluation using the ordinary run commands and its child UUID.

Storage lives beneath ``.mlef/evaluations/BATCH``. Catalog inspection is read-only;
corrupt records are reported through diagnostics. Source/SDK bundles are copied
per candidate initially, and no data is deleted automatically. Run transfers
include each child's evidence, but do not automatically transfer the batch or
model catalogs.

Limits, CLI exits and Python API
--------------------------------

``evaluate`` accepts ``--timeout``, ``--cpus``, ``--memory``, ``--network``,
``--max-log-bytes`` and ``--max-runner-log-bytes``. These controls apply to each
child run independently; there is no aggregate batch deadline or disk quota.
Normal dependency and Docker prerequisites apply to the captured environment.

The CLI returns 0 for a complete batch, 1 for partial/failed execution, 2 for
configuration errors and 130 for cancellation. ``--json`` emits the saved batch
envelope without forwarding worker logs. API callers inspect the returned typed
batch state:

.. code-block:: python

    from mlef import api

    batch = api.evaluate(
        "evaluation.toml", ["classifier@VERSION_A", "classifier@VERSION_B"],
        options=api.RunOptions(timeout_seconds=300),
    )
    history = api.list_evaluations(model="classifier@VERSION_A", limit=20)
    saved = api.evaluation_batch(batch.id)

All methods accept the optional project ``root``. Schemas are exported as
``evaluation-spec-v1``, ``evaluation-batch-v1`` and ``evaluation-listing-v1``.

Comparisons and leaderboards
----------------------------

Choose a scalar metric and an explicit successful baseline evaluation run:

.. code-block:: text

    mlef evaluations compare EVAL_A EVAL_B --baseline EVAL_A --metric mae --scope test
    mlef evaluations leaderboard --batch BATCH --baseline EVAL_A --metric mae --scope test --json

Comparison accepts 1 through 100 distinct evaluation run UUIDs. They can belong
to different batches. Leaderboards select the child UUIDs of one saved batch.
These queries work offline and do not execute evaluator or model code.

Metric names must identify one artifact per evaluation. Use ``--scope`` and
``--output`` to qualify ambiguous names; ``--output`` here is the metrics artifact
output name, not a destination path. Curves, series, tables, distributions,
matrices, external descriptors and undefined values are never implicitly reduced
to scalar scores. Undefined reasons remain visible.

Queries reverify captured source/SDK, pipeline wiring, typed role evidence and
metric inventory bytes. They reconstruct recipe identity from captured source
excluding the generated model-specific definition, and check that the context's
model selection and specification match that definition. Individual comparisons
use each child's ``evaluation.json`` and do not require the original batch/model
catalog or training run. The context descriptor is limited to 1 MiB when read.

Per-row compatibility is ``declared-compatible``, ``incompatible`` or ``unknown``.
Known recipe, environment/control, dataset, split, scope or metric-semantic
conflicts take precedence over missing information. Metric checks reuse
``Metrics.semantic_differences`` for definitions/versions, units, directions,
targets, aggregation, parameters, dimensions and sample counts. Both absent split
references are acceptable under the same verified recipe and dataset; actual
row use is still a declaration. Missing sample counts or complete environment
evidence prevent ranking.

Only successful, compatible scalar observations with a declared ordering rank.
Maximize/minimize use the scalar value; target metrics use distance from their
declared target. Direction ``none`` has no ranking. Ties share competition ranks
(1, 1, 3). Repeated model producers or reused metric evidence are excluded from
independent model rankings and identify the primary row through ``duplicate_of``
(the baseline takes precedence).
Leaderboards additionally require successful saved batch outcomes, including the
baseline. Unstarted and failed candidates remain visible with reasons.

Rows include values, signed deltas, relative deltas, scope/output, artifact/data
references, sample counts and any supplied intervals. Relative deltas use
``(candidate - baseline) / abs(baseline)`` and are absent for a zero baseline.
Unrepresentable arithmetic remains unavailable with a reason. Incompatible
evidence may still show raw values/deltas, but cannot support an improvement
observation or rank. Corrupt or missing individual evidence produces an unknown,
unranked row rather than trusting cached batch summaries.

An observed improvement means an ordered score difference under verified
declarations. It does not establish significance, held-out gains, leakage freedom
or hardware-normalized runtime performance. Queries return exit 0 for a valid
comparison even when rows are unranked; inspect ``compatibility`` and ``eligible``
in automation. Ambiguous metric selection returns a categorized query error.

.. code-block:: python

    comparison = api.compare_evaluations(
        [baseline_run, candidate_run], baseline=baseline_run,
        metric="mae", scope="test",
    )
    table = api.evaluation_leaderboard(
        batch.id, baseline=baseline_run, metric="mae", scope="test",
    )

Both APIs accept optional ``output`` and project ``root``. Their versioned schema
is ``evaluation-comparison-v1``; the portable child descriptor schema is
``evaluation-context-v1``. A script may execute successfully while its batch item
fails verification if metric payloads identify a model other than the selected
version. Such evidence cannot rank as that model.
