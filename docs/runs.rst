Run and CLI reference
=====================

:doc:`Documentation home <index>`

Commands
--------

Namespaced history queries provide bounded offline inspection:

.. code-block:: text

    mlef runs list --state succeeded --limit 50 --offset 0 --json
    mlef runs list --since 2026-10-01T00:00:00Z
    mlef runs show RUN --json
    mlef runs steps RUN --json

History JSON is a versioned envelope containing ``runs``, ``diagnostics``,
``total`` matching records and ``next_offset`` (null on the final page).
Results are newest first with run ID as a stable tie breaker. Offsets apply
after filtering; concurrent new runs can shift pages. ``--since`` is inclusive
and requires a timezone offset. Limits are 1 through 1000, defaulting to 50.
``runs show`` includes saved run/result evidence; ``runs steps`` returns the
pipeline record and requires a captured pipeline run. Queries never start Docker.
Python callers use ``api.run_history`` with the same filters and pagination.
The existing ``list`` JSON array and ``status`` response remain unchanged.

Inspect root and completed pipeline metric evidence together:

.. code-block:: text

    mlef runs metrics RUN --json
    mlef runs metrics RUN --step evaluate --scope test --metric mae --json

``api.run_metrics(RUN, step=None, scope=None, metric=None)`` returns the same
versioned envelope. ``root`` retains legacy result evidence, including missing or
invalid status; it is omitted when filtering by step or scope because legacy
metrics have no verified step/scope identity. ``artifacts`` contains structured
metric envelopes with step, output and exact producer references. Name filters
return every matching qualified artifact rather than choosing an ambiguous score.
No ranking or comparison is implied.

Pipeline queries verify completed output inventories, contracts and wiring.
Incomplete attempts produce diagnostics; corrupt evidence fails the query.
JSON metric artifacts must contain one payload file. Each payload is limited to
32 MiB by default (``--max-bytes`` / ``max_bytes``), checked again against saved
size and checksum after reading. This is a per-payload limit, not a total memory
or I/O quota; inventory verification still reads saved output bytes. External
metric references remain compact and are never resolved automatically. These
queries do not verify source archives or rerun evaluators; use
``api.pipeline_status(RUN, verify=True)`` for full captured-source verification.

Run commands from the experiment's Git root. Replace RUN, BASELINE and CANDIDATE
with complete UUID4 identifiers printed by MLEF. Use ``mlef COMMAND --help``
for argument help.

.. code-block:: text

    mlef init
    mlef init --storage-only
    mlef run
    mlef run another-pipeline.toml --timeout 180
    mlef run --reuse-from RUN --from-step train
    mlef run --reuse-from RUN --from-step train --resume
    mlef run --cache-from RUN --force train
    mlef cache RUN --json
    mlef run train.py --include data.csv --output results -- --seed 7
    mlef list --json
    mlef status RUN --json
    mlef get RUN
    mlef get RUN metrics.json --output saved-metrics.json
    mlef compare BASELINE CANDIDATE --json
    mlef recover RUN --json
    mlef publish RUN --to s3://my-bucket/experiments --json
    mlef fetch RUN --from s3://my-bucket/experiments --json
    mlef tabular compare BASELINE CANDIDATE --limit 20 --json
    mlef tabular profile RUN --json
    mlef tabular export RUN --data public.csv --output .mlef/shared/RUN --profile
    mlef tabular diagnose BASELINE CANDIDATE --minimum-count 20 --limit 20 --json

``init`` also creates a :doc:`four-step example <pipelines>` when neither
``mlef.toml`` nor ``steps`` exists. Existing scaffold locations are left untouched.
``--storage-only`` creates only storage and its ignore entry.

``publish`` / ``fetch`` explicitly transfer terminal runs through local directories
or S3, with verified bytes and immutable publication. See :doc:`storage` for the
optional host dependency, credentials, retries and privacy limitations.

``run`` without a path executes ``mlef.toml``; a ``.toml`` path selects another
:doc:`captured script pipeline <pipelines>`. Pipeline definitions require a locked
experiment environment and scripts using ``step.context()``. Script arguments,
``--output`` and ``--experiment`` apply only to single-script runs, not
pipelines. Shared run limits and ``--include`` also apply to pipeline runs.
Explicit ``--reuse-from RUN --from-step STEP`` copies verified completed attempts
outside the selected step's downstream dependency frontier. Add ``--resume`` to
retain the original source/SDK snapshot instead of capturing edited source;
``--include`` is not accepted for resume. Both modes create a new run and require
a terminal source run with no active owner or pending resource cleanup. See
:doc:`pipelines` for compatibility checks and producer lineage semantics.

Outputs, cancellation and limits
--------------------------------

Scripts write durable results to the directory in ``MLEF_OUTPUT_DIR``. To collect
another directory written under the project working directory, use
``mlef run train.py --output results``. The script argument separator is ``--``.
Declared outputs are required: missing outputs or destination conflicts fail
the run without overwriting existing artifacts.
Use Ctrl+C to cancel; mlef gives the container five seconds to stop before
forcing termination and collecting available outputs.

``mlef run train.py --timeout 60`` places one wall-clock deadline around the entire
worker, including source preparation, image pulls/builds, environment probes and
execution. On expiry, mlef kills the owned worker process tree and attempts
ownership-checked container/output cleanup within a further 15-second grace budget
(process-tree termination may take up to ten additional seconds). Available outputs
remain retrievable; interrupted evidence is marked unavailable. Engine failures
are recorded as incomplete cleanup rather than claimed successful cleanup. The
deadline does not quota artifact writes or cancel arbitrary remote engine work.
Omitting the option preserves the existing foreground behavior and stage timeouts.

Runtime controls: ``mlef run train.py --cpus 0.5 --memory 64m --network none``.
Memory accepts positive bytes or integer k/m/g/t suffixes and disables additional
swap; CPU values use at most nine decimal places. Docker capability checks and
container inspection must confirm requested limits before execution. Requested
and applied values are recorded separately. The execution probe receives CPU and
memory controls and always uses isolated networking. Runtime controls do not
constrain build workers or make image/dependency downloads offline. Docker-confirmed
OOM termination is recorded separately from a script's explicit exit code 137.

``--max-log-bytes 1000000`` bounds combined saved and forwarded script stdout/stderr;
``--max-runner-log-bytes 1000000`` bounds preparation/build diagnostics in runner.log.
Limits count bytes, including multibyte text, and stop the run on exhaustion with
an ``output_limit`` diagnostic. Native command pipes are drained through the limiter;
they never write directly to the saved log descriptor. Cleanup output is discarded
when necessary and its outcome is retained in metadata. Structured Docker replies
have a separate 4 MiB capture ceiling. These limits do not quota artifacts or total
disk use; combine them with a deadline for a total worker time budget.

Abandoned runs
--------------

``mlef recover <run-id> [--json]`` explicitly finalizes an abandoned owned run.
Kernel locks protect live runners, supervising parents and concurrent recovery;
ordinary status/list queries remain read-only. Recovery requires the recorded
project identity and exact Docker ownership labels, refuses unknown
ownership, and leaves metadata unchanged when Docker is unavailable. Missing
containers never imply success. Collected outputs are never overwritten; a
collection failure retains the container for inspection/retry. Recovery marks an
unfinished run failed with an ``abandoned`` diagnostic and can be safely repeated.
Terminal runs without pending cleanup are returned unchanged, including offline.

.. _captured-environment:

Captured source and environment
-------------------------------

The runner archives tracked files and the explicitly named script into
``source.tar.gz``, then builds from that saved archive. Include other untracked
source with ``--include path``. Each run saves the archive SHA-256, included
untracked paths, Git commit/branch/dirty state, a binary-capable ``changes.patch``
relative to HEAD (the empty tree for an unborn repository), and the generated
``Dockerfile``. Detached HEAD records a null branch. The archive defines the actual
source inputs; the patch contains tracked changes only.
Root dependency manifests and locks are included even when untracked.
Runs use Pydantic metadata.
It resolves ``python:3.12-slim`` to an immutable digest and executes from ``/workspace``; your working tree
is never mounted into the experiment. Run storage and credential files are
excluded. Symlink inputs are currently unsupported. Changes made while source
is being copied can affect the copy, so avoid editing inputs during preparation.

Dependencies use an existing ``pyproject.toml`` plus ``uv.lock``. The pinned uv 0.13.6
installer runs ``uv sync --locked`` with the container's Python, no dependency groups
or extras, and no installation of the project itself. Registry packages must have
compatible wheels; source builds, workspace/path/Git dependencies, alternative
dependency managers and separate ``uv.toml`` configuration are unsupported. A stale
lock or unsupported input fails preparation and preserves the captured inputs.
Scripts with no dependency manifest run without uv. A tooling-only pyproject
without runtime requirements can also run without a lock; declared dependencies,
dynamic metadata or Python requirements need a lock.

``environment.json`` records the host, resolved base/uv/execution image identities,
install command, and actual container Python, OS and package versions. ``run.json``
also records dependency and Dockerfile hashes. Capture failures prevent execution
and retain an incomplete-environment warning. Host environment variables are not
collected. Recorded images and packages still need to be available for a rebuild;
datasets, network responses and nondeterminism remain the caller's responsibility.
Background execution is still planned. Experiment code has network access by
default and can write to its artifact mount.

Offline inspection and artifact retrieval
-----------------------------------------

Inspect runs without Docker using ``mlef list [--json]`` and
``mlef status <run-id> [--json]``. ``list`` orders newest first, skips corrupt records
with stderr diagnostics, and returns a JSON array. An unfinished state describes
the last saved state; current process liveness is unknown. Unknown IDs, unsupported
versions, and corrupt records return exit 1; JSON failures include a structured
error on stdout. Invalid arguments return exit 2.

``mlef get <run-id>`` lists artifacts. Retrieve a file or directory with
``mlef get <run-id> <artifact-path> --output <destination>``; the default destination
is the artifact basename in the current directory. Existing destinations, escaping
paths, links and special files are refused. Full UUID4 run IDs are required.

Experiment intent, metrics and generic comparison
-------------------------------------------------

Use ``mlef run train.py --experiment experiment.json`` to store a hypothesis,
``parent_run_id``, ``changed_variables``, canonical ``parameters``, training ``seeds``,
``dataset`` reference/fingerprint, ``evaluation`` identity, and ``metric_directions``.
The version-1 specification rejects unknown fields, invalid types and non-finite
numbers before allocating a run. Parent IDs must already exist. Metadata is saved
by value and does not inject settings into the script or clone a parent's source.

Scripts may write ``metrics.json`` under ``MLEF_OUTPUT_DIR``: a flat object of finite
numbers, such as ``{"accuracy": 0.9}``. Booleans, duplicate keys, nested values, NaN
and infinity are invalid. ``result.json`` records normalized metrics, artifact
references and stable diagnostics. Missing/invalid optional metrics remain
retrievable and do not turn a successful script into a failed run. Missing or
corrupt saved results are displayed as unavailable/invalid evidence.

``mlef compare <baseline-id> <candidate-id> [--json]`` reports the union of metrics,
absolute deltas and relative deltas ``(candidate - baseline) / abs(baseline)``
(unavailable for a zero baseline), parameter changes, provenance, execution time,
intentional changes and artifact references. Human status/comparison summaries
show at most 20 metrics; JSON includes all metrics without embedding logs.
Per-metric compatibility is ``declared-compatible``, ``incompatible`` or ``unknown``.
Compatible declarations require matching nonempty dataset fingerprints,
evaluation ``protocol_id`` and ``split_fingerprint``, explicitly supplied evaluation
``seeds`` (an empty map is valid), metric definitions, and maximize/minimize
directions. Known conflicts take precedence over missing information; unsuccessful
runs and missing evidence cannot support improvement claims. These checks compare
declarations, not actual split use, statistical significance or leakage freedom.

Rebuilding saved inputs
-----------------------

To manually repeat saved inputs after editing the working tree, create a fresh
build directory containing ``workspace/`` extracted from the saved ``source.tar.gz``
and the saved ``Dockerfile``. Verify the recorded SHA-256 values first. Archives
created by mlef contain regular files; for external archives use an extractor that
rejects escaping paths and links. Build with ``docker build -t mlef-repeat .``, then
execute the saved ``run.json`` command in that image with ``/workspace`` as its working
directory and ``MLEF_OUTPUT_DIR=/outputs`` bound to a fresh output directory. Use the
same user ID/group as the original run when reproducing permissions. The saved
recipe pins the original base and uv image digests and installs from the archived
lock. Retrieve declared ``--output`` directories before removing the container.

.. _run-exit-codes:

Exit codes and troubleshooting
------------------------------

.. list-table:: Exit codes
   :header-rows: 1

   * - Code
     - Meaning
   * - 0
     - Successful execution/query, including incompatible or unknown comparisons.
   * - 1
     - Execution, preparation, storage, collection or query/retrieval failure.
   * - 2
     - Invalid arguments, experiment metadata or project configuration.
   * - 130
     - Foreground run cancelled by the user.

For installation errors, see :ref:`setup troubleshooting <setup-troubleshooting>`.
Use the complete UUID printed by the runner. Retrieval never overwrites an existing
destination. If metrics are missing/invalid, inspect ``result.json`` diagnostics
and retrieve the original artifact. If a run remains unfinished after its owner
has exited, inspect saved evidence and use ``mlef recover RUN``.
