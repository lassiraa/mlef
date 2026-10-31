Script pipelines
================

Pipelines describe named script steps and their typed input/output wiring.
``mlef validate`` checks these contracts without importing scripts, executing
top-level code, installing dependencies or starting Docker. It creates no run
storage. ``mlef run`` executes ``mlef.toml`` as a captured Docker pipeline;
``mlef run another.toml`` selects a different definition. Existing
``mlef run script.py`` runs individual scripts. Scripts save
typed outputs and bind verified inputs using :doc:`script-artifacts`.

Start with a script pipeline
----------------------------

``mlef init`` initializes storage and, when neither ``mlef.toml`` nor ``steps``
exists, writes a four-step example: ingest observations, split training/test
observations, train a constant predictor, and evaluate mean absolute error.
It uses only the standard library and the MLEF SDK, not an ML framework.
The data, partitions and model are generic JSON file/directory artifacts;
evaluation writes a structured ``Metrics`` payload with an exact model reference.
These toy artifacts do not claim the specialized dataset/split/model contracts.

Existing files, directories and symbolic links at either scaffold location cause
the *entire* example to be skipped; initialization still prepares storage and the
ignore entry. Repeating initialization never overwrites edits or recreates deleted
example files. Use ``mlef init --storage-only`` or ``api.init(scaffold=False)`` to
initialize storage without creating any example files.

From a fresh project's Git root, with MLEF installed in the active Python
environment, validate the wiring and run the trusted scripts locally in order:

.. code-block:: text

    mlef init
    mlef validate --json
    python steps/ingest.py .mlef/example
    python steps/split.py .mlef/example
    python steps/train.py .mlef/example
    python steps/evaluate.py .mlef/example

Every command takes the same parent directory. Each script creates its own new
attempt, reads verified read-only copies of upstream outputs, and publishes only
on a clean context exit. The example reports MAE 3 on two test observations for a
prediction of 2.5 learned from four training observations. Missing or corrupt
upstream outputs fail rather than silently rerunning ingestion.

For another execution, pass a different fresh parent directory to all four
commands. Existing attempt directories are never overwritten. These local SDK
attempts are **not** captured Docker runs and do not appear in ``mlef list``;
inspect them with ``read_step(Path(".mlef/example/evaluate"))``. Captured pipeline
execution is described next. Existing
``mlef run script.py`` behavior remains unchanged.

Captured Docker execution
-------------------------

Pipeline scripts use ``with step.context() as context:`` to obtain the explicit
input bindings and managed output directory assigned by the runner, then call
``step.save([...], context=context)``. The generated scripts use this mode when
called without a local directory argument. No context is inferred from a latest
run or from the current directory; calling ``step.context()`` outside a managed
pipeline fails. Keep using ``step.local(...)`` for explicit local SDK attempts.

Pipelines require a root ``pyproject.toml`` and existing ``uv.lock`` containing
the SDK runtime dependencies. In a fresh initialized project, create these with:

.. code-block:: text

    uv init --bare --python 3.12
    uv add "pydantic>=2,<3" "typing-extensions>=4.12"
    mlef validate
    mlef run

For an existing project, use ``uv add`` to add the missing runtime dependencies
to its own environment instead of replacing its project files or lock. The host
SDK is separate from experiment dependencies. The container uses Python 3.12, so
the experiment's Python requirement must allow it. MLEF never updates or resolves
the experiment lock during execution.

The runner captures the definition, all declared scripts and explicit contract
files, tracked project files, root dependency inputs and additional ``--include``
paths in **one** source archive. It revalidates the extracted snapshot, not live
edited files. It also captures its loaded pure-Python SDK and distribution metadata
as a deterministic wheel, retains the exact bytes and SHA256, and installs that
wheel into the locked experiment environment with ``--no-deps --no-index``. The
build verifies the wheel hash and runs ``uv pip check``; missing or incompatible
locked runtime dependencies fail preparation rather than triggering resolution.
The environment probe records installed package versions, including the SDK.

Steps run sequentially in the validated dependency order, as independent Python
processes within one container and one environment. The runner verifies each
script's captured hash before launching it. Inputs are verified read-only copies
from exact completed producers; absent optional outputs leave optional inputs
unbound. Each step currently has one attempt. There is no parallel scheduler,
automatic retry or implicit ingestion fallback. Artifact reuse is explicit, as
described below. Opt-in fingerprint caching is also available, without an implicit
cache search or background service.

A managed save stays pending even after a clean context exit. Only the runner
publishes it after the **whole script process** exits successfully and required
outputs, original/staged input bytes, output inventories, contracts and producer
references verify. This also applies to imperative managed saves without a
``with`` block. Exceptions, nonzero exits, abrupt termination, missing saves,
changed declarations or mutated payloads fail the attempt and prevent later
steps from starting. Completed earlier steps remain diagnostic evidence; partial
failed outputs are kept but cannot be loaded as reusable artifacts.

The existing supervisor owns the full worker lifetime, including preparation,
builds and all steps. ``--timeout`` is one wall-clock deadline for the entire run;
CPU, memory, network and combined log limits apply to the shared container/run.
Ctrl+C, timeout and explicit recovery preserve available artifacts, fail or cancel
the active attempt, and mark unstarted steps skipped. Engine failures can leave
cleanup pending; do not treat last-recorded liveness as proof that work stopped.
Scripts are trusted, not sandboxed from one another or from their own writable
artifact storage. Hard artifact quotas and arbitrary delayed-engine work remain
outside these controls; see :doc:`runs`.

Explicit reuse and resume
-------------------------

To change training without running ingestion again, edit the training script and
fork from an existing local run:

.. code-block:: text

    mlef run --reuse-from RUN --from-step train
    mlef run --reuse-from RUN --from-step train --resume

The first command captures the current checkout and SDK. The second creates a
new run from the source run's original source archive, dependency lock and SDK
wheel, ignoring live script/definition edits or deletions. It does not restore an
old container: base images are resolved and the locked environment is rebuilt by
the normal runner. Resume rejects ``--include`` and requires a captured SDK that
supports reuse. Older pipeline runs remain inspectable and can supply artifacts
to a fork, but cannot be resumed with a pre-reuse SDK.

Both commands require ``--reuse-from`` and ``--from-step`` together. The named
step and all its transitive consumers execute. Every other step, including
independent branches, must have a completed, verified attempt in the source run.
Selection follows dependencies, not positions in the TOML file. The source run
must be terminal and have no active owner or pending resource cleanup; recover
abandoned executions first. Failed and cancelled runs may supply completed
upstream attempts, never failed/partial attempts.

Before building the environment, MLEF verifies the saved source/SDK bytes,
contracts, script identities, output inventories and producer wiring. Reused
steps must retain their declarations, direct script hash and input bindings.
Changed downstream scripts are allowed in a fork. This is an explicit request to
freeze selected outputs, not a claim that changed helpers or environments would
produce identical outputs. This frontier does not perform fingerprint matching
and never falls back to ingestion. Opt-in caching is a separate mode below.
Missing, corrupt or incompatible reuse fails preparation instead of executing
the step again. To rerun ingestion explicitly, select ``--from-step ingest``.

Reused output directories and their manifests are copied, not linked, into the
new run and verified again. No source-run paths or cloud credentials are stored.
Original producer references and step-result run IDs are retained, including
through chains of forks; newly executed steps refer to these original producers.
The new pipeline records ``reuse_from``, ``from_step``, ``resume`` and each copied
attempt's ``reused_from`` run ID. The new run is self-contained for downstream
execution and verification even if the source run is moved or removed. Reused
attempts retain their original timestamps, not a new process execution time.

Python uses the same selection and validation:

.. code-block:: python

    fork = api.run(reuse_from=previous.id, from_step="train")
    resumed = api.run(reuse_from=previous.id, from_step="train", resume=True)

Explainable cache reuse
-----------------------

Cache reuse is opt-in per step and selects **one explicit local candidate run**.
It does not scan runs, consult a global index, contact remote storage or bypass
the environment build/probe. Declare deterministic steps in ``mlef.toml``:

.. code-block:: toml

    [[steps]]
    name = "ingest"
    script = "steps/ingest.py"
    cache = true
    cache_version = "immutable-public-data-v1"

    [[steps]]
    name = "train"
    script = "steps/train.py"
    bindings = { data = "ingest.data" }
    parameters = { seed = 42 }
    cache = true

This fragment illustrates policy, not a replacement for the generated example's
actual names/types. ``cache`` defaults to false. A step with no bound producers
is considered mutable ingestion and also requires a nonempty ``cache_version``.
That version is the caller's explicit assertion that external input is immutable;
MLEF does not discover or validate remote data. Bump it when external data changes.
Cacheable scripts must depend only on captured source, declared parameters,
verified inputs and the captured environment. Do not opt in scripts that depend
on wall-clock time, uncontrolled randomness, hidden environment variables, network
state, undeclared files or cross-step workspace mutations. Cacheability is a
promise by trusted code, not an automatic determinism proof.

Declare policy before the first run to warm its fingerprints, then select it:

.. code-block:: text

    mlef run
    mlef run --cache-from RUN
    mlef run --cache-from RUN --force train
    mlef cache NEW_RUN
    mlef cache NEW_RUN --json

.. code-block:: python

    result = api.run(cache_from=previous.id, force=["train"])
    pipeline = api.pipeline_status(result.id, verify=True)
    print(pipeline.attempts["train"].cache)

``--force STEP`` can be repeated and requires ``--cache-from``. Cache selection
cannot be combined with ``--reuse-from``, ``--from-step`` or ``--resume`` and is
not available for single-script runs. Forced steps run even on a matching key;
their dependents also run because their new input bytes are not known yet.
Independent matching branches may still be copied. Likewise, any mismatch or
disabled upstream step conservatively invalidates downstream reuse; outputs are
not speculatively compared after rerunning an upstream step.

The runner captures a versioned fingerprint for executed steps after their
upstream outputs exist. Components cover the **entire captured source tree's
paths, bytes and modes** (not archive timestamps), direct script bytes, declared
parameter JSON, exact input artifact content identities (including metadata and
absent optional inputs), the SDK wheel SHA256, script/pipeline contracts and their
versions. Environment identity includes immutable base/installer images, probed
Python/platform/package versions, dependency file hashes, installation recipe
and requested controls. The derived execution-image ID is deliberately not a
replacement for these content identities.

Source identity is deliberately conservative: editing any captured source,
definition, helper, parameter declaration or dependency file can invalidate every
step, even when the affected dependency is not used by that step. Use the explicit
frontier mode when deliberately freezing upstream evidence across code edits.
Changing only archive/file timestamps does not invalidate byte-identical source.

``pipeline.json`` records the selected ``cache_from``, source/environment
identities and every attempt's ``cache`` decision: action, human-readable reasons
and component fingerprints when known. Reasons distinguish forced execution,
disabled caching, unversioned ingestion, incomplete candidates,
upstream execution and individual changed components. Attempts without
fingerprints are unknown/misses, never guessed hits. ``mlef cache`` verifies a
terminal run and prints this evidence; its JSON output is the full versioned
pipeline record. ``pipeline_status(..., verify=True)`` also checks saved source,
environment and input fingerprints against their bytes.

The chosen run must be terminal, cleanup-complete and verifiable under an
ownership lock. Corruption fails preparation rather than silently recomputing.
Hits copy and reverify outputs and preserve original producer references,
including transitive hits. New runs do not depend on the candidate remaining
available. There is no artifact deduplication, retention policy, garbage collector
or automatic deletion.

Inspection and example
----------------------

Parent ``run.json`` records are inspected with the regular run commands.
Pipeline evidence is stored beneath that run's artifact directory:

.. code-block:: text

    .mlef/runs/RUN/
      source.tar.gz
      sdk/mlef-VERSION-py3-none-any.whl
      environment.json
      artifacts/pipeline/
        pipeline.json
        contexts/STEP.json
        steps/STEP/1/step.json
        steps/STEP/1/inputs/INPUT/
        steps/STEP/1/outputs/OUTPUT/

``pipeline.json`` records the captured contracts and wiring, dependency order,
source/SDK identities and attempt states/timestamps/exit codes. Its schema is
``schemas/pipeline-run-v1.schema.json``. Pipeline ``complete`` means every step
finished; the parent run still records any later execution/collection failure.
Step output manifests use the parent run UUID and declared producer step name.
Step status markers and script output share the parent's stdout/stderr logs.
Structured step metrics remain typed artifacts, not root ``metrics.json``
evidence. Run-level ``compare`` does not infer comparisons between these step metrics.

.. code-block:: text

    mlef status RUN --json
    mlef get RUN
    mlef get RUN pipeline/pipeline.json --output pipeline.json
    mlef get RUN pipeline --output saved-pipeline

From Python, ``api.run()`` defaults to ``mlef.toml``. ``api.pipeline_status(RUN)``
reads the structural pipeline record and checks its parent run identity without
executing scripts. ``verify=True`` additionally verifies the source archive,
captured SDK wheel, complete output inventories and exact input wiring, and
refuses verification while execution-resource cleanup is pending. Verification
also revalidates the source archive's definition, contracts and
script identities against the pipeline record without executing source code.
Ordinary inspection is read-only; recovery remains an explicit operation.

``examples/pipeline/demo.py`` creates a new Git project, exports the four scripts,
copies its supplied minimal locked environment and runs the complete Docker
pipeline through the Python API:

.. code-block:: text

    uv run --locked python examples/pipeline/demo.py ../mlef-pipeline-demo

The destination must not already exist. No ML packages are needed for this toy
constant-predictor example. Initialization never overwrites existing scripts.

Script declarations
-------------------

Declare exactly one top-level variable named ``step`` using literal keyword
arguments. For example, ``ingest.py`` can declare:

.. code-block:: python

    from mlef.pipeline import Output, Step
    from mlef.artifacts import ArtifactType

    step = Step(
        name="ingest",
        outputs=[Output(name="raw", type=ArtifactType.DATASET, format="csv")],
    )

A downstream ``train.py`` can declare:

.. code-block:: python

    from mlef.pipeline import Input, Output, Step
    from mlef.artifacts import ArtifactType

    step = Step(
        name="train",
        inputs=[Input(name="training", type=ArtifactType.DATASET, format="csv")],
        outputs=[Output(name="model", type=ArtifactType.MODEL, format="joblib")],
    )

Python constructors require ``ArtifactType`` enum members. The static inspector
also understands literal enum strings, but use enum members in runnable Python.
Inputs may omit ``format`` to accept any format of their semantic type. Both
inputs and outputs default to ``required=True``. An unbound optional input is
valid; a required input cannot depend on an optional output. Names start with an
ASCII letter and contain only letters, digits, underscores and hyphens. Each
step needs at least one output. Input names and output names must each be unique;
the two namespaces may overlap. ``Step.schema_version`` defaults to integer 1.

Typed step roles
----------------

Opt into a semantic role using the version-2 script contract:

.. code-block:: python

    step = Step(
        schema_version=2,
        kind="train",
        name="fit_baseline",
        inputs=[Input(name="training", type=ArtifactType.DATASET)],
        outputs=[Output(name="model", type=ArtifactType.MODEL, format="joblib")],
    )

The initial roles validate declared required inputs and outputs:

* ``generic`` accepts ordinary typed script contracts.
* ``train`` requires at least one required training input and a required model
  output. Training inputs can include custom artifact types.
* ``predict`` requires model and dataset inputs and a predictions output.
* ``evaluate`` requires a dataset and either a model or predictions as inputs,
  plus a metrics output. All metrics outputs use ``application/json``.

Roles are independent of step names. ``mlef validate`` checks these rules without
executing scripts, including declarations supplied in explicit contract files.
The runner retains the role in captured contracts and enforces ordinary managed
output publication. Specialized model/prediction metadata requirements still
apply; a role does not replace an artifact contract or implement model loading.

Output evidence is checked during SDK saves, clean context exit, managed
publication after process success, and verified pipeline inspection:

* A training model's optional ``training_dataset`` must reference a bound dataset
  or the dataset identified by a bound split. Omitting it preserves unknown
  training lineage.
* Prediction metadata must reference the exact bound model and dataset. An
  optional split must be bound and identify that dataset.
* Evaluation metrics must be one structured JSON payload per metrics artifact,
  identify the bound dataset, and identify either a bound model or bound
  predictions. A declared split must identify a bound split for that dataset.
* When scoring bound predictions, the metric model, dataset, split and scope
  must match their prediction metadata exactly. Direct model evaluation can
  declare its scope; there is no protocol scope to validate against yet.

Each evaluation JSON payload has a 32 MiB limit at publication and is checked
against its saved size/checksum before parsing. Use the existing external metric
representation for large series/tables; these references remain unresolved.
Malformed or inconsistent evidence fails publication and retains failed attempt
diagnostics. Generic steps retain their existing behavior.

These checks validate declared identities and payload structure. They do not
execute a scorer, establish evaluation protocol identity, validate actual row
usage, or prevent trusted scripts from fitting on evaluation data. See
:doc:`evaluations` for frozen evaluation recipes and sequential model batches.
``read_step`` alone
verifies saved output inventories; ``pipeline_status(..., verify=True)`` also
rechecks typed role evidence against the captured producer wiring.

Existing declarations default to ``schema_version=1`` and ``kind="generic"``.
Version-1 serialization omits ``kind`` and retains its original field order,
preserving saved contract fingerprints and compatibility with captured old SDKs.
Non-generic roles require version 2 explicitly. The current reader accepts both
versions, while old readers/SDKs cannot execute version-2 declarations. The
published ``step-v1`` structural schema remains unchanged; ``step-v2`` describes
the current reader, with role/version and input/output invariants additionally
enforced by Pydantic.

The version above belongs to each embedded script contract. Local-only pipeline
definitions can keep ``schema_version=1``; cross-run bindings require version 2
of the pipeline definition, independently of the script contract version.

Wiring with TOML
----------------

Store this definition in ``mlef.toml`` at the Git project root:

.. code-block:: toml

    schema_version = 1

    [[steps]]
    name = "ingest"
    script = "ingest.py"

    [[steps]]
    name = "train"
    script = "train.py"
    [steps.bindings]
    training = "ingest.raw"

In version-1 definitions, each binding maps a declared input to an exact
``step.output`` reference in this pipeline. Version-2 definitions also support
the explicit cross-run references below. Steps must have
unique names matching their declarations. References must resolve, semantic
types and any requested formats must match, and the graph must be acyclic.
Independent ready steps use lexicographic ordering, regardless of TOML order.
An invalid graph has no executable order. Other errors may still return a graph
order for diagnostics; callers must check the report's overall status first.

Cross-run inputs
----------------

Use a version-2 pipeline definition to bind saved artifacts from another captured
run. Local ``step.output`` bindings still work and can be mixed with exact
external references. For example, an evaluation-only pipeline can bind:

.. code-block:: toml

    schema_version = 2

    [[steps]]
    name = "evaluate"
    script = "steps/evaluate.py"

    [steps.bindings.model]
    run_id = "LOCATION-RUN-UUID"
    [steps.bindings.model.artifact]
    run_id = "ORIGINAL-PRODUCER-RUN-UUID"
    step = "train"
    output = "model"
    sha256 = "FULL-ARTIFACT-CONTENT-SHA256"

Replace these placeholders with complete canonical UUID4 identifiers and the
64-character artifact content hash from saved evidence. Declare other required
inputs, such as evaluation data, in the same way. The outer run ID locates the
saved copy; the inner artifact reference identifies its original producer. They
are equal for directly produced outputs and may differ for reused artifacts.
No alias, path search, implicit latest run or remote fetch is performed.

``mlef validate`` checks external reference structure without reading run storage.
It reports that payload/type verification will happen during preparation. The
runner then requires terminal, cleanup-complete source runs without active owners
and verifies their captured source, SDK, completed output inventories, contracts
and wiring. Exact producer identities and input types/formats must match; missing
or incompatible evidence fails preparation rather than rerunning training.

The runner copies each selected producer's complete step outputs into
``artifacts/pipeline/imports/HASH`` and verifies them again. This may include sibling
outputs from that producer step. The new run remains self-contained for execution,
verified inspection and artifact transfer after original source runs are removed.
Producer references remain unchanged. Embedded metric/model references do not
cause automatic traversal to other artifacts; bind any required targets explicitly.

External inputs create no dependencies on steps in the new pipeline. Cache
fingerprints include their verified content and exact binding declarations.
Explicit reuse still compares input bindings; normal forks and cache runs stage
their external inputs afresh from the declared source locations. ``--resume``
copies the saved imports from the resumed run, so its original external sources
need not remain available. Copies retain original producer timestamps and are
not new training executions.

Python callers can construct ``ExternalInput(run_id=location, artifact=reference)``
from ``mlef``. For registered models, the location is
``version.artifact.location_run_id`` and the reference is
``version.artifact.manifest.producer``. Pipeline definitions still contain fixed
references when captured; :doc:`evaluations` selects registered models and
generates these bindings for evaluation batches. Older SDKs cannot read version-2 pipeline definitions. The
published ``pipeline-v1`` schema remains unchanged; ``pipeline-v2`` describes the
current reader, with the version/binding restriction enforced by Pydantic.

Script, definition and explicit contract paths use normalized project-relative
POSIX syntax. Absolute paths, parent traversal, symlinks, missing files, Git
internals and run-storage paths are rejected. Paths resolve against the project
root, including when a definition lives in a subdirectory. Scripts must use the
``.py`` extension; definitions and contracts must use ``.toml``.

Computed contracts
------------------

The inspector reads the AST, not Python runtime values. It accepts literal
strings, numbers, booleans, nulls, lists and dictionaries; keyword-only
``Step(...)``, ``Input(...)``, ``Output(...)`` and ``CustomType(...)`` constructors; and named
``ArtifactType`` enum members. It does not follow aliases, variable references,
imports, helper functions, comprehensions, attribute calls or expanded arguments.
Annotations on ``step`` are allowed. Multiple top-level assignments or a computed
declaration are reported as deferred, never silently executed.

For a computed declaration, provide an explicit contract using the same step
schema. For example, ``train-contract.toml``:

.. code-block:: toml

    schema_version = 1
    name = "train"

    [[inputs]]
    name = "training"
    type = "dataset"
    format = "csv"

    [[outputs]]
    name = "model"
    type = "model"
    format = "joblib"

Add ``contract = "train-contract.toml"`` to the corresponding ``[[steps]]`` entry,
before its ``[steps.bindings]`` table. The explicit contract is authoritative for
static checking; script existence and syntax are still checked. It does not
prove that runtime declarations agree. More generally, static declarations are
not Python execution analysis: rebinding constructors, changing a declaration
later or hiding work in control flow cannot establish runtime correctness.
Runtime output/payload validation remains a separate requirement.

CLI and Python reports
----------------------

.. code-block:: text

    uv run --locked mlef validate
    uv run --locked mlef validate path/to/pipeline.toml --json

Run from the exact Git project root. Exit codes are 0 for verified, 1 for invalid
and 3 for deferred; ordinary CLI argument/project errors retain exit code 2.
JSON reports include ``schema_version``, ``status``, the parsed ``definition``,
verified ``contracts``, dependency ``order`` and individual ``checks`` with
status, code, message and optional step name. Definition errors return an invalid
report without a definition. Invalid takes precedence over deferred.

.. code-block:: python

    from mlef import api

    report = api.validate("mlef.toml", root="/path/to/project")
    if report.status == "verified":
        print(report.order)

``api.validate`` returns a ``ValidationReport`` without printing. Definition,
script and wiring errors appear as checks rather than exceptions. Invalid Git
root configuration still raises ``api.MlefError``. Lower-level callers can use
``mlef.pipeline.validate(root, definition)`` without requiring Git.

Verified means only the requested static checks passed. It does not attest to
payload bytes, actual dataset columns, row identities, installed packages,
model deserialization safety or script success. Unknown producer contracts defer
dependent compatibility checks rather than pretending that those checks passed.

Public structural schemas are ``schemas/step-v1.schema.json``,
``schemas/pipeline-v1.schema.json`` and
``schemas/pipeline-validation-v1.schema.json``. Pydantic and the validator enforce
cross-field and graph invariants beyond JSON Schema. See :doc:`artifacts` for
artifact types and evaluation evidence.
