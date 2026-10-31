Script pipeline validation
==========================

Pipelines describe named script steps and their typed input/output wiring.
``mlef validate`` checks these contracts without importing scripts, executing
top-level code, installing dependencies or starting Docker. It creates no run
storage. Existing ``mlef run script.py`` and historical run readers are unchanged.
Pipeline execution is not yet implemented. Scripts can already save typed outputs
and bind verified local inputs using the explicit SDK in :doc:`script-artifacts`.

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
execution is separate from this local demonstration. Existing
``mlef run script.py`` behavior remains unchanged.

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

Each binding maps a declared input to an exact ``step.output`` reference in this
pipeline, not a run ID, storage URL or implicit latest output. Steps must have
unique names matching their declarations. References must resolve, semantic
types and any requested formats must match, and the graph must be acyclic.
Independent ready steps use lexicographic ordering, regardless of TOML order.
An invalid graph has no executable order. Other errors may still return a graph
order for diagnostics; callers must check the report's overall status first.

Script, definition and explicit contract paths use normalized project-relative
POSIX syntax. Absolute paths, parent traversal, symlinks, missing files, Git
internals and run-storage paths are rejected. Paths resolve against the project
root, including when a definition lives in a subdirectory. Scripts must use the
``.py`` extension; definitions and contracts must use ``.toml``.

Computed contracts
------------------

The inspector reads the AST, not Python runtime values. It accepts literal
strings, numbers, booleans, nulls, lists and dictionaries; keyword-only
``Step(...)``, ``Input(...)`` and ``Output(...)`` constructors; and named
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
