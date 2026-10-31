Script pipeline validation
==========================

Pipelines describe named script steps and their typed input/output wiring.
``mlef validate`` checks these contracts without importing scripts, executing
top-level code, installing dependencies or starting Docker. It creates no run
storage. Existing ``mlef run script.py`` and historical run readers are unchanged.
This release validates definitions only; pipeline execution and artifact saving
are not yet implemented.

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
