Saving script artifacts
=======================

Scripts can save typed file and directory artifacts without Git, Docker, an ML
framework or a storage service. The SDK uses the :doc:`artifacts` contracts and
:doc:`pipelines` declarations. It does not yet schedule or execute a pipeline.
Existing ``mlef run script.py`` behavior and historical run records are unchanged.

Local contexts and managed paths
--------------------------------

The main SDK entry points are exported from ``mlef``:

.. code-block:: python

    from pathlib import Path
    from mlef import Artifact, ArtifactType, Output, Step, read_step

    step = Step(
        name="report",
        outputs=[Output(name="summary", type=ArtifactType.TEXT, format="text/plain")],
    )
    with step.local(Path(".mlef/report-attempt")) as context:
        path = context.output_path("summary", "summary.txt")
        path.write_text("An ordinary script produced this evidence.", encoding="utf-8")
        step.save(
            [Artifact(name="summary", type=ArtifactType.TEXT, format="text/plain", path=path)],
            context=context,
        )
    record = read_step(Path(".mlef/report-attempt"))
    assert record.state == "complete"

``Step.local(directory, *, inputs=None, run_id=None)`` opens an explicit
``LocalContext`` for a new attempt. It creates parent directories, but never
overwrites an existing attempt directory. ``run_id`` is an optional canonical
UUID4 string; by default each context generates its own ID. A caller coordinating
several steps can supply one run ID. Paths become absolute when the context or
input locator is constructed; later working-directory changes do not redirect it.

``context.output_path(name, relative=None)`` checks the declared output name,
creates required directories and returns a managed ``Path``. With no relative
path it returns the output directory; with a relative path it creates only the
parent directories. Relative names use normalized POSIX syntax. Absolute paths,
parent traversal, symlinks, Windows reparse points, hard links and special files
are rejected. The caller chooses the trusted attempt directory; payload paths
must stay inside the corresponding output's managed directory.

``Artifact`` carries ``name``, ``type``, ``format``, ``path`` and optional
specialized ``metadata``. Its type and format must match the output declaration.
For datasets, splits, predictions and models, supply the metadata models from
``mlef.artifacts``. The SDK inventories actual file bytes, attaches an exact
producer reference and revalidates metadata before accepting the artifact.
It does not infer dataset columns or execute model deserializers.

A file artifact has one payload file. A directory artifact may contain several
files and nested directories, with inventory paths relative to the managed output
directory. Files outside a declared payload subtree are rejected. Empty artifacts
are unsupported; empty subdirectories do not contribute to content identity.

Saving, serialization and atomic finalization
---------------------------------------------

Call ``step.save(artifacts, *, context)`` once with the complete list of outputs.
Names must be unique and declared, every required output must be present, and
payloads must exist and satisfy their artifact contracts. Optional outputs may
be omitted. Unreported output directories and extra files outside a file payload
are rejected, rather than silently published. The context snapshots its step
declaration and rejects subsequent changes. Finalized contexts cannot save again
or allocate further output paths; retries require new attempt directories.

If a third-party serializer should write the payload, pass a trusted callback
``serializer(path) -> None`` on the ``Artifact``. It receives the managed path and
must create the file or directory there. The path must not already exist; no
overwrite is permitted. For a directory serializer, allocate a non-existing child
path such as ``context.output_path("model", "bundle")``, not the pre-created output
directory. No serializer dependency is installed or imported automatically.

Use ``with step.local(...) as context`` for failure-safe scope handling:

* The initial ``step.json`` has state ``running`` and cannot supply reusable outputs.
* ``step.save`` verifies bytes and saves a pending record. Its returned
  ``StepResult`` still has state ``running`` while the context is open.
* Successful context exit rechecks inputs and outputs, then atomically replaces
  ``step.json`` with a ``complete`` record. Only then may another step consume it.
* An exception, missing save, changed payload or finalization failure leaves a
  ``failed`` record where possible. Partial payloads and already computed manifests
  remain as diagnostic evidence and are never accepted for reuse.
* Abrupt termination before context exit leaves a non-reusable ``running`` record.

Manifest replacement uses a unique temporary file, file flush and atomic rename.
If even failure recording is impossible, the earlier running record remains
non-reusable. A per-attempt file lock prevents concurrent save operations; the
context itself is intended for a single writer. A failed or corrupted record is
not automatically repaired, resumed or overwritten.

For imperative callers, ``context = step.local(...)`` followed by
``step.save(..., context=context)`` outside a ``with`` block publishes immediately.
Such a save cannot observe exceptions or process termination after it returns.
The context-managed form is recommended. Completion attests to the local SDK
scope, not an independently supervised script's exit status.

Verified read-only input copies
-------------------------------

Use ``load_artifacts(producer_directory)`` after the producer's context has exited.
It returns named ``ArtifactSource`` locators only after validating the complete
step record and recomputing all output inventories. Bind selected sources to the
consumer's declared inputs:

.. code-block:: python

    from pathlib import Path
    from mlef import load_artifacts

    sources = load_artifacts(Path(".mlef/producer-attempt"))
    bindings = {"training": sources["raw"]}

Pass ``inputs=bindings`` when opening a consumer with a matching ``training`` input
declaration, then read ``context.input_path("training", "data.csv")`` inside its
context and save its declared outputs.

Input names and required bindings are checked before script work. Semantic types
and any requested formats must match. Failed, incomplete, missing or corrupted
sources cannot be bound; there is no implicit latest artifact or ingestion fallback.

Each input is copied into ``inputs/<name>/`` and verified again. Files in those
copies have their write permissions removed. ``context.input_path(name)`` returns
the input directory; ``input_path(name, relative)`` returns an existing contained
file or directory. ``context.input_reference(name)`` returns its exact producer
reference, suitable for dataset/model/split metadata. An unbound optional input
is absent, not a fabricated empty artifact; requesting it raises ``ValueError``.

Source artifacts are not passed to scripts as writable working paths. The SDK
rechecks both staged copies and original sources before finalization, detecting
changed, missing or added files. It does not silently accept updated producer
manifests. Directory paths are ordinary ``Path`` objects, not a security sandbox:
the owning process can change permissions or create files. Changes to payload bytes
or file inventory invalidate finalization, but preventing malicious owners or races
requires process/container isolation. Serializers and user scripts must be trusted.
There is no hard artifact-size quota or automatic garbage collection.

Records, inspection and example
-------------------------------

An attempt contains ``step.json``, ``outputs/<output>/`` payloads, optional
``inputs/<input>/`` copies and ``active.lock``. ``StepResult`` records schema version,
run ID, the step contract, state, exact input references, output manifests and an
optional failure message. It is distinct from historical ``run.json`` records.
The structural schema is ``schemas/step-result-v1.schema.json``; Python validation
also checks required declarations, compatibility and producer identity.

``read_step(directory)`` returns a verified complete ``StepResult`` or raises.
``load_artifacts(directory)`` returns verified source locators. Both recheck actual
file bytes; merely constructing an ``ArtifactSource`` does not validate or copy
anything. Neither function executes code stored in payloads or imports ML packages.

The runnable ``examples/artifacts/local_handoff.py`` example creates an ingestion
attempt and a transformation attempt, copies a read-only input and prints the
final record:

.. code-block:: text

    uv run --locked python examples/artifacts/local_handoff.py .mlef/sdk-demo

Use a fresh directory for each invocation. The SDK does not silently overwrite
the two attempts when the example runs again.
