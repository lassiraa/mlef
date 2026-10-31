Artifact and metric contracts
=============================

These contracts describe evidence. The explicit local SDK in :doc:`script-artifacts`
saves and verifies artifacts using these contracts. Captured :doc:`pipelines`
use the same manifests and verified handoffs. Local/S3 publication is documented
in :doc:`storage`; arbitrary artifact formats do not imply automatic metric
calculation. Existing single-script execution and historical readers are unchanged.

Artifact manifests
------------------

Import contracts from mlef.artifacts. ArtifactManifest separates semantic type
from storage format: datasets and predictions can both use Parquet.
Each file has a normalized relative POSIX path, size and SHA-256 digest.
Inventories reject duplicate paths and file/directory conflicts.

content_sha256() hashes version, semantic type, format, metadata and sorted file
inventory. It excludes the artifact name and producer reference. Moving between
stores or renaming the output does not change identity. Upstream references in
semantic metadata participate in identity. Callers must verify actual file bytes;
the digest only attests to the declared inventory.

For a single file, construct its inventory from the bytes you intend to preserve:

.. code-block:: python

    import hashlib
    from pathlib import Path

    from mlef.artifacts import ArtifactFile, ArtifactManifest, ArtifactType

    path = Path("report.txt")
    payload = path.read_bytes()
    manifest = ArtifactManifest(
        name="report",
        type=ArtifactType.TEXT,
        format="text/plain",
        files=[ArtifactFile(
            path=path.name,
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )],
    )
    identity = manifest.content_sha256()
    serialized_manifest = manifest.model_dump_json()

Constructors use strict Python types, including ArtifactType enum members.
JSON deserialization accepts the enum's string values. Treat validated contracts
as snapshots: mutating nested lists or dictionaries bypasses assignment validation.
Revalidate serialized data before accepting externally modified evidence.

Dataset metadata records all columns, dtypes, roles, row count and row identity.
Statistics are marked omitted, exact, sampled or unavailable. Splits reference a
dataset and an assignments file in the inventory. Predictions reference the
dataset/model/split, identity/value columns, class ordering and prediction scope.
These declarations do not prove row uniqueness, split disjointness or prediction
coverage: those checks require reading payloads. Model metadata records framework
and serialization; loading models is outside these contracts.

Optional dataset-byte and sorted-row-ID SHA256 identities and bounded profiles
extend dataset metadata. Profiles carry rows examined, method and at most 20
column missing/distinct counts; sampled summaries require a seed and sampled-row
identity. These extensions are omitted from content identity when absent, so
historical dataset producer hashes remain valid. See :doc:`tabular` for an
offline exporter that verifies actual rows, split payloads and prediction links.

The type inventory covers files, directories, tables, datasets, splits,
predictions, models, checkpoints, metrics, parameters, schemas, arrays,
embeddings, indexes, plots, reports, text, images, audio, video, annotations,
explanations and archives. Specialized metadata is required for datasets, splits,
predictions and models. Other types use generic metadata. Custom types use the
explicit versioned identities and codecs described below.

Custom artifact codecs
----------------------

``CustomType(name="example.positive_integer", version=1)`` is a semantic type
identity, usable wherever an ``ArtifactType`` is accepted: ``Input``, ``Output``,
``Artifact`` and ``ArtifactManifest``. Names require a lowercase dotted namespace
(letters, digits and underscores within components); versions are positive
integers, not booleans. The identity is immutable. Built-in types keep their
existing enum/string representation and content hashes.

Custom identities serialize as objects containing ``schema_version``, ``name``
and ``version``. Two custom types match only when their entire identities match;
requested formats are checked separately. Custom artifacts currently use generic
metadata, not one of the specialized built-in metadata kinds. Their type identity,
format and file inventory all contribute to content identity.

Use an explicit, process-local ``ArtifactRegistry`` for trusted serialization and
semantic validation. Each ``ArtifactCodec`` registers one exact custom identity
and format. Duplicate registration fails rather than replacing code. There is no
global registry, plugin discovery, automatic package installation, dynamic import
from manifests or fallback to a different version/format.

.. code-block:: python

    import json
    from pathlib import Path
    from mlef import ArtifactCodec, ArtifactRegistry, CustomType, Output, Step

    identity = CustomType(name="example.positive_integer", version=1)
    media_format = "application/json"

    def serialize_integer(value: object, path: Path) -> None:
        path.write_text(json.dumps(value), encoding="utf-8")

    def validate_integer(path: Path) -> None:
        value = json.loads(path.read_text(encoding="utf-8"))
        if type(value) is not int or value <= 0:
            raise ValueError("Expected a positive integer.")

    codecs = ArtifactRegistry()
    codecs.register(ArtifactCodec(
        type=identity,
        format=media_format,
        serializer=serialize_integer,
        validator=validate_integer,
    ))
    step = Step(name="produce", outputs=[Output(
        name="data",
        type=CustomType(name="example.positive_integer", version=1),
        format="application/json",
    )])
    with step.local(Path(".mlef/positive-attempt")) as context:
        step.save([codecs.artifact(
            name="data",
            artifact_type=identity,
            media_format=media_format,
            value=42,
            path=context.output_path("data", "data.json"),
        )], context=context)

``registry.artifact`` returns an ordinary ``Artifact`` whose trusted serializer
first writes the payload and then calls its validator. Validators return normally
on success and raise on invalid payloads; returning false does not reject data.
Callbacks receive the same file or directory path. The destination must not exist,
and the normal SDK path, inventory, no-overwrite and failure-safe finalization
rules still apply. A serialization/validation exception leaves partial diagnostic
bytes in a failed attempt, never reusable output. Callback functions are not
serialized. Bump the custom type version when its semantic contract or codec
behavior changes incompatibly, and capture the codec source with the pipeline.

Consumers declare the same custom identity and optionally the exact format.
Inside a matching context they can explicitly revalidate a verified input copy:

.. code-block:: python

    codecs.validate_payload(identity, media_format,
                            context.input_path("raw", "data.json"))

``validate_payload`` requires a registered exact identity/format and an existing
regular file or directory without symlinks/reparse points/hard links. It runs
trusted user code; it is not a sandbox and does not independently compare a path
against a producer manifest. Use verified SDK input paths when checking inputs.

Ordinary inspection, transfer, cache selection and explicit reuse never execute
custom codecs. They verify declared types, contracts and bytes, not custom payload
semantics, and work without a registry. Likewise, constructing a plain ``Artifact``
with a custom identity does not run a registered validator. Use
``registry.artifact`` when producing and ``validate_payload`` when consuming if
semantic validation is required. Registration is not an enforcement boundary.
Older MLEF versions cannot read custom identity objects, but existing built-in
artifact records and captured SDKs remain unchanged and readable.

Metric evidence
---------------

Import metric contracts from mlef.metrics:

.. code-block:: python

    from mlef.metrics import Metrics, Scalar, Undefined

    performance = Metrics(
        scope="validation",
        values={
            "accuracy": Scalar(
                definition="fraction_correct",
                value=0.91,
                direction="maximize",
                sample_count=100,
            ),
            "auc": Undefined(
                definition="roc_auc",
                reason="Only one class in this slice.",
            ),
        },
    )
    serialized = performance.model_dump_json()
    restored = Metrics.model_validate_json(serialized)

Metrics is the JSON payload for an artifact of type metrics. Optional references
identify model, dataset, split and predictions. A manifest describes the payload
file and checksum; these models do not save or finalize it.

* Scalar: value with optional method-labelled confidence interval.
* Series: paired numeric coordinates with a named index (epoch, time, etc.).
* Curve: paired coordinates, axis labels and optional point-aligned thresholds.
  Normalize endpoint/sentinel thresholds before constructing this representation.
* Matrix: rectangular values with labelled and oriented axes.
* MetricTable: declared columns with matching flat rows, useful for slices/classes.
* Distribution: samples with optional aligned nonnegative weights.
* Undefined: a reason instead of NaN or an invented numeric score.

All representations carry definition/version, unit, aggregation, sample count,
parameters and dimensions. Direction can maximize, minimize, target a specified
value, or express no ordering. Numeric values must be finite. This is a
framework-independent representation, not a catalog of calculators.
Series/curve order is preserved; monotonicity is not universally required.

Matching names does not establish comparability. Consumers must check definitions,
versions, units, parameters, aggregation and evaluation references. Curves have no
implicit scalar ranking. External storage for large metric arrays and streaming
serializers remain follow-up work.

``Metrics.semantic_differences(other)`` checks declared scope, names, common
scoring semantics and labelled axes without ranking measured values. It does
not establish dataset/split compatibility or execution success; the tabular
comparator separately validates those prerequisites.

Schemas
-------

schemas/artifact-v1.schema.json, schemas/custom-type-v1.schema.json and
schemas/metrics-v1.schema.json expose structural
contracts. Pydantic additionally enforces cross-field invariants such as matrix
shape, required metadata and producer content identity. JSON Schema alone cannot
establish those invariants.
