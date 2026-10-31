Artifact and metric contracts
=============================

These contracts describe evidence. The explicit local SDK in :doc:`script-artifacts`
saves and verifies artifacts using these contracts. Captured :doc:`pipelines`
use the same manifests and verified handoffs. Automatic metric calculation and
S3 storage are not yet provided. Existing single-script execution and historical
run readers are unchanged.

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
predictions and models. Other types use generic metadata; custom type registration
and additional schemas are follow-up work.

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

schemas/artifact-v1.schema.json and schemas/metrics-v1.schema.json expose structural
contracts. Pydantic additionally enforces cross-field invariants such as matrix
shape, required metadata and producer content identity. JSON Schema alone cannot
establish those invariants.
