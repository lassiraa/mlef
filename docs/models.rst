Model discovery and versions
================================

Model queries use typed, completed pipeline artifacts. They work offline without
Docker or ML packages and never deserialize model files or import experiment code.

.. code-block:: text

    mlef models list --run RUN --json
    mlef models register RUN/train/model --name churn-classifier
    mlef models list --json
    mlef models show churn-classifier@VERSION --json
    mlef models evaluations churn-classifier@VERSION --json
    mlef models get churn-classifier@VERSION --output saved-model

Replace RUN with a complete run UUID and VERSION with the complete version hash
printed by registration. ``list --run`` discovers model outputs, including
unregistered ones; plain ``list`` lists registered versions. Responses contain a
schema version, ``artifacts``, ``versions`` and ``diagnostics``. Corrupt catalog
entries are skipped with diagnostics rather than treated as valid versions.

Declare models explicitly
-------------------------

A producing step must declare an output with ``ArtifactType.MODEL`` and publish
it with ``ModelMetadata``. Consumers must declare the same semantic type. For
example, a training script can use:

.. code-block:: python

    from mlef import Artifact, ArtifactType, Output, Step
    from mlef.artifacts import ModelMetadata

    step = Step(
        name="train",
        outputs=[Output(name="model", type=ArtifactType.MODEL, format="joblib")],
    )
    # In the script's managed context, after writing model_path:
    artifact = Artifact(
        name="model", type=ArtifactType.MODEL, format="joblib", path=model_path,
        metadata=ModelMetadata(framework="sklearn", serialization="joblib"),
    )

The existing SDK save rules apply. The initializer's toy model is deliberately a
generic JSON file and does not appear as a typed model. Filenames, model
references in metric payloads and legacy script files do not automatically
establish a model contract. Standalone script outputs and local SDK attempts
are not cataloged by this initial implementation.

Registration and evidence
-------------------------

Names use the existing portable step naming rules: start with an ASCII letter,
then letters, digits, underscores or hyphens. A version is a SHA-256 identity of
the exact original producer reference (run, step, output and content checksum).
Repeated registration of that producer under the same name returns the existing
version. Equal bytes from different producers retain distinct versions.

Registration requires a terminal run, no active owner and no pending cleanup.
It verifies captured source/SDK evidence, completed inventories, contracts and
input wiring. Completed model outputs from failed or cancelled runs may be
registered; this does not establish a successful evaluation of the model.

The version records the complete model manifest and a snapshot of the containing
run's metadata, including available experiment parameters, Git and environment
input hashes. Model metadata contains framework/serialization and an optional
training-dataset reference. Unknown metadata remains unknown. Reused outputs
retain their original producer reference; ``captured_run`` describes the run
containing the verified copy and must not be mistaken for its original training
execution. The first registration retains its chosen artifact location.

Catalog files live in ``.mlef/models/NAME/VERSION.json``. Updates use a catalog
ownership lock and atomic publication; registration never overwrites a version.
Listing/showing versions reads saved descriptors. Retrieval reverifies source
evidence and artifact identity, copies the complete output directory, and checks
the retrieved inventory. Existing destinations are refused.

Versions reference existing run storage; they do not duplicate model bytes.
Deleting or moving the containing run can make retrieval unavailable while the
descriptor remains inspectable. Existing run publish/fetch does not transfer the
model catalog; catalog portability, aliases and reference-protected deletion
remain planned platform features. :doc:`evaluations` records batch history by
exact model selector. No command in
this initial catalog deletes models or runs.

Python API
----------

.. code-block:: python

    from mlef import api

    discovered = api.list_models(run_id=run_id)
    version = api.register_model(run_id, "train", "model", name="churn-classifier")
    versions = api.list_models()
    selector = f"{version.name}@{version.version}"
    details = api.model_version(selector)
    destination = api.get_model(selector, output="saved-model")

All functions accept the usual optional project ``root`` and raise categorized
``MlefError`` failures. Schemas are exported as ``model-version-v1`` and
``model-listing-v1``.
