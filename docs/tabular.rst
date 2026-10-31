Tabular workflows and agent feedback
====================================

:doc:`Documentation home <index>`

The complete example compares ordinary sklearn Pipelines and CPU XGBoost,
then retrieves a selected model and runs inference. See :doc:`setup <setup>`
for the copy-and-run commands (use a fresh directory for each demo).

A model is an ordinary Python factory
-------------------------------------

The evaluator imports ``module:factory`` and calls it with a seed. For example,
the numeric-only case can use this ``model.py``:

.. code-block:: python

    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    def make_estimator(seed):
        return make_pipeline(
            SimpleImputer(add_indicator=True),
            LogisticRegression(random_state=seed, max_iter=1000),
        )

Declare sklearn in the experiment's ``pyproject.toml`` and lock it with uv.
For mixed numeric/categorical features, reuse the ColumnTransformer factories in
`examples/sklearn/model.py <https://github.com/lassiraa/mlef/blob/main/examples/sklearn/model.py>`_.

Save folds, then evaluate candidates
------------------------------------

Start from a fresh copy of ``examples/sklearn`` with ``git init``, as in
setup. Export the evaluator and initialize storage before preparing folds:

.. code-block:: text

    uv run --project ../mlef --locked python -c "from mlef import tabular; tabular.export_helper('evaluate.py')"
    mlef init

Then run:

.. code-block:: text

    mlef run evaluate.py --include data.csv -- prepare --data data.csv --numeric radius texture smoothness --categorical radius_band

Retrieve ``protocol.json`` using the printed preparation run ID, then evaluate
both factories against those saved assignments:

.. code-block:: text

    mlef get PREPARATION_RUN protocol.json
    mlef run evaluate.py --include data.csv --include protocol.json --include model.py -- evaluate --data data.csv --factory model:make_dummy
    mlef run evaluate.py --include data.csv --include protocol.json --include model.py -- evaluate --data data.csv --factory model:make_estimator
    mlef tabular compare BASELINE CANDIDATE --json
    mlef tabular diagnose BASELINE CANDIDATE --json

Replace the uppercase identifiers with complete run UUIDs. Add ``--refit`` to the
evaluator's ``evaluate`` arguments to save ``model.joblib`` and
``model-recipe.json``. The full demo also records experiment hypotheses and parent
runs; use its Python source as the orchestration example.

Evaluation evidence
-------------------

Preparation saves actual stratified folds once in ``protocol.json``; evaluation
refuses changed dataset bytes, IDs, invalid assignments or missing class coverage.
Each fold clones/fits the entire Pipeline on training rows only. The versioned
``evaluation.json`` contains the full protocol, per-fold scores, timing and failures;
``predictions.csv`` preserves fold/row alignment and has a recorded SHA-256. A failed
fold keeps partial evidence, exits unsuccessfully, and never produces a successful
aggregate ``metrics.json``. Provided group/time protocols are validated too: group
partitions must be disjoint, and training timestamps must strictly precede scoring
timestamps (timezone-aware ISO-8601). Temporal protocols may omit early scoring rows.

Dataset attribution
-------------------

The classification fixture adapts three features from the
`Wisconsin Diagnostic dataset <https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic>`_
by Wolberg, Mangasarian, Street and Street (1993), DOI 10.24432/C5DW2B,
under `CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/>`_. Row IDs are
renumbered, every seventeenth radius is blank, and ``radius_band`` uses a fixed
radius threshold of 14. These are illustrative alterations; positive class ``1``
means benign in sklearn's source labels. This example supplies no clinical claim.
Editable experiment code remains trusted: protocol hashes and pipeline evaluation
do not establish tamper resistance or absence of all forms of leakage.

Offline comparison
------------------

Compare saved fold evidence without Docker or ML libraries using
``mlef tabular compare BASELINE CANDIDATE --json`` or
``api.compare_tabular(baseline, candidate)``. This checks actual assignments,
prediction byte hashes, row coverage and labels, and recomputes ROC-AUC from saved
predictions. Both execution and every required fold must succeed. Different
protocols are incompatible; absent/partial evidence cannot support an improvement
claim. Fold summaries default to 20 entries (``--limit``/``limit``, at most 100) and
report omitted counts; full evidence remains in referenced artifacts. Each input
evidence file is limited to 32 MiB for offline inspection. Generic ``compare``
retains its original declaration-based behavior. These are consistency checks of
saved evidence, not independent execution attestation or statistical significance.

Early stopping, refit and inference
-----------------------------------

The sklearn demo also evaluates CPU-only XGBoost candidates, including a missing-
value-indicator/preprocessing hypothesis. XGBoost's early-stopping validation rows
come exclusively from each outer training partition; grouped/time protocols retain
those constraints for the inner partition. Preprocessing is fitted on inner training
rows and explicitly transforms the validation set. Evaluation v2 records the actual
inner assignments and best iteration; v1 evidence remains readable offline.

After comparing all candidates, the demo refits the best validation candidate on
the complete exploration dataset. Boosted models use the median CV best tree count
with early stopping disabled for this refit. ``model.joblib`` and ``model-recipe.json``
preserve preprocessing, feature order, label semantics, parameters and versions.
``evaluate.py predict`` explicitly loads a trusted model and writes positive-class
probabilities; offline inspection never deserializes models. The demo retrieves
the selected model and runs this prediction step through Docker too. No final-test
labels participate; validation selection alone is not evidence of generalization.
XGBoost host tests run on Linux/Windows; the base macOS CLI/package matrix remains
independent of XGBoost, and macOS container execution remains preview.

Regression
----------

Run ``python demo.py --regression`` from a fresh copy of the sklearn example for
dummy/linear/XGBoost regression, candidate selection, refitting and inference.
``regression.csv`` is an Apache-2.0 synthetic fixture reproducible with
``make_regression.py`` (seed 1729), not a real-world performance benchmark. New
protocol v2 supports binary classification and regression; binary protocol v1
remains readable. Regression uses KFold (or supplied group/time partitions),
finite original-unit targets/predictions, and lower-is-better ``rmse_mean``.
That score is the unweighted mean of fold RMSEs, not pooled OOF RMSE. Regressor
wrappers may inverse-transform targets through ordinary sklearn prediction;
the evaluator scores returned predictions in the original target units.
Inference writes ``prediction`` for regression and ``positive_probability`` for
classification. Early-stopping XGBoost must be the final estimator directly or
inside a Pipeline; arbitrary nested early-stopping wrappers are unsupported.

Shared script artifacts
------------------------

Export a successful evaluation into the same typed artifacts that ordinary
scripts consume with ``Step.local`` / ``load_artifacts``:

.. code-block:: text

    mlef tabular export RUN --data public.csv --output .mlef/shared/RUN --profile

.. code-block:: python

    from pathlib import Path
    from mlef import api, tabular, load_artifacts

    bundle = api.export_tabular_artifacts(
        run.id, "public.csv", ".mlef/shared/" + run.id, profile=True,
    )
    verified = tabular.read_artifacts(Path(".mlef/shared") / run.id)
    inputs = load_artifacts(Path(".mlef/shared") / run.id / "predictions")

The explicit CSV must match the captured protocol's dataset hash and ordered,
unique row IDs. Prediction labels must agree with that public CSV, not only with
the reported score. Export reads only that CSV and the named run's saved evidence;
it never discovers datasets, reads private held-out labels, imports ML libraries
or deserializes fitted models. Successful execution, complete folds and completed
resource cleanup are required. Destinations must not exist. Historical version-1
evaluations are adapted without modifying their source run.

A bundle contains ``evaluation.json``, a final ``bundle.json`` publication marker,
and five completed SDK step directories: ``dataset``, ``split``, ``model``,
``predictions`` and ``metrics``. Each has ``step.json`` and managed input/output
directories. The artifact types use ``mlef.tabular-NAME-v1`` formats. The dataset
contains the original ``data.csv``; split assignments explicitly persist fold,
row ID and train/score roles. Metadata records original data-byte and sorted-row-ID
hashes, dataset/model/split links, OOF prediction scope and binary class order
``[negative_label, positive_label]``. The one probability column refers to the
second class; regression has no class order and retains original target units.
The row identity hash uses the UTF-8 JSON array of sorted row IDs, ASCII-escaped
and without whitespace; the dataset hash uses the unchanged CSV bytes.
All producer references retain the source run ID and ``tabular_NAME`` step names.

The ``model`` output is a **CV recipe descriptor**, not a fitted estimator. Its
``mlef.tabular-cv-recipe-v1`` serialization records parameters, actual inner-fit
assignments when available, protocol identity and source-evaluation hash. OOF
predictions link to this recipe, never falsely to the separately refitted model.
Existing ``model.joblib`` / ``model-recipe.json`` inference workflows are unchanged;
they are not loaded or bundled by the offline exporter.

Profiles are omitted by default. ``--profile`` adds exact missing/distinct counts
for at most 20 columns, truncated names (200 characters) with full-name hashes,
rows examined and omitted-column counts, never raw values or example rows. Shared
profile contracts also describe sampled summaries with required sampling seed and
sampled-row identity; this exporter does not sample automatically. Dataset metadata
does not replace byte/row verification. Each parsed/exported payload is limited to
32 MiB; large-data streaming and arbitrary custom formats remain separate work.

``tabular.read_artifacts`` rechecks bytes, contracts, producer wiring, public row
truth, full split assignments, class ordering, prediction coverage, recipe links
and structured metrics. The bundle remains verifiable after moving it or removing
the source run/CSV. Export never overwrites or repairs prior evidence; failed
exports may leave diagnostic directories without a publication marker. Treat
bundle verification, not the existence of an individual directory, as publication.
Both binary and regression demo modes now export these bundles under
``.mlef/tabular-artifacts/RUN`` for every successful candidate.

New evaluations include a structured ``metric_evidence`` payload in addition to
legacy numeric ``metrics.json``. Comparison checks scope, definition/version,
units, aggregation, sample counts, dimensions and parameters after matching the
actual dataset/folds; equal metric names alone are insufficient. One historical
run without these declarations and one new run produce ``unknown`` rather than
an inferred winner. Two historical runs retain the prior built-in comparison.
Export can reconstruct the built-in ROC AUC/RMSE evidence from verified historical
folds, but does not invent custom metric definitions or independent attestations.

Profiles and agent feedback
---------------------------

For agent feedback, use ``mlef tabular profile RUN --json`` /
``api.tabular_profile(run)`` and ``mlef tabular diagnose BASELINE CANDIDATE --json`` /
``api.tabular_diagnostics(baseline, candidate)``. Evaluation saves a training/search
dataset profile with missingness, cardinality, constants, duplicate-feature rows,
target distribution/range and declared group/time structure. Profiles return at
most 20 feature names (200 characters each, with full-name hashes) and omitted
counts; no raw categorical values or row samples are included.

Diagnostics compare aligned OOF absolute probability errors for classification
and absolute target errors for regression. Fixed slices cover the whole dataset,
target class/probability bands, or baseline-prediction quartiles. The default
minimum count is 20 (``--minimum-count``, at least 2); output defaults to 20 slices
and reports omissions from filtering/truncation. These overlapping exploratory
slices help formulate hypotheses; they do not establish leakage or causal gains.
Both operations are offline and need no ML libraries.

A supervised agent workflow is: inspect the task/profile and existing runs;
establish baselines; state a hypothesis and parent run in ``Experiment``; change a
small set of ordinary Python features/preprocessing/parameters; evaluate on saved
folds; compare scores, errors and runtime; retain or reject the candidate; then
freeze a finalist for independent testing. The demo records hypothesis/parent
metadata, including the selected refit's actual source candidate. Set per-run
deadlines/log/resource limits and an explicit exploration trial/time budget.
MLEF currently enforces per-run controls, not a global exploration budget or hard
artifact quota. Keep final-test labels outside the agent workspace and run inputs.
