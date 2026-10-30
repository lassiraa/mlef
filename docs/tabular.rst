Tabular workflows and agent feedback
====================================

`Documentation home <../README.md>`_

The complete example compares ordinary sklearn Pipelines and CPU XGBoost,
then retrieves a selected model and runs inference. See `setup <setup.rst>`_
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
`examples/sklearn/model.py <../examples/sklearn/model.py>`_.

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
