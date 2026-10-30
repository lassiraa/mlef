Evaluating agent workflows
==========================

:doc:`Documentation home <index>`

This is an advanced coordinator guide. Start with the :doc:`tabular workflow <tabular>`
for model development. Commands below run from the MLEF checkout unless stated
otherwise; first install development dependencies as in :doc:`development <development>`.

The matched tabular benchmark starts with
``uv run --locked --extra dev python benchmarks/fixture.py prepare-tabular TRIAL --task classification-mixed``
(also ``classification-missing``, ``regression-mixed``, ``regression-missing``). It writes
common code, a pinned environment, saved exploration folds and unlabeled submission
features into ``TRIAL/agent``; final labels stay in ``TRIAL/private``. The classification
variants use the Wisconsin data attributed in the :doc:`tabular guide <tabular>`; regression variants use the
reproducible synthetic fixture. ``task.json`` records source/license/modifications
and dataset/protocol fingerprints. These variants are harness fixtures, not four
independent real-world datasets or evidence of agent uplift.

For a conventional reference dry run, enter the agent workspace and execute
``uv run --locked python reference.py``. It uses four randomized sklearn/XGBoost
search candidates, the same actual saved folds, one training thread, and no final
labels. Its output contains a selected model, validation score, timing and frozen
submission predictions. Both control and MLEF conditions receive the same starting
files; initialize Git and use the installed MLEF tools only in the MLEF condition.
For custom Python transformers, include their source modules (such as ``model.py``)
alongside the model/recipe when running inference; joblib does not bundle that code.
Only explicitly load trusted model files.

The coordinator runs
``python benchmarks/fixture.py score-tabular TRIAL/private/labels.json PREDICTIONS.csv --candidate MODEL.joblib``
outside agent access. This checks exact IDs, finite probabilities/predictions and
metric direction, then creates ``labels.score.json`` under a kernel-held lock. Exact
retries return the same score; changed predictions, model bytes or private labels
are refused after scoring. Failed submissions do not consume the final score.
The ledger records all three hashes and the task/protocol identity.
Scoring preserves exact prediction/model snapshots beside the private ledger;
prediction validation and hashing use the same captured bytes. Stop exploration
before submitting. An interrupted freeze fails closed and retains its private
files for coordinator inspection rather than overwriting them on a retry.

Record a version-2 trial using ``fixture.py record INPUT.json OUTPUT.json --score-ledger LEDGER``.
The schema is `tabular trial v2 <https://github.com/lassiraa/mlef/blob/main/schemas/tabular-trial-v2.schema.json>`_. Completed
trials require a matching frozen-score ledger and stay within their declared
budget. Failed/cancelled trials retain reasons, missing held-out scores and actual
elapsed time, including overruns. Record agent/model/settings, seed, repetition,
valid experiment/comparison counts, first valid validation-gain time, output bytes,
interventions and the selected candidate's hashes. ``fixture.py pair CONTROL MLEF``
requires matching task/protocol/model/settings/budget and reports direction-aware
held-out improvement alongside costs and failures. Legacy accuracy-only trial
recording/pairing remains available; regression is never stored as accuracy.
``fixture.py report RECORDS_DIRECTORY --limit 20`` summarizes repeated v2 records
with state counts, matched pairs, failures, unmatched/duplicate groups and invalid
records. It reports omissions explicitly and never pools AUC/RMSE magnitudes or
quietly selects the best duplicate observation. Keep trial records in their own
directory, separate from score ledgers and task manifests.

Use separate matched workspaces/repetitions and report every trial. Configure actual
OS/container/tool access so exploration cannot read coordinator files; the fixture
creates directories, not a sandbox. Public dataset labels can also be looked up,
so real claims need controlled access and suitably private final evaluation data.
Keep final scoring out of iterative model selection. Benchmark accounting validates
recorded budgets but does not enforce a global process deadline. No LLM provider
is called by this harness, and reference/unit dry runs are not measured LLM trials.

Legacy accuracy fixture
-----------------------

The matched-trial fixture starts with
``uv run --locked --extra dev python benchmarks/fixture.py prepare <trial-directory>``.
Give an agent access only to the generated ``agent/`` workspace, initialize Git there,
and keep ``private/`` outside its accessible roots and Docker inputs. Final-test
labels are held in the coordinator's private directory. The fixture does not
enforce operating-system access controls; the evaluator must configure them.
Use identical data fingerprints, agent/model versions, seeds, budgets and repetition
indices for control (ordinary Python) and treatment (MLEF) trials. Train on
``train.csv``, make decisions using ``validation.csv``, and submit test predictions
only after exploration ends. Score them outside the agent workspace with
``fixture.py score <private-labels> <predictions>``.

Record a trial JSON with ``fixture.py record <input> <output>``: ``task_fingerprint``,
``condition`` (``control`` or ``mlef``), ``agent``, ``model``, ``budget_seconds``, ``seed``,
``repetition``, ``elapsed_seconds``, ``tool_output_bytes``, ``intervention_count``,
``valid_comparisons``, ``time_to_first_valid_gain_seconds`` (null if absent), and
``heldout_accuracy``. These are evaluator-recorded measurements, not automatic agent
telemetry. ``fixture.py pair <control> <mlef>`` rejects mismatched conditions and
reports paired differences. Repeat matched trials to assess repeatability; this
release includes the fixture, not a measured claim that MLEF improves agents.
