MLEF documentation
==================

Run Python experiments and pipelines in Docker, track runs and model versions,
and compare saved models with a frozen evaluator. Use the CLI yourself or the
Python API from an agent. Source snapshots, environments, logs, metrics and
artifacts stay in your Git project's ``.mlef/`` directory. No account, database
or LLM provider is required.

Quick start
-----------

Install Python 3.11–3.14, Git, uv and Docker with a running Linux-container engine:

.. code-block:: text

    git clone https://github.com/lassiraa/mlef.git
    cd mlef
    uv tool install --python 3.12 .
    mlef init
    mlef run examples/train.py -- --accuracy 0.95
    mlef list

Replace ``RUN`` with the complete run UUID printed above:

.. code-block:: text

    mlef status RUN --json
    mlef get RUN metrics.json --output saved-metrics.json

This script writes illustrative metrics and a dummy model. Follow :doc:`setup`
to run the real tabular example in its own project.

What can you do with it?
------------------------

* Keep source, locked environments, logs, metrics and outputs together, including
  partial outputs after failure. Filter run history, group experiments, edit
  labels/tags/notes and archive runs without deleting them. See :doc:`runs`.
* Build typed pipelines with training, prediction and evaluation roles, verified
  cross-run inputs, explicit reuse/resume and opt-in caching. See :doc:`pipelines`
  and :doc:`script-artifacts`.
* Discover model artifacts, register immutable versions, retrieve verified model
  bytes and manage aliases with change history. See :doc:`models`.
* Evaluate saved models sequentially with frozen source and data bindings, inspect
  per-model outcomes, and compare compatible scalar results in a leaderboard.
  See :doc:`evaluations`.
* Inspect root and pipeline metrics offline, preserving their scope and artifact
  references. See :doc:`runs` and :doc:`artifacts`.
* Compare baseline/candidate sklearn and XGBoost models on saved folds, refit the
  selected model and retrieve predictions. See :doc:`tabular`.
* Publish and fetch self-contained terminal runs through local directories or
  S3 with integrity checks. See :doc:`storage`.
* Give agents bounded profiles, comparison results and error slices through
  :doc:`api`. See :doc:`benchmarks` for independent final scoring.

Experiment execution requires Docker; offline inspection runs natively without
ML libraries. Validation selection alone does not establish held-out performance.

Guides and reference
--------------------

.. toctree::
   :maxdepth: 2

   setup
   runs
   models
   evaluations
   pipelines
   script-artifacts
   artifacts
   tabular
   api
   storage
   benchmarks
   development

`Source repository <https://github.com/lassiraa/mlef>`_ ·
`Apache-2.0 license <https://github.com/lassiraa/mlef/blob/main/LICENSE>`_
