MLEF documentation
==================

Run Python experiments in Docker, preserve their evidence, and compare sklearn
and XGBoost models on the same saved folds. Use the CLI yourself or the Python API
from an agent. No account, database or LLM provider is required.

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
  partial outputs after failure. See :doc:`runs`.
* Compare baseline/candidate sklearn and XGBoost models on saved folds, refit the
  selected model and retrieve predictions. See :doc:`tabular`.
* Give agents bounded profiles, comparison results and error slices through
  :doc:`api`. See :doc:`benchmarks` for independent final scoring.

Experiment execution requires Docker; offline inspection runs natively without
ML libraries. Validation selection alone does not establish held-out performance.

Guides and reference
--------------------

.. toctree::
   :maxdepth: 2

   setup
   tabular
   api
   artifacts
   pipelines
   runs
   benchmarks
   development

`Source repository <https://github.com/lassiraa/mlef>`_ ·
`Apache-2.0 license <https://github.com/lassiraa/mlef/blob/main/LICENSE>`_
