Installation and project setup
==============================

:doc:`Documentation home <index>`

Requirements
------------

* Host: Git, Python 3.11â€“3.14, and `uv <https://docs.astral.sh/uv/getting-started/installation/>`_.
  Development and the container installer use uv 0.13.6.
* Execution: a running `Docker Linux-container engine <https://docs.docker.com/engine/install/>`_.
  Check connectivity with ``docker info``.
* CLI/package checks cover Linux, macOS and Windows. Container execution is
  validated on Linux and Windows Docker Desktop. macOS execution remains preview.

Host and experiment Python versions are independent. Experiments currently use
Python 3.12 in a digest-pinned ``python:3.12-slim`` image. Docker is required for
execution; saved-run inspection works natively and offline.

Install the CLI
---------------

Install the current release candidate from a checkout:

.. code-block:: text

    git clone https://github.com/lassiraa/mlef.git
    cd mlef
    uv tool install --python 3.12 .
    mlef --version

If your shell cannot find ``mlef``, run ``uv tool update-shell``, then open a new
terminal. After updating the checkout, reinstall with
``uv tool install --force --python 3.12 .``.

Initialize your experiment
--------------------------

From your own project's exact Git root, using your existing script and data:

.. code-block:: text

    git init
    mlef init
    mlef run train.py --include data.csv -- --seed 7

``init`` creates ``.mlef/runs/`` and adds ``/.mlef/`` to ``.gitignore``. If neither
``mlef.toml`` nor ``steps`` exists, it also creates a :doc:`four-step script
example <pipelines>`, without overwriting existing project files. Use
``mlef init --storage-only`` to omit the example. The runner
captures tracked files, the script, root dependency files and explicit untracked
``--include`` paths. Scripts write durable outputs to ``MLEF_OUTPUT_DIR``;
``--output results`` also collects a required directory under the container's
project working directory. Use the examples below if you do not have a script yet.

Host API and experiment dependencies
------------------------------------

``uv tool install`` installs an isolated CLI. It does not make ``import mlef``
available to an unrelated Python environment. For Python orchestration, run
``uv sync --locked`` in the MLEF checkout. From an experiment project, use:

.. code-block:: text

    uv run --project /path/to/mlef --locked python demo.py

Replace ``/path/to/mlef`` with your checkout path. This selects the host API
without changing the experiment working directory. No host ML libraries are needed.

The experiment has its own ``pyproject.toml`` and ``uv.lock``. Declare runtime
dependencies with ``uv add`` and commit both files. Inside Docker, MLEF installs
compatible registry wheels from that lock, without extras, dependency groups or
installation of the project itself. Path/Git/workspace dependencies, source builds,
separate ``uv.toml`` configuration and other dependency managers are unsupported.
Do not add the MLEF checkout as a path dependency to the experiment. Export the
portable evaluator when needed; it becomes an ordinary captured input. See
:ref:`environment details <captured-environment>`.

Minimal locked-dependency example
---------------------------------

From the MLEF checkout, copy to a new directory:

.. code-block:: text

    python -c "import shutil; shutil.copytree('examples/locked', '../mlef-locked-demo', ignore=shutil.ignore_patterns('.venv', '__pycache__'))"
    cd ../mlef-locked-demo
    git init
    mlef init
    mlef run train.py

This installs the example's pinned ``packaging`` dependency and saves its results.

Complete tabular demo
---------------------

Start again in the MLEF checkout. Each invocation needs a fresh copied project:

.. code-block:: text

    uv sync --locked
    python -c "import shutil; shutil.copytree('examples/sklearn', '../mlef-tabular-demo', ignore=shutil.ignore_patterns('.venv', '__pycache__'))"
    cd ../mlef-tabular-demo
    git init
    uv run --project ../mlef --locked python demo.py

For regression, copy into a different directory and run the last command with
``demo.py --regression``. Both demos initialize storage and export ``evaluate.py``.
They perform multiple Docker runs, print IDs/comparisons and save ``protocol.json``,
``model.joblib``, ``model-recipe.json`` and ``inference.csv``. Retrieved files are
intentionally never overwritten. See :doc:`tabular workflows <tabular>` for data
attribution and interpretation.

.. _setup-troubleshooting:

If setup fails
--------------

* Project-root error: use the directory containing ``.git``, then ``mlef init``.
* Docker unavailable: start a Linux-container engine and check ``docker info``.
* Stale lock: deliberately run ``uv lock`` in the experiment, inspect and commit
  changes. Ordinary checks use ``--locked``.
* Host demo import error: use the checkout environment as shown above.
* Existing destination: choose a new output path or fresh copied demo project.

Inspect ``mlef status RUN --json`` and saved logs for execution failures. More
failure modes and exit codes are in the :ref:`CLI reference <run-exit-codes>`.
