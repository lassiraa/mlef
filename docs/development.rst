Development and release checks
==============================

`Documentation home <../README.md>`_

.. code-block:: text

    uv sync --locked --extra dev
    uv run --locked --extra dev ruff check .
    uv run --locked --extra dev ruff format --check .
    uv run --locked --extra dev ty check
    uv run --locked --extra dev pytest -v


Set ``MLEF_TEST_DOCKER=1`` to include real Docker integration checks. CI runs unit
checks on Linux, macOS, and Windows, container checks on Linux, and clean wheel and
sdist installs on all three hosts. The container suite includes failure,
cancellation, declared outputs, concurrent-run isolation, locked installation,
saved-recipe rebuild, malformed optional metrics, required-result storage failure,
and the complete tabular run/inspect/compare/retrieve loop.

Ruff enforces formatting, imports, annotations (including tests), bug-risk checks,
pytest conventions, and simplification rules. ty checks source, tests, and examples
against Python 3.11 across platforms, with all diagnostics treated as errors.
Both checks run in CI. Keep any necessary suppressions specific and explain them.

Versioned JSON Schemas are in `schemas <../schemas>`_. The core contracts are
`run v3 <../schemas/run-v3.schema.json>`_, `experiment v1 <../schemas/experiment-v1.schema.json>`_,
`result v1 <../schemas/result-v1.schema.json>`_, `environment v1 <../schemas/environment-v1.schema.json>`_,
and `comparison v1 <../schemas/comparison-v1.schema.json>`_. Regenerate them with
``uv run --locked --extra dev python scripts/export_schemas.py``; CI checks that the
committed copies match the Pydantic contracts. ``run.json``, ``environment.json`` and
``result.json`` are saved atomically. Run directories and artifacts are authoritative;
there is no database, hosted backend, daemon or mandatory LLM service.

Contributing and security
-------------------------

Contributions should include a minimal reproduction or concrete behavior change,
focused pytest coverage for meaningful logic, and the uv checks above. Use
Conventional Commit titles and keep PR descriptions limited to committed changes
and their validation under Summary, Validation and optional Compatibility sections.
For security reports, share sensitive details privately with the maintainer before
posting a sanitized public reproduction. Do not upload credentials, private datasets
or complete sensitive run artifacts to issues. Experiment code has network access
by default and a writable artifact mount; use only trusted scripts. Deadlines,
runtime resource controls and log caps do not enforce a hard artifact quota.
Unattended-agent execution remains outside this release.

Release artifacts
-----------------

Build candidate artifacts using ``uv build --out-dir .mlef/dist``, then verify them
with ``uv run --locked --extra dev python scripts/check_dist.py .mlef/dist``.
Both formats are installed and exercised outside the checkout. Review the complete
PR stack and platform limitations before tagging or publishing. Publication is a
separate release action; no CI push or PR event publishes this package.

User documentation is plain reStructuredText in ``docs/`` with README navigation.
It renders directly on GitHub and ships in the source distribution; no docs server
or separate build tool is required. Keep runnable examples in ``examples/`` and
link guides to those sources rather than copying complete programs.
