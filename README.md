# mlef

Run a Python experiment in Docker and keep its logs and output files together.
Early development: the local runner saves source and Git provenance;
locked dependencies and environment provenance are planned next.

Requires Python 3.11+, Git, and Docker running Linux containers. Host targets are
Linux, macOS, and Windows; platform validation is still in progress.

```text
python -m pip install -e .
mlef init
mlef run examples/train.py -- --accuracy 0.95
mlef run examples/train.py -- --fail
```

Run these commands from the Git project root. `init` creates `.mlef/runs/` and
adds `/.mlef/` to `.gitignore`. Each run prints its ID and result directory.
Inspect that directory's `run.json`, `stdout.log`, `stderr.log`, `runner.log`,
and `artifacts/`. Failed runs keep their logs and any outputs already written.
The example failure returns CLI exit code 1 and records script exit code 7.

Scripts write durable results to the directory in `MLEF_OUTPUT_DIR`. To collect
another directory written under the project working directory, use
`mlef run train.py --output results`. The script argument separator is `--`.
Declared outputs are required: missing outputs or destination conflicts fail
the run without overwriting existing artifacts.
Use Ctrl+C to cancel; mlef gives the container five seconds to stop before
forcing termination and collecting available outputs.

The runner archives tracked files and the explicitly named script into
`source.tar.gz`, then builds from that saved archive. Include other untracked
source with `--include path`. Each run saves the archive SHA-256, included
untracked paths, Git commit/branch/dirty state, a binary-capable `changes.patch`
relative to HEAD (the empty tree for an unborn repository), and the generated
`Dockerfile`. Detached HEAD records a null branch. The archive defines the actual
source inputs; the patch contains tracked changes only.
These are additive fields on the current version-1 run metadata; the complete
version-2 inspection/result contract is still planned.
It uses `python:3.12-slim` and executes from `/workspace`; your working tree
is never mounted into the experiment. Run storage and credential files are
excluded. Symlink inputs are currently unsupported. Changes made while source
is being copied can affect the copy, so avoid editing inputs during preparation.

Dependencies are not installed yet. Environment snapshots, result-query
commands, and background execution are not implemented. These runs do not yet
provide the full reproducibility contract. Experiment code runs with the
container's network access and can write to its artifact mount.

Development:

```text
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
ty check
python -m pytest -v
```

Set `MLEF_TEST_DOCKER=1` to include real Docker integration checks. CI runs unit
checks on Linux, macOS, and Windows and container checks on Linux.

Ruff enforces formatting, imports, annotations (including tests), bug-risk checks,
pytest conventions, and simplification rules. ty checks source, tests, and examples
against Python 3.11 across platforms, with all diagnostics treated as errors.
Both checks run in CI. Keep any necessary suppressions specific and explain them.

Licensed under Apache-2.0; see [LICENSE](LICENSE).
