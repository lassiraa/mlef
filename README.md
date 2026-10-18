# mlef

Run a Python experiment in Docker and keep its logs and output files together.
Early development: the local runner works with the Python standard library;
locked dependencies and Git/environment provenance are planned next.

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

The runner copies tracked files and the explicitly named script into a temporary
Docker build context. Include other untracked source with `--include path`.
It uses `python:3.12-slim` and executes from `/workspace`; your working tree
is never mounted into the experiment. Run storage and credential files are
excluded. Symlink inputs are currently unsupported. Changes made while source
is being copied can affect the copy, so avoid editing inputs during preparation.

Dependencies are not installed yet. Source/environment snapshots, result-query
commands, and background execution are not implemented. These runs do not yet
provide the full reproducibility contract. Experiment code runs with the
container's network access and can write to its artifact mount.

Development:

```text
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
python -m pytest -v
```

Set `MLEF_TEST_DOCKER=1` to include real Docker integration checks. CI runs unit
checks on Linux, macOS, and Windows and container checks on Linux.

Licensed under Apache-2.0; see [LICENSE](LICENSE).
