# mlef

Run a Python experiment in Docker and keep its logs and output files together.
Early development: the local runner saves source, Git and environment provenance
and installs supported locked dependencies.

Requires Python 3.11+, Git, and Docker running Linux containers. Host targets are
Linux, macOS, and Windows; platform validation is still in progress.

```text
uv tool install .
mlef init
mlef run examples/train.py -- --accuracy 0.95
mlef run examples/train.py -- --fail
```

Run these commands from the Git project root. `init` creates `.mlef/runs/` and
adds `/.mlef/` to `.gitignore`. Each run prints its ID and result directory.
Inspect that directory's `run.json`, `environment.json`, `stdout.log`, `stderr.log`, `runner.log`,
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
Root dependency manifests and locks are included even when untracked.
These are additive fields on the current version-1 run metadata; the complete
version-2 inspection/result contract is still planned.
It resolves `python:3.12-slim` to an immutable digest and executes from `/workspace`; your working tree
is never mounted into the experiment. Run storage and credential files are
excluded. Symlink inputs are currently unsupported. Changes made while source
is being copied can affect the copy, so avoid editing inputs during preparation.

Dependencies use an existing `pyproject.toml` plus `uv.lock`. The pinned uv 0.13.6
installer runs `uv sync --locked` with the container's Python, no dependency groups
or extras, and no installation of the project itself. Registry packages must have
compatible wheels; source builds, workspace/path/Git dependencies, alternative
dependency managers and separate `uv.toml` configuration are unsupported. A stale
lock or unsupported input fails preparation and preserves the captured inputs.
Scripts with no dependency manifest run without uv. A tooling-only pyproject
without runtime requirements can also run without a lock; declared dependencies,
dynamic metadata or Python requirements need a lock.

`environment.json` records the host, resolved base/uv/execution image identities,
install command, and actual container Python, OS and package versions. `run.json`
also records dependency and Dockerfile hashes. Capture failures prevent execution
and retain an incomplete-environment warning. Host environment variables are not
collected. Recorded images and packages still need to be available for a rebuild;
datasets, network responses and nondeterminism remain the caller's responsibility.
Result-query commands and background execution are still planned. Experiment code
runs with network access and can write to its artifact mount.

Try the small locked example in its own Git project (from this checkout):

```text
python -c "import shutil; shutil.copytree('examples/locked', '../mlef-locked-demo')"
cd ../mlef-locked-demo
git init
mlef init
mlef run train.py
```

To manually repeat saved inputs after editing the working tree, create a fresh
build directory containing `workspace/` extracted from the saved `source.tar.gz`
and the saved `Dockerfile`. Verify the recorded SHA-256 values first. Archives
created by mlef contain regular files; for external archives use an extractor that
rejects escaping paths and links. Build with `docker build -t mlef-repeat .`, then
execute the saved `run.json` command in that image with `/workspace` as its working
directory and `MLEF_OUTPUT_DIR=/outputs` bound to a fresh output directory. Use the
same user ID/group as the original run when reproducing permissions. The saved
recipe pins the original base and uv image digests and installs from the archived
lock. Retrieve declared `--output` directories before removing the container.

Development:

```text
uv sync --locked --extra dev
uv run --locked --extra dev ruff check .
uv run --locked --extra dev ruff format --check .
uv run --locked --extra dev ty check
uv run --locked --extra dev pytest -v
```

Set `MLEF_TEST_DOCKER=1` to include real Docker integration checks. CI runs unit
checks on Linux, macOS, and Windows and container checks on Linux.

Ruff enforces formatting, imports, annotations (including tests), bug-risk checks,
pytest conventions, and simplification rules. ty checks source, tests, and examples
against Python 3.11 across platforms, with all diagnostics treated as errors.
Both checks run in CI. Keep any necessary suppressions specific and explain them.

Licensed under Apache-2.0; see [LICENSE](LICENSE).
