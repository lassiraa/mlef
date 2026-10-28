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
New runs write version-2 Pydantic metadata. Inspection reads pre-release version-1
records without rewriting them and marks unavailable evidence as unknown.
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
Background execution is still planned. Experiment code
runs with network access and can write to its artifact mount.

Inspect runs without Docker using `mlef list [--json]` and
`mlef status <run-id> [--json]`. `list` orders newest first, skips corrupt records
with stderr diagnostics, and returns a JSON array. An unfinished state describes
the last saved state; current process liveness is unknown. Unknown IDs, unsupported
versions, and corrupt records return exit 1; JSON failures include a structured
error on stdout. Invalid arguments return exit 2.

`mlef get <run-id>` lists artifacts. Retrieve a file or directory with
`mlef get <run-id> <artifact-path> --output <destination>`; the default destination
is the artifact basename in the current directory. Existing destinations, escaping
paths, links and special files are refused. Full UUID4 run IDs are required.

Use `mlef run train.py --experiment experiment.json` to store a hypothesis,
`parent_run_id`, `changed_variables`, canonical `parameters`, training `seeds`,
`dataset` reference/fingerprint, `evaluation` identity, and `metric_directions`.
The version-1 specification rejects unknown fields, invalid types and non-finite
numbers before allocating a run. Parent IDs must already exist. Metadata is saved
by value and does not inject settings into the script or clone a parent's source.

Scripts may write `metrics.json` under `MLEF_OUTPUT_DIR`: a flat object of finite
numbers, such as `{"accuracy": 0.9}`. Booleans, duplicate keys, nested values, NaN
and infinity are invalid. `result.json` records normalized metrics, artifact
references and stable diagnostics. Missing/invalid optional metrics remain
retrievable and do not turn a successful script into a failed run. Missing or
corrupt saved results are displayed as unavailable/invalid evidence.

`mlef compare <baseline-id> <candidate-id> [--json]` reports the union of metrics,
absolute deltas and relative deltas `(candidate - baseline) / abs(baseline)`
(unavailable for a zero baseline), parameter changes, provenance, execution time,
intentional changes and artifact references. Human status/comparison summaries
show at most 20 metrics; JSON includes all metrics without embedding logs.
Per-metric compatibility is `declared-compatible`, `incompatible` or `unknown`.
Compatible declarations require matching nonempty dataset fingerprints,
evaluation `protocol_id` and `split_fingerprint`, explicitly supplied evaluation
`seeds` (an empty map is valid), metric definitions, and maximize/minimize
directions. Known conflicts take precedence over missing information; unsuccessful
runs and missing evidence cannot support improvement claims. These checks compare
declarations, not actual split use, statistical significance or leakage freedom.

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
