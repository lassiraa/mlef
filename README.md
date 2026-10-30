# mlef

Run a Python experiment in Docker and keep its logs and output files together.
Early development: the local runner saves source, Git and environment provenance
and installs supported locked dependencies.

Release candidate: `0.1.0rc1`. Requires Python 3.11-3.14, Git, uv 0.13.6 for
development, and Docker running Linux containers. CLI and distribution checks run
on Linux, macOS, and Windows. Container integration is validated on Linux and on
Windows with Docker Desktop's Linux engine. macOS Docker execution remains a
preview until equivalent host checks pass.

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

`mlef run train.py --timeout 60` places one wall-clock deadline around the entire
worker, including source preparation, image pulls/builds, environment probes and
execution. On expiry, mlef kills the owned worker process tree and attempts
ownership-checked container/output cleanup within a further 15-second grace budget
(process-tree termination may take up to ten additional seconds). Available outputs
remain retrievable; interrupted evidence is marked unavailable. Engine failures
are recorded as incomplete cleanup rather than claimed successful cleanup. The
deadline does not quota artifact writes or cancel arbitrary remote engine work.
Omitting the option preserves the existing foreground behavior and stage timeouts.

Runtime controls: `mlef run train.py --cpus 0.5 --memory 64m --network none`.
Memory accepts positive bytes or integer k/m/g/t suffixes and disables additional
swap; CPU values use at most nine decimal places. Docker capability checks and
container inspection must confirm requested limits before execution. Requested
and applied values are recorded separately. The execution probe receives CPU and
memory controls and always uses isolated networking. Runtime controls do not
constrain build workers or make image/dependency downloads offline. Docker-confirmed
OOM termination is recorded separately from a script's explicit exit code 137.

The runner archives tracked files and the explicitly named script into
`source.tar.gz`, then builds from that saved archive. Include other untracked
source with `--include path`. Each run saves the archive SHA-256, included
untracked paths, Git commit/branch/dirty state, a binary-capable `changes.patch`
relative to HEAD (the empty tree for an unborn repository), and the generated
`Dockerfile`. Detached HEAD records a null branch. The archive defines the actual
source inputs; the patch contains tracked changes only.
Root dependency manifests and locks are included even when untracked.
New runs write version-3 Pydantic metadata. Inspection reads pre-release version-1/version-2
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

For a complete baseline/candidate workflow, copy `examples/tabular` into a fresh
directory, run `git init` there, then run `python demo.py` with the installed
`mlef` command on PATH. It records the actual data hash, runs a majority-class
baseline, fills the candidate's parent ID from that run, evaluates a
training-selected version threshold on the same fixed validation rows, compares
the results, and retrieves candidate metrics. The toy validation accuracy changes
from 0.5 to 1.0. This is a reproducible smoke example, not evidence of general ML
or agent performance. `event_count` is deliberately irrelevant; the script fits
the threshold on training rows only.

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
checks on Linux, macOS, and Windows, container checks on Linux, and clean wheel and
sdist installs on all three hosts. The container suite includes failure,
cancellation, declared outputs, concurrent-run isolation, locked installation,
saved-recipe rebuild, malformed optional metrics, required-result storage failure,
and the complete tabular run/inspect/compare/retrieve loop.
The full suite also passed on Windows/Python 3.12 with Docker Desktop's Linux
engine 29.8.0: 81 passed, with three Windows-inapplicable checks skipped (POSIX
signal delivery, named pipes, and the Unix symlink test).

Ruff enforces formatting, imports, annotations (including tests), bug-risk checks,
pytest conventions, and simplification rules. ty checks source, tests, and examples
against Python 3.11 across platforms, with all diagnostics treated as errors.
Both checks run in CI. Keep any necessary suppressions specific and explain them.

Versioned JSON Schemas are in [schemas](schemas). The core contracts are
[run v3](schemas/run-v3.schema.json), [experiment v1](schemas/experiment-v1.schema.json),
[result v1](schemas/result-v1.schema.json), [environment v1](schemas/environment-v1.schema.json),
and [comparison v1](schemas/comparison-v1.schema.json). Regenerate them with
`uv run --locked --extra dev python scripts/export_schemas.py`; CI checks that the
committed copies match the Pydantic contracts. `run.json`, `environment.json` and
`result.json` are saved atomically. Run directories and artifacts are authoritative;
there is no database, hosted backend, daemon or mandatory LLM service.

| Exit code | Meaning |
| --- | --- |
| 0 | Successful execution or query, including incompatible/unknown comparisons. |
| 1 | Execution, preparation, storage, collection, or query/retrieval failure. |
| 2 | Invalid arguments, experiment metadata, or project configuration. |
| 130 | Foreground run cancelled by the user. |

| Symptom | Resolution |
| --- | --- |
| Project root or initialization error | Run from the Git root; use `git init` and `mlef init`. |
| Docker engine unavailable | Start a Linux-container engine and check `docker info`. Offline queries still work. |
| Missing/stale lock | Intentionally run `uv lock` in the experiment project, then inspect and commit the dependency changes. |
| Unsupported dependency input | Use a single pyproject/uv.lock project with compatible registry wheels. |
| Unknown run | Use the complete UUID printed when the run was allocated. |
| Existing output destination | Choose a fresh destination; `get` and output collection never overwrite it. |
| Missing/invalid metrics | Inspect `result.json` diagnostics and retrieve the original artifact. |
| Unfinished run after abrupt termination | Treat its saved state as historical; liveness and recovery are not implemented. |

The matched-trial fixture starts with
`uv run --locked --extra dev python benchmarks/fixture.py prepare <trial-directory>`.
Give an agent access only to the generated `agent/` workspace, initialize Git there,
and keep `private/` outside its accessible roots and Docker inputs. Final-test
labels are held in the coordinator's private directory. The fixture does not
enforce operating-system access controls; the evaluator must configure them.
Use identical data fingerprints, agent/model versions, seeds, budgets and repetition
indices for control (ordinary Python) and treatment (MLEF) trials. Train on
`train.csv`, make decisions using `validation.csv`, and submit test predictions
only after exploration ends. Score them outside the agent workspace with
`fixture.py score <private-labels> <predictions>`.

Record a trial JSON with `fixture.py record <input> <output>`: `task_fingerprint`,
`condition` (`control` or `mlef`), `agent`, `model`, `budget_seconds`, `seed`,
`repetition`, `elapsed_seconds`, `tool_output_bytes`, `intervention_count`,
`valid_comparisons`, `time_to_first_valid_gain_seconds` (null if absent), and
`heldout_accuracy`. These are evaluator-recorded measurements, not automatic agent
telemetry. `fixture.py pair <control> <mlef>` rejects mismatched conditions and
reports paired differences. Repeat matched trials to assess repeatability; this
release includes the fixture, not a measured claim that MLEF improves agents.

Contributions should include a minimal reproduction or concrete behavior change,
focused pytest coverage for meaningful logic, and the uv checks above. Use
Conventional Commit titles and keep PR descriptions limited to committed changes
and their validation under Summary, Validation and optional Compatibility sections.
For security reports, share sensitive details privately with the maintainer before
posting a sanitized public reproduction. Do not upload credentials, private datasets
or complete sensitive run artifacts to issues. Experiment code has network access
and a writable artifact mount; use only trusted scripts. Timeouts, CPU/memory quotas,
hard output limits and unattended-agent execution are outside this release.

Release-candidate notes: versioned offline inspection, safe artifact retrieval,
locked execution provenance, experiment intent, normalized evidence and honest
baseline/candidate comparisons form the first complete feedback loop. Version-1
run reads are preserved. Python 3.11-3.14 is the tested CLI range; experiment Python
is independently pinned by the saved image recipe. macOS container validation
and broader dependency layouts remain follow-up work.

Build candidate artifacts using `uv build --out-dir .mlef/dist`, then verify them
with `uv run --locked --extra dev python scripts/check_dist.py .mlef/dist`.
Both formats are installed and exercised outside the checkout. Review the complete
PR stack and platform limitations before tagging or publishing. Publication is a
separate release action; no CI push or PR event publishes this package.

Licensed under Apache-2.0; see [LICENSE](LICENSE).
