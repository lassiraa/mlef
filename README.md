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

`--max-log-bytes 1000000` bounds combined saved and forwarded script stdout/stderr;
`--max-runner-log-bytes 1000000` bounds preparation/build diagnostics in runner.log.
Limits count bytes, including multibyte text, and stop the run on exhaustion with
an `output_limit` diagnostic. Native command pipes are drained through the limiter;
they never write directly to the saved log descriptor. Cleanup output is discarded
when necessary and its outcome is retained in metadata. Structured Docker replies
have a separate 4 MiB capture ceiling. These limits do not quota artifacts or total
disk use; combine them with a deadline for a total worker time budget.

`mlef recover <run-id> [--json]` explicitly finalizes an abandoned owned run.
Kernel locks protect live runners, supervising parents and concurrent recovery;
ordinary status/list queries remain read-only. Recovery requires the recorded
project identity and exact Docker ownership labels, refuses unknown legacy
ownership, and leaves metadata unchanged when Docker is unavailable. Missing
containers never imply success. Collected outputs are never overwritten; a
collection failure retains the container for inspection/retry. Recovery marks an
unfinished run failed with an `abandoned` diagnostic and can be safely repeated.
Terminal runs without pending cleanup are returned unchanged, including offline.

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
Background execution is still planned. Experiment code has network access by
default and can write to its artifact mount.

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

Python callers can use the supported synchronous `mlef.api` entry points:

```python
from mlef import api

api.init()  # Run from the Git root, or pass root="/path/to/project".
result = api.run(
    "train.py",
    arguments=["--seed", "7"],
    outputs=["results"],
    options=api.RunOptions(
        timeout_seconds=120,
        cpus=0.5,
        memory_bytes=268435456,
        network="none",
        max_log_bytes=65536,
        max_runner_log_bytes=1048576,
    ),
)
print(result.id, result.state, result.result.metrics)
```

`init`, `run`, `list_runs`, `status`, `artifacts`, `get`, `compare` and `recover`
reuse the CLI's execution, storage, comparison and recovery services. `run`,
`status` and `recover` return a Pydantic `RunStatus` containing run metadata plus
normalized `result` evidence; `list_runs` returns `RunListing(runs, diagnostics)`.
Artifact listing returns relative names; `get` returns the retrieved `Path`;
`compare` returns the existing `Comparison` contract. Use `model_dump(mode="json")`
for JSON-compatible output. Version-1/version-2 records use the same read adapters
as CLI queries. Control options are validated by `RunOptions` before execution.

Calls block until execution and finalization finish, and produce no console output.
Script failures, timeouts and Ctrl+C return their saved failed/cancelled status;
invalid project/run configuration and query failures raise `MlefError`, whose
`category` matches the service error (for example `unknown_run`, `unsafe_path` or
`configuration`). Failed execution is data, so check `state` before claiming success.
Optional invalid/missing metrics do not change the execution outcome. Pydantic
rejects invalid option values when constructing `RunOptions`.

Every operation accepts optional `root`: the exact Git root, defaulting to the
current directory. Script, include, output and experiment paths are relative to
that root; artifact names are relative to the saved artifact directory. Retrieval
destinations are relative to the caller's current directory and never overwritten.
No global working-directory change or shell command parsing is performed.

For live output, pass `on_output=lambda event: ...` to `run`. `LogEvent.stream` is
`stdout` or `stderr`; `data` contains raw bytes from the worker, including run
progress messages. Calls are serialized on background drain threads; callbacks
must return promptly and must decode partial UTF-8 chunks incrementally if needed.
`callback_bytes` defaults to 65536 bytes total per call; further delivery is
discarded and a warning is added to the returned status. Saved logs still follow
their separate limits. A callback exception stops the owned run, preserves partial
evidence, and raises `MlefError(category="callback")`. Call-specific delivery
warnings are not persisted. This API supplies foreground execution, without a
scheduler or asynchronous job protocol.

Copy `examples/tabular` to a fresh Git project and run `python python_demo.py` for
the Python baseline/candidate/compare/retrieve example, or `python demo.py` for
the CLI equivalent. Both require Docker and an installed mlef package.

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

For sklearn tabular evaluation, copy `examples/sklearn` into a fresh Git project
and run `python demo.py` with the installed mlef package and Docker. The host needs
no ML libraries. The demo exports the installed evaluator's source as `evaluate.py`,
captures it with the dataset/model/protocol, and installs the example's locked
Pydantic/pandas/sklearn dependencies inside the container. There is no dependency
on an unpublished mlef package in the experiment. `model.py` is ordinary Python:
its `make_estimator(seed)` returns a sklearn estimator/Pipeline.

Preparation saves actual stratified folds once in `protocol.json`; evaluation
refuses changed dataset bytes, IDs, invalid assignments or missing class coverage.
Each fold clones/fits the entire Pipeline on training rows only. The versioned
`evaluation.json` contains the full protocol, per-fold scores, timing and failures;
`predictions.csv` preserves fold/row alignment and has a recorded SHA-256. A failed
fold keeps partial evidence, exits unsuccessfully, and never produces a successful
aggregate `metrics.json`. Provided group/time protocols are validated too: group
partitions must be disjoint, and training timestamps must strictly precede scoring
timestamps (timezone-aware ISO-8601). Temporal protocols may omit early scoring rows.

The classification fixture adapts three features from the
[Wisconsin Diagnostic dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
by Wolberg, Mangasarian, Street and Street (1993), DOI 10.24432/C5DW2B,
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Row IDs are
renumbered, every seventeenth radius is blank, and `radius_band` uses a fixed
radius threshold of 14. These are illustrative alterations; positive class `1`
means benign in sklearn's source labels. This example supplies no clinical claim.
Editable experiment code remains trusted: protocol hashes and pipeline evaluation
do not establish tamper resistance or absence of all forms of leakage.

Compare saved fold evidence without Docker or ML libraries using
`mlef tabular compare BASELINE CANDIDATE --json` or
`api.compare_tabular(baseline, candidate)`. This checks actual assignments,
prediction byte hashes, row coverage and labels, and recomputes ROC-AUC from saved
predictions. Both execution and every required fold must succeed. Different
protocols are incompatible; absent/partial evidence cannot support an improvement
claim. Fold summaries default to 20 entries (`--limit`/`limit`, at most 100) and
report omitted counts; full evidence remains in referenced artifacts. Each input
evidence file is limited to 32 MiB for offline inspection. Generic `compare`
retains its original declaration-based behavior. These are consistency checks of
saved evidence, not independent execution attestation or statistical significance.

The sklearn demo also evaluates CPU-only XGBoost candidates, including a missing-
value-indicator/preprocessing hypothesis. XGBoost's early-stopping validation rows
come exclusively from each outer training partition; grouped/time protocols retain
those constraints for the inner partition. Preprocessing is fitted on inner training
rows and explicitly transforms the validation set. Evaluation v2 records the actual
inner assignments and best iteration; v1 evidence remains readable offline.

After comparing all candidates, the demo refits the best validation candidate on
the complete exploration dataset. Boosted models use the median CV best tree count
with early stopping disabled for this refit. `model.joblib` and `model-recipe.json`
preserve preprocessing, feature order, label semantics, parameters and versions.
`evaluate.py predict` explicitly loads a trusted model and writes positive-class
probabilities; offline inspection never deserializes models. The demo retrieves
the selected model and runs this prediction step through Docker too. No final-test
labels participate; validation selection alone is not evidence of generalization.
XGBoost host tests run on Linux/Windows; the base macOS CLI/package matrix remains
independent of XGBoost, and macOS container execution remains preview.

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
| Unfinished run after abrupt termination | Inspect saved evidence; use `mlef recover <run-id>` after the foreground owner has exited. |

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
by default and a writable artifact mount; use only trusted scripts. Deadlines,
runtime resource controls and log caps do not enforce a hard artifact quota.
Unattended-agent execution remains outside this release.

Release-candidate notes: versioned offline inspection, safe artifact retrieval,
locked execution provenance, experiment intent, normalized evidence and honest
baseline/candidate comparisons form the first complete feedback loop. Version-1
run reads are preserved, alongside version-2 reads. Python 3.11-3.14 is the tested CLI range; experiment Python
is independently pinned by the saved image recipe. macOS container validation
and broader dependency layouts remain follow-up work.

Build candidate artifacts using `uv build --out-dir .mlef/dist`, then verify them
with `uv run --locked --extra dev python scripts/check_dist.py .mlef/dist`.
Both formats are installed and exercised outside the checkout. Review the complete
PR stack and platform limitations before tagging or publishing. Publication is a
separate release action; no CI push or PR event publishes this package.

Licensed under Apache-2.0; see [LICENSE](LICENSE).
