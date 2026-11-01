# MLEF

Run Python experiments and pipelines in Docker, track runs and model versions,
and compare saved models with a frozen evaluator. Use the CLI yourself or the
Python API from an agent.

MLEF keeps source snapshots, locked environments, logs, metrics and artifacts
together in your Git project's `.mlef/` directory. No account, database or LLM
provider is required. Version: **0.1.0**.

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), Git,
Python 3.11-3.14, and Docker with a running Linux-container engine:

```text
git clone https://github.com/lassiraa/mlef.git
cd mlef
uv tool install --python 3.12 .
mlef init
mlef run examples/train.py -- --accuracy 0.95
mlef list
```

The first run downloads/builds its container environment. Copy the complete run
UUID printed by `run` and use it in place of `RUN`:

```text
mlef status RUN --json
mlef get RUN
mlef get RUN metrics.json --output saved-metrics.json
```

The script writes illustrative metrics and a dummy model. For real training, use
the tabular example below. See [setup](docs/setup.rst) for installation in your
own project, Python API environments, dependencies and platform support.

In a fresh project, `mlef init` also creates `mlef.toml` and four `steps/` scripts
demonstrating typed artifact handoffs. Existing scaffold paths are left untouched;
`mlef init --storage-only` omits the example. See the
[pipeline guide](docs/pipelines.rst) to validate and run these scripts locally or
as a captured Docker pipeline with `mlef run`. Pipelines require an existing
locked experiment environment containing the SDK runtime dependencies; the runner
installs and records its exact SDK wheel without resolving dependencies.
`mlef run script.py` continues to run individual scripts as before.
To edit training without rerunning ingestion, use
`mlef run --reuse-from RUN --from-step train`: verified upstream outputs are
copied into a self-contained new run with their original producer references.
Add `--resume` to use the original captured source and SDK rather than live edits.

## What makes it useful?

### Track runs, metrics and experiments

Group runs, give them names and tags, and inspect their saved evidence offline:

```text
mlef run examples/train.py --experiment-name demo --name baseline --tag family=linear
mlef runs list --experiment demo --state succeeded --limit 20 --json
mlef runs show RUN --json
mlef runs metrics RUN --json
mlef runs annotate RUN --note "Reviewed baseline" --tag reviewed=yes
mlef experiments show demo --json
```

History supports state, time, experiment, name and tag filters with pagination.
Metric queries include root metrics and completed pipeline metric artifacts;
use `--step`, `--scope` and `--metric` to narrow pipeline results. Labels and
notes are editable without changing captured evidence. Archive a run with
`mlef runs annotate RUN --archive` to hide it from history without deleting it.
See [run tracking](docs/runs.rst).

### Track model versions and compare evaluations

Discover typed model outputs from captured pipelines, register immutable versions,
and assign aliases such as `baseline` or `champion`:

```text
mlef models list --run RUN --json
mlef models register RUN/train/model --name classifier
mlef models show classifier@VERSION_A --json
mlef models alias classifier@VERSION_A --alias baseline
mlef models alias-history classifier:baseline --json
```

Use a run with a declared model output; the quick-start dummy model is a generic
file. Replace `RUN/train/model` with its actual run, step and output, and
`VERSION_A`/`VERSION_B` with hashes returned by registration. Define
`evaluation.toml` and its evaluator using the [evaluation guide](docs/evaluations.rst),
then evaluate registered models on explicitly bound saved data:

```text
mlef evaluate evaluation.toml --model classifier:baseline --model classifier@VERSION_B
mlef evaluations show BATCH --json
mlef evaluations leaderboard --batch BATCH --baseline EVAL_A --metric mae --scope test --json
mlef models evaluations classifier:baseline --json
mlef models get classifier:baseline --output saved-model
```

`BATCH` is the returned batch UUID; `EVAL_A` is its baseline child run UUID.
Choose a metric and scope emitted by your evaluator. Each model runs sequentially
in its own container with captured evaluator source and exact model/data inputs.
Moving an alias cannot change a captured selection. Comparisons verify evidence
and only rank compatible successful scalar results; failed, missing and
incompatible results remain visible. See [models and aliases](docs/models.rst).

### Build typed pipelines and reuse saved inputs

Declare script inputs and outputs with the SDK, validate their wiring with
`mlef validate`, and run the pipeline with `mlef run`. Steps can declare `train`,
`predict` and `evaluate` roles, with model/data/prediction lineage checked when
outputs are saved. Exact cross-run bindings let an evaluator consume existing
models and datasets without repeating training. Explicit reuse, resume and
opt-in caching retain the original artifact producers in each new run.
See [script pipelines](docs/pipelines.rst) and the [artifact SDK](docs/script-artifacts.rst).

### Keep results when an experiment fails

```text
mlef run examples/train.py -- --fail
```

This deliberately fails after writing outputs. Logs, metrics and partial artifacts
remain available through `status` and `get`. Your own scripts write results to
`os.environ["MLEF_OUTPUT_DIR"]`:

```python
import json
import os
from pathlib import Path

output = Path(os.environ["MLEF_OUTPUT_DIR"])
(output / "metrics.json").write_text(json.dumps({"accuracy": 0.9}))
```

### Compare sklearn and XGBoost on saved folds

The [tabular demo](examples/sklearn/demo.py) prepares folds once, evaluates
baseline and candidate models, compares predictions, refits the selected candidate
and retrieves a model plus inference CSV. It also exports verified SDK dataset,
split, CV recipe, prediction and metric artifacts for each candidate. Begin in the MLEF checkout:

```text
python -c "import shutil; shutil.copytree('examples/sklearn', '../mlef-tabular-demo', ignore=shutil.ignore_patterns('.venv', '__pycache__'))"
cd ../mlef-tabular-demo
git init
uv run --project ../mlef --locked python demo.py
```

`--project` supplies the host API environment while keeping the copied experiment
as the working directory. ML dependencies are installed inside Docker. For
regression, use another fresh copy and run `demo.py --regression`.
See [tabular workflows](docs/tabular.rst) for minimal factories, individual commands,
score definitions and dataset attribution. Validation selection alone does not
establish held-out gains.

### Give an agent structured feedback

From the experiment's Git root, replace the identifiers with saved run UUIDs:

```text
mlef tabular profile RUN --json
mlef tabular compare BASELINE CANDIDATE --json
mlef tabular diagnose BASELINE CANDIDATE --json
```

An agent can inspect missingness, propose a change to ordinary Python model code,
evaluate it on saved folds, and compare scores, error slices and runtime. These
queries work offline without Docker or ML libraries. The [Python API](docs/api.rst)
exposes the same operations with Pydantic results.

## Documentation

Read the **[documentation website](https://lassiraa.github.io/mlef/)** for searchable
guides and API reference. It is hosted free on GitHub Pages and updated from `main`.
The documentation sources are linked below.

| Guide | Contents |
| --- | --- |
| [Setup](docs/setup.rst) | Installation, host vs experiment environments, troubleshooting |
| [Tabular workflows](docs/tabular.rst) | sklearn/XGBoost, regression, refit and agent feedback |
| [Artifact contracts](docs/artifacts.rst) | Versioned manifests, custom codecs and inline/external metric evidence |
| [Script pipelines](docs/pipelines.rst) | Captured execution, explicit reuse/resume and explainable opt-in caching |
| [Script artifact SDK](docs/script-artifacts.rst) | Managed outputs, atomic saves and verified read-only input handoff |
| [Python API](docs/api.rst) | Execution, tracking, model and evaluation APIs, errors and callbacks |
| [Run and CLI reference](docs/runs.rst) | Filtered history, experiment groups, annotations, metrics and recovery |
| [Models](docs/models.rst) | Discover and register versions, retrieve models, manage aliases and history |
| [Evaluation batches](docs/evaluations.rst) | Frozen multi-model evaluation, verified comparisons and leaderboards |
| [Run storage](docs/storage.rst) | Verified local/S3 publication, retrieval, retries and credentials |
| [Agent benchmarks](docs/benchmarks.rst) | Matched trials, private scoring and reporting |
| [Development](docs/development.rst) | Checks, schemas, packaging and contributions |

Experiment execution currently requires Docker. Host tools and offline inspection
run natively. Use trusted experiment code and model files. Per-run controls do not
enforce a global agent budget or hard artifact quota. macOS container execution
remains preview; CLI/package checks cover Linux, macOS and Windows.

Licensed under [Apache-2.0](LICENSE). The classification example includes adapted
[CC BY 4.0 data](docs/tabular.rst#dataset-attribution).
