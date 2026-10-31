"""The initial mlef command line: initialize a project and run a script."""

import argparse
import json
import sys
from pathlib import Path
from typing import NoReturn

from typing_extensions import override

from . import __version__
from .comparison import compare
from .evidence import read_result
from .pipeline import validate
from .project import initialize, project_root
from .recovery import recover
from .runner import run
from .storage import QueryError, artifacts, inspect, list_runs, retrieve
from .tabular_comparison import compare_tabular
from .tabular_diagnostics import diagnose, read_profile


def init(root: Path, *, scaffold: bool = True) -> None:
    initialize(root, scaffold=scaffold)
    print(f"Initialized mlef storage: {root / '.mlef'}")


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    json_requested = (
        "--json" in arguments[: arguments.index("--")]
        if "--" in arguments
        else "--json" in arguments
    )

    class Parser(argparse.ArgumentParser):
        @override
        def error(self, message: str) -> NoReturn:
            if json_requested:
                print(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "error": {"category": "arguments", "message": message},
                        }
                    )
                )
            super().error(message)

    parser = Parser(prog="mlef", description="Run Python experiments in Docker.")
    parser.add_argument("--version", action="version", version=f"mlef {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    evaluation = commands.add_parser("evaluate", help="Evaluate exact model versions sequentially.")
    evaluation.add_argument("definition", type=Path)
    evaluation.add_argument("--model", action="append", required=True)
    evaluation.add_argument("--timeout", type=float)
    evaluation.add_argument("--cpus", type=float)
    evaluation.add_argument("--memory")
    evaluation.add_argument("--network", choices=["default", "none"], default="default")
    evaluation.add_argument("--max-log-bytes", type=int)
    evaluation.add_argument("--max-runner-log-bytes", type=int)
    evaluation.add_argument("--json", action="store_true")
    evaluations = commands.add_parser(
        "evaluations", help="Inspect saved evaluation batches offline."
    )
    evaluation_commands = evaluations.add_subparsers(dest="evaluations_command", required=True)
    evaluation_list = evaluation_commands.add_parser("list")
    evaluation_list.add_argument("--model")
    evaluation_list.add_argument("--limit", type=int, default=50)
    evaluation_list.add_argument("--offset", type=int, default=0)
    evaluation_list.add_argument("--json", action="store_true")
    evaluation_show = evaluation_commands.add_parser("show")
    evaluation_show.add_argument("batch_id")
    evaluation_show.add_argument("--json", action="store_true")
    for name in ["compare", "leaderboard"]:
        comparison = evaluation_commands.add_parser(name)
        if name == "compare":
            comparison.add_argument("run_ids", nargs="+")
        else:
            comparison.add_argument("--batch", required=True)
        comparison.add_argument("--baseline", required=True)
        comparison.add_argument("--metric", required=True)
        comparison.add_argument("--scope")
        comparison.add_argument("--output", help="Qualify the metrics artifact output name.")
        comparison.add_argument("--json", action="store_true")
    model_parser = commands.add_parser(
        "models", help="Discover and register saved model artifacts."
    )
    model_commands = model_parser.add_subparsers(dest="models_command", required=True)
    model_list = model_commands.add_parser("list")
    model_list.add_argument("--run", dest="run_id")
    model_list.add_argument("--json", action="store_true")
    model_register = model_commands.add_parser("register")
    model_register.add_argument("producer", help="RUN/STEP/OUTPUT")
    model_register.add_argument("--name", required=True)
    model_register.add_argument("--json", action="store_true")
    model_show = model_commands.add_parser("show")
    model_show.add_argument("selector", help="MODEL@VERSION")
    model_show.add_argument("--json", action="store_true")
    model_evaluations = model_commands.add_parser("evaluations")
    model_evaluations.add_argument("selector")
    model_evaluations.add_argument("--limit", type=int, default=50)
    model_evaluations.add_argument("--offset", type=int, default=0)
    model_evaluations.add_argument("--json", action="store_true")
    model_get = model_commands.add_parser("get")
    model_get.add_argument("selector", help="MODEL@VERSION")
    model_get.add_argument("--output", required=True, type=Path)
    history = commands.add_parser("runs", help="Inspect saved run history offline.")
    history_commands = history.add_subparsers(dest="runs_command", required=True)
    history_list = history_commands.add_parser("list")
    history_list.add_argument(
        "--state", choices=["preparing", "running", "succeeded", "failed", "cancelled"]
    )
    history_list.add_argument("--since", help="Inclusive ISO timestamp with timezone offset.")
    history_list.add_argument("--limit", type=int, default=50)
    history_list.add_argument("--offset", type=int, default=0)
    history_list.add_argument("--json", action="store_true")
    metric_query = history_commands.add_parser("metrics")
    metric_query.add_argument("run_id")
    metric_query.add_argument("--step")
    metric_query.add_argument("--scope")
    metric_query.add_argument("--metric")
    metric_query.add_argument("--max-bytes", type=int, default=32 * 1024 * 1024)
    metric_query.add_argument("--json", action="store_true")
    for name in ["show", "steps"]:
        detail = history_commands.add_parser(name)
        detail.add_argument("run_id")
        detail.add_argument("--json", action="store_true")
    initialization = commands.add_parser("init", help="Initialize storage and a script pipeline.")
    initialization.add_argument(
        "--storage-only", action="store_true", help="Do not create the example pipeline."
    )
    for name, flag in [("publish", "--to"), ("fetch", "--from")]:
        transfer = commands.add_parser(name, help="Transfer a run through a local or S3 store.")
        transfer.add_argument("run_id")
        transfer.add_argument(flag, dest="store", required=True)
        transfer.add_argument("--json", action="store_true")
    validation = commands.add_parser("validate", help="Validate pipeline wiring without execution.")
    validation.add_argument("definition", nargs="?", type=Path, default=Path("mlef.toml"))
    validation.add_argument("--json", action="store_true")
    tabular = commands.add_parser(
        "tabular", help="Inspect actual saved tabular evaluation evidence."
    )
    tabular_commands = tabular.add_subparsers(dest="tabular_command", required=True)
    tabular_export = tabular_commands.add_parser("export")
    tabular_export.add_argument("run_id")
    tabular_export.add_argument("--data", type=Path, required=True)
    tabular_export.add_argument("--output", type=Path, required=True)
    tabular_export.add_argument("--profile", action="store_true")
    tabular_compare = tabular_commands.add_parser("compare")
    tabular_compare.add_argument("baseline")
    tabular_compare.add_argument("candidate")
    tabular_compare.add_argument("--limit", type=int, default=20)
    tabular_compare.add_argument("--json", action="store_true")
    profile_parser = tabular_commands.add_parser("profile")
    profile_parser.add_argument("run_id")
    profile_parser.add_argument("--json", action="store_true")
    diagnostics = tabular_commands.add_parser("diagnose")
    diagnostics.add_argument("baseline")
    diagnostics.add_argument("candidate")
    diagnostics.add_argument("--minimum-count", type=int, default=20)
    diagnostics.add_argument("--limit", type=int, default=20)
    diagnostics.add_argument("--json", action="store_true")
    execute = commands.add_parser("run", help="Run a script or captured pipeline in Docker.")
    execute.add_argument("script", type=Path, nargs="?", default=Path("mlef.toml"))
    execute.add_argument("--include", action="append", default=[], type=Path, metavar="PATH")
    execute.add_argument("--output", action="append", default=[], type=Path, metavar="DIRECTORY")
    execute.add_argument("--experiment", type=Path, metavar="JSON_FILE")
    execute.add_argument("--reuse-from", metavar="RUN_ID")
    execute.add_argument("--from-step", metavar="STEP")
    execute.add_argument("--resume", action="store_true")
    execute.add_argument("--cache-from", metavar="RUN_ID")
    execute.add_argument("--force", action="append", default=[], metavar="STEP")
    execute.add_argument("--timeout", type=float, metavar="SECONDS")
    execute.add_argument("--cpus", type=float)
    execute.add_argument("--memory")
    execute.add_argument("--network", choices=["default", "none"], default="default")
    execute.add_argument("--max-log-bytes", type=int)
    execute.add_argument("--max-runner-log-bytes", type=int)
    for name in ["list", "status", "get", "compare", "recover", "cache"]:
        query = commands.add_parser(name, help=f"{name.capitalize()} saved local runs/artifacts.")
        if name == "compare":
            query.add_argument("baseline")
            query.add_argument("candidate")
        elif name != "list":
            query.add_argument("run_id")
        if name == "get":
            query.add_argument("artifact", nargs="?")
            query.add_argument("--output", type=Path)
        else:
            query.add_argument("--json", action="store_true")

    # Explicitly split here so script flags can never be interpreted as mlef options.
    script_arguments = []
    if "--" in arguments:
        boundary = arguments.index("--")
        script_arguments = arguments[boundary + 1 :]
        arguments = arguments[:boundary]
    args = parser.parse_args(arguments)
    if args.command != "run" and script_arguments:
        parser.error("Only mlef run accepts script arguments after --.")
    try:
        root = project_root()
        if args.command in {"runs", "models", "evaluate", "evaluations"}:
            from . import api

            try:
                if args.command == "evaluate":
                    from .limits import memory_bytes

                    batch = api.evaluate(
                        args.definition,
                        args.model,
                        root=root,
                        options=api.RunOptions(
                            timeout_seconds=args.timeout,
                            cpus=args.cpus,
                            memory_bytes=memory_bytes(args.memory)
                            if args.memory is not None
                            else None,
                            network=args.network,
                            max_log_bytes=args.max_log_bytes,
                            max_runner_log_bytes=args.max_runner_log_bytes,
                        ),
                    )
                    print(batch.model_dump_json(indent=None if args.json else 2))
                    return (
                        0 if batch.state == "complete" else 130 if batch.state == "cancelled" else 1
                    )
                if args.command == "evaluations":
                    if args.evaluations_command in {"compare", "leaderboard"}:
                        report = (
                            api.compare_evaluations(
                                args.run_ids,
                                root=root,
                                baseline=args.baseline,
                                metric=args.metric,
                                scope=args.scope,
                                output=args.output,
                            )
                            if args.evaluations_command == "compare"
                            else api.evaluation_leaderboard(
                                args.batch,
                                root=root,
                                baseline=args.baseline,
                                metric=args.metric,
                                scope=args.scope,
                                output=args.output,
                            )
                        )
                        if args.json:
                            print(report.model_dump_json())
                        else:
                            print(f"Metric: {report.metric}; baseline: {report.baseline_id}")
                            for row in report.rows:
                                print(
                                    f"{row.rank or '-'}  {row.run_id}  {row.model or '?'}  "
                                    f"value={row.value}  delta={row.delta}  "
                                    f"{row.compatibility}  {row.observation}"
                                )
                                for reason in row.reasons:
                                    print(f"  {reason}")
                        return 0
                    if args.evaluations_command == "list":
                        batches = api.list_evaluations(
                            root=root, model=args.model, limit=args.limit, offset=args.offset
                        )
                        print(batches.model_dump_json(indent=None if args.json else 2))
                    else:
                        batch = api.evaluation_batch(args.batch_id, root=root)
                        print(batch.model_dump_json(indent=None if args.json else 2))
                    return 0
                if args.command == "models":
                    if args.models_command == "evaluations":
                        batches = api.list_evaluations(
                            root=root, model=args.selector, limit=args.limit, offset=args.offset
                        )
                        print(batches.model_dump_json(indent=None if args.json else 2))
                        return 0
                    if args.models_command == "get":
                        print(api.get_model(args.selector, root=root, output=args.output))
                        return 0
                    if args.models_command == "list":
                        model_result = api.list_models(root=root, run_id=args.run_id)
                        print(model_result.model_dump_json(indent=None if args.json else 2))
                    elif args.models_command == "register":
                        parts = args.producer.split("/")
                        if len(parts) != 3:
                            raise ValueError("Expected RUN/STEP/OUTPUT.")
                        version = api.register_model(*parts, name=args.name, root=root)
                        print(
                            version.model_dump_json()
                            if args.json
                            else f"{version.name}@{version.version}"
                        )
                    else:
                        version = api.model_version(args.selector, root=root)
                        print(version.model_dump_json(indent=None if args.json else 2))
                    return 0
                if args.runs_command == "list":
                    result = api.run_history(
                        root=root,
                        state=args.state,
                        since=args.since,
                        limit=args.limit,
                        offset=args.offset,
                    )
                    if args.json:
                        print(result.model_dump_json())
                    else:
                        for diagnostic in result.diagnostics:
                            print(f"mlef: {diagnostic}", file=sys.stderr)
                        for record in result.runs:
                            print(
                                f"{record.id}  {record.state}  "
                                f"{record.created_at}  {record.command}"
                            )
                        print(
                            f"{len(result.runs)} of {result.total} runs; "
                            f"next offset: {result.next_offset}"
                        )
                elif args.runs_command == "metrics":
                    metrics = api.run_metrics(
                        args.run_id,
                        root=root,
                        step=args.step,
                        scope=args.scope,
                        metric=args.metric,
                        max_bytes=args.max_bytes,
                    )
                    print(metrics.model_dump_json(indent=None if args.json else 2))
                elif args.runs_command == "steps":
                    pipeline = api.pipeline_status(args.run_id, root=root)
                    if args.json:
                        print(pipeline.model_dump_json())
                    else:
                        for name in pipeline.order:
                            print(f"{name}: {pipeline.attempts[name].state}")
                else:
                    status = api.status(args.run_id, root=root)
                    print(status.model_dump_json(indent=None if args.json else 2))
            except api.MlefError as error:
                if error.category == "configuration":
                    raise ValueError(str(error)) from error
                raise QueryError(error.category, str(error)) from error
            return 0
        if args.command == "cache":
            from .reuse import verify_source

            record = verify_source(root, args.run_id)
            if args.json:
                print(record.model_dump_json())
            else:
                for name in record.order:
                    decision = record.attempts[name].cache
                    print(
                        f"{name}: unknown (no cache decision recorded)"
                        if decision is None
                        else f"{name}: {decision.action}: {' '.join(decision.reasons)}"
                    )
            return 0
        if args.command in {"publish", "fetch"}:
            from .transfers import backend, fetch, publish

            action = publish if args.command == "publish" else fetch
            result = action(root, args.run_id, backend(args.store))
            print(result.model_dump_json() if args.json else f"{args.command}: {result.run_id}")
            return 0
        if args.command == "validate":
            report = validate(root, args.definition)
            if args.json:
                print(report.model_dump_json())
            else:
                print(f"Pipeline: {report.status}")
                for item in report.checks:
                    print(f"{item.status}: {item.step or 'pipeline'}: {item.message}")
            return {"verified": 0, "invalid": 1, "deferred": 3}[report.status]
        if args.command == "tabular":
            if args.tabular_command == "export":
                from .tabular_artifacts import export_artifacts

                bundle = export_artifacts(
                    root, args.run_id, root / args.data, root / args.output, profile=args.profile
                )
                print(bundle.model_dump_json())
                return 0
            if args.tabular_command == "profile":
                profile = read_profile(root, args.run_id)
                print(
                    profile.model_dump_json()
                    if args.json
                    else f"{profile.row_count} rows; {profile.feature_count} features"
                )
                return 0
            if args.tabular_command == "diagnose":
                feedback = diagnose(
                    root,
                    args.baseline,
                    args.candidate,
                    minimum_count=args.minimum_count,
                    limit=args.limit,
                )
                print(
                    feedback.model_dump_json()
                    if args.json
                    else f"{feedback.comparison.observation}; {len(feedback.slices)} error slices"
                )
                return 0
            result = compare_tabular(root, args.baseline, args.candidate, limit=args.limit)
            print(
                result.model_dump_json()
                if args.json
                else f"{result.compatibility}: {result.observation}; delta={result.delta}"
            )
            return 0
        if args.command == "init":
            init(root, scaffold=not args.storage_only)
            return 0
        if args.command == "list":
            records, diagnostics = list_runs(root)
            for diagnostic in diagnostics:
                print(f"mlef: {diagnostic}", file=sys.stderr)
            if args.json:
                print(json.dumps([record.model_dump(mode="json") for record in records]))
            else:
                for record in records:
                    print(f"{record.id}  {record.state}  {record.created_at}  {record.command}")
            return 0
        if args.command == "status":
            record = inspect(root, args.run_id)
            evidence = read_result(root, args.run_id)
            if args.json:
                print(
                    json.dumps(
                        {
                            **record.model_dump(mode="json"),
                            "result": evidence.model_dump(mode="json"),
                        }
                    )
                )
            else:
                print(f"{record.id}: {record.state}\nCommand: {record.command}")
                print(f"Created: {record.created_at}\nStarted: {record.started_at}")
                print(f"Finished: {record.finished_at}")
                print(f"Execution exit code: {record.execution_exit_code}")
                if record.error is not None:
                    print(f"Error ({record.error.category}): {record.error.message}")
                for warning in record.warnings:
                    print(f"Warning: {warning}")
                print(f"Evidence: {evidence.evidence_status}")
                for name, value in list(evidence.metrics.items())[:20]:
                    print(f"  {name}: {value}")
                if len(evidence.metrics) > 20:
                    print(f"  {len(evidence.metrics) - 20} metrics omitted; use --json.")
                for diagnostic in evidence.diagnostics:
                    print(f"{diagnostic.code}: {diagnostic.message}")
                print(f"Artifacts: .mlef/runs/{record.id}/artifacts/")
            return 0
        if args.command == "recover":
            record = recover(root, args.run_id)
            print(record.model_dump_json() if args.json else f"{record.id}: {record.state}")
            return 0
        if args.command == "compare":
            comparison = compare(root, args.baseline, args.candidate)
            if args.json:
                print(comparison.model_dump_json())
            else:
                print(
                    f"{comparison.baseline_id} -> {comparison.candidate_id}: "
                    f"{comparison.compatibility}"
                )
                print(
                    f"Execution seconds: {comparison.baseline_seconds} -> "
                    f"{comparison.candidate_seconds}"
                )
                for metric in comparison.metrics[:20]:
                    print(
                        f"{metric.name}: {metric.baseline} -> {metric.candidate}; "
                        f"delta={metric.delta}; relative={metric.relative_delta}; "
                        f"{metric.compatibility}; {metric.observation}"
                    )
                    for reason in metric.reasons:
                        print(f"  {reason}")
                if len(comparison.metrics) > 20:
                    print(f"{len(comparison.metrics) - 20} metrics omitted; use --json.")
                print(
                    f"Changed parameters: {len(comparison.parameters)}; "
                    f"provenance fields: {list(comparison.provenance)}"
                )
                for name, change in list(comparison.parameters.items())[:20]:
                    print(f"  {name}: {change.baseline!r} -> {change.candidate!r}")
                if len(comparison.parameters) > 20:
                    print(
                        f"{len(comparison.parameters) - 20} parameter changes omitted; use --json."
                    )
                print(
                    f"Artifacts: .mlef/runs/{args.baseline}/artifacts/ "
                    f"and .mlef/runs/{args.candidate}/artifacts/"
                )
                for limitation in comparison.limitations:
                    print(limitation)
            return 0
        if args.command == "get":
            if args.artifact is None:
                if args.output is not None:
                    parser.error("--output requires an artifact path.")
                print("\n".join(artifacts(root, args.run_id)))
            else:
                print(retrieve(root, args.run_id, args.artifact, args.output))
            return 0
        if (
            args.max_log_bytes is not None
            or args.max_runner_log_bytes is not None
            or args.timeout is not None
            or args.cpus is not None
            or args.memory is not None
            or args.network != "default"
            or args.reuse_from is not None
            or args.from_step is not None
            or args.resume
            or args.cache_from is not None
            or args.force
        ):
            return run(
                root,
                args.script,
                script_arguments,
                args.include,
                args.output,
                args.experiment,
                timeout=args.timeout,
                cpus=args.cpus,
                memory=args.memory,
                network=args.network,
                max_log_bytes=args.max_log_bytes,
                max_runner_log_bytes=args.max_runner_log_bytes,
                reuse_from=args.reuse_from,
                from_step=args.from_step,
                resume=args.resume,
                cache_from=args.cache_from,
                force=args.force,
            )
        if args.experiment is not None:
            return run(
                root, args.script, script_arguments, args.include, args.output, args.experiment
            )
        return run(root, args.script, script_arguments, args.include, args.output)
    except QueryError as error:
        if getattr(args, "json", False):
            print(
                json.dumps(
                    {
                        "schema_version": 1,
                        "error": {"category": error.category, "message": str(error)},
                    }
                )
            )
        print(f"mlef: {error}", file=sys.stderr)
        return 1
    except ValueError as error:
        if getattr(args, "json", False):
            print(
                json.dumps(
                    {
                        "schema_version": 1,
                        "error": {"category": "configuration", "message": str(error)},
                    }
                )
            )
        print(f"mlef: {error}", file=sys.stderr)
        return 2
    except FileNotFoundError as error:
        if getattr(args, "json", False):
            print(
                json.dumps(
                    {"schema_version": 1, "error": {"category": "storage", "message": str(error)}}
                )
            )
        print(f"mlef: Required executable or file is missing: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        if getattr(args, "json", False):
            print(
                json.dumps(
                    {"schema_version": 1, "error": {"category": "storage", "message": str(error)}}
                )
            )
        print(f"mlef: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("mlef: Cancelled.", file=sys.stderr)
        return 130
