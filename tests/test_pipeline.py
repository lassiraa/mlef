"""Pipeline inspection reads source text, never imports or executes user scripts."""

import json
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from mlef import api
from mlef.artifacts import ArtifactRef, ArtifactType
from mlef.cli import main
from mlef.pipeline import (
    ExternalInput,
    Input,
    Output,
    PipelineDefinition,
    PipelineStep,
    Step,
    ValidationReport,
    validate,
)


def script(root: Path, name: str, *, inputs: str = "", output: str = "") -> Path:
    path = root / f"{name}.py"
    path.write_text(
        "from mlef.pipeline import Step, Input, Output\n"
        "from mlef.artifacts import ArtifactType\n"
        f"step = Step(name={name!r}, inputs=[{inputs}], outputs=["
        + (output or 'Output(name="data", type=ArtifactType.FILE, format="text")')
        + "])\n",
        encoding="utf-8",
    )
    return path


def definition(root: Path, entries: str) -> Path:
    path = root / "mlef.toml"
    path.write_text("schema_version = 1\n" + entries, encoding="utf-8")
    return path


def entry(name: str, bindings: str = "") -> str:
    return f'[[steps]]\nname = "{name}"\nscript = "{name}.py"\n' + (
        f"[steps.bindings]\n{bindings}\n" if bindings else ""
    )


def test_declaration_contracts() -> None:
    step = Step(
        name="train",
        inputs=[Input(name="data", type=ArtifactType.DATASET)],
        outputs=[Output(name="model", type=ArtifactType.MODEL, format="joblib")],
    )
    assert Step.model_validate_json(step.model_dump_json()) == step
    with pytest.raises(ValidationError, match="unique"):
        Step(name="train", inputs=step.inputs * 2, outputs=step.outputs)
    with pytest.raises(ValidationError, match="unique"):
        Step(name="train", outputs=step.outputs * 2)
    for version in [True, "1", 3]:
        with pytest.raises(ValidationError):
            Step.model_validate_json(
                step.model_dump_json().replace(
                    '"schema_version":1', f'"schema_version":{json.dumps(version)}'
                )
            )


def test_legacy_step_serialization_is_unchanged() -> None:
    from mlef.pipeline import StepV1

    legacy = StepV1(
        name="plain", outputs=[Output(name="file", type=ArtifactType.FILE, format="text")]
    )
    current = Step.model_validate_json(legacy.model_dump_json())
    assert current.kind == "generic"
    assert current.model_dump_json() == legacy.model_dump_json()
    assert current.model_dump(mode="json") == legacy.model_dump(mode="json")


def test_external_bindings_require_version_two() -> None:
    identifier = str(uuid4())
    reference = ExternalInput(
        run_id=identifier,
        artifact=ArtifactRef(
            run_id=identifier,
            step="train",
            output="model",
            sha256="a" * 64,
        ),
    )
    entry = PipelineStep(name="evaluate", script="evaluate.py", bindings={"model": reference})
    with pytest.raises(ValueError, match="schema_version=2"):
        PipelineDefinition(steps=[entry])
    definition = PipelineDefinition(schema_version=2, steps=[entry])
    assert PipelineDefinition.model_validate_json(definition.model_dump_json()) == definition


@pytest.mark.parametrize(
    ("kind", "producer", "output"),
    [
        ("train", ArtifactType.DATASET, ArtifactType.MODEL),
        ("predict", ArtifactType.MODEL, ArtifactType.PREDICTIONS),
        ("evaluate", ArtifactType.MODEL, ArtifactType.METRICS),
        ("evaluate", ArtifactType.PREDICTIONS, ArtifactType.METRICS),
    ],
)
def test_typed_role_contracts(kind: str, producer: ArtifactType, output: ArtifactType) -> None:
    declarations = [Input(name="subject", type=producer)]
    if producer != ArtifactType.DATASET:
        declarations.append(Input(name="data", type=ArtifactType.DATASET))
    payload = {
        "schema_version": 2,
        "name": "work",
        "kind": kind,
        "inputs": [item.model_dump(mode="json") for item in declarations],
        "outputs": [
            Output(name="result", type=output, format="application/json").model_dump(mode="json")
        ],
    }
    step = Step.model_validate_json(json.dumps(payload))
    assert step.kind == kind
    assert Step.model_validate_json(step.model_dump_json()) == step
    payload["schema_version"] = 1
    with pytest.raises(ValueError, match="schema_version=2"):
        Step.model_validate_json(json.dumps(payload))
    payload["schema_version"] = 2
    payload["inputs"] = []
    with pytest.raises(ValueError, match="require"):
        Step.model_validate_json(json.dumps(payload))


def test_role_validation_is_static(project: Path) -> None:
    path = script(project, "train")
    path.write_text(
        path.read_text().replace("name='train'", 'schema_version=2, kind="evaluate", name="train"')
    )
    definition(project, entry("train"))
    report = validate(project)
    assert report.status == "invalid"
    assert "Evaluate steps require" in report.model_dump_json()


def test_static_inspection_does_not_execute_imports_or_top_level_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = script(tmp_path, "ingest")
    path.write_text(
        "import nonexistent_dangerous_dependency\n"
        "from pathlib import Path\n"
        "Path('executed').write_text('unsafe')\n"
        "raise RuntimeError('do not execute')\n" + path.read_text(),
        encoding="utf-8",
    )
    definition(tmp_path, entry("ingest"))
    report = validate(tmp_path)
    assert report.status == "verified"
    assert report.order == ["ingest"]
    assert not (tmp_path / "executed").exists()
    assert ValidationReport.model_validate_json(report.model_dump_json()) == report


def test_deterministic_dependency_order_and_fan_out(tmp_path: Path) -> None:
    for name in ["zebra", "alpha"]:
        script(tmp_path, name, inputs='Input(name="raw", type="file")')
    script(tmp_path, "ingest")
    definition(
        tmp_path,
        entry("zebra", 'raw = "ingest.data"')
        + entry("alpha", 'raw = "ingest.data"')
        + entry("ingest"),
    )
    report = validate(tmp_path)
    assert report.status == "verified"
    assert report.order == ["ingest", "alpha", "zebra"]


@pytest.mark.parametrize(
    ("inputs", "binding", "code"),
    [
        ('Input(name="raw", type="file")', "", "binding"),
        ("", 'raw = "ingest.data"', "binding"),
        ('Input(name="raw", type="file")', 'raw = "missing.data"', "reference"),
        ('Input(name="raw", type="file")', 'raw = "ingest.missing"', "reference"),
        ('Input(name="raw", type="file")', 'raw = "ingest.data.extra"', "reference"),
        ('Input(name="raw", type="array")', 'raw = "ingest.data"', "compatibility"),
        ('Input(name="raw", type="file", format="csv")', 'raw = "ingest.data"', "compatibility"),
    ],
)
def test_invalid_wiring(tmp_path: Path, inputs: str, binding: str, code: str) -> None:
    script(tmp_path, "ingest")
    script(tmp_path, "train", inputs=inputs)
    definition(tmp_path, entry("ingest") + entry("train", binding))
    report = validate(tmp_path)
    assert report.status == "invalid"
    assert any(item.status == "invalid" and item.code == code for item in report.checks)


def test_optional_input_and_output_rules(tmp_path: Path) -> None:
    script(
        tmp_path, "ingest", output='Output(name="data", type="file", format="text", required=False)'
    )
    script(tmp_path, "train", inputs='Input(name="raw", type="file")')
    definition(tmp_path, entry("ingest") + entry("train", 'raw = "ingest.data"'))
    assert validate(tmp_path).status == "invalid"
    script(tmp_path, "train", inputs='Input(name="raw", type="file", required=False)')
    assert validate(tmp_path).status == "verified"
    definition(tmp_path, entry("ingest") + entry("train"))
    assert validate(tmp_path).status == "verified"


@pytest.mark.parametrize("self_cycle", [True, False])
def test_cycles_fail(tmp_path: Path, self_cycle: bool) -> None:
    script(tmp_path, "first", inputs='Input(name="raw", type="file")')
    script(tmp_path, "second", inputs='Input(name="raw", type="file")')
    definition(
        tmp_path,
        entry("first", f'raw = "{"first" if self_cycle else "second"}.data"')
        + entry("second", 'raw = "first.data"'),
    )
    report = validate(tmp_path)
    assert report.status == "invalid"
    assert report.order == []
    assert any(item.code == "graph" and item.status == "invalid" for item in report.checks)


@pytest.mark.parametrize(
    "source",
    [
        "step = build_contract()",
        "step = Step(**options)",
        "",
        "step = Step(name=get_name(), outputs=[])\n",
        "step = Step(name='train', outputs=[])\nstep = other\n",
    ],
)
def test_computed_contracts_are_deferred(tmp_path: Path, source: str) -> None:
    (tmp_path / "train.py").write_text(source, encoding="utf-8")
    definition(tmp_path, entry("train"))
    report = validate(tmp_path)
    assert report.status == "deferred"
    assert any(item.status == "deferred" and item.code == "contract" for item in report.checks)


def test_explicit_contract_for_computed_declaration(tmp_path: Path) -> None:
    (tmp_path / "train.py").write_text("step = compute_contract()", encoding="utf-8")
    (tmp_path / "train.toml").write_text(
        'schema_version = 1\nname = "train"\n[[outputs]]\n'
        'name = "data"\ntype = "file"\nformat = "text"\n',
        encoding="utf-8",
    )
    definition(tmp_path, entry("train") + 'contract = "train.toml"\n')
    assert validate(tmp_path).status == "verified"


@pytest.mark.parametrize(
    "path",
    [
        "../escape.py",
        "/absolute.py",
        ".git/config.py",
        ".mlef/run.py",
        "missing.py",
        "train.txt",
        "a\\b.py",
    ],
)
def test_unsafe_or_missing_script(tmp_path: Path, path: str) -> None:
    definition(tmp_path, f'[[steps]]\nname = "train"\nscript = {json.dumps(path)}\n')
    assert validate(tmp_path).status == "invalid"


def test_symlink_source_rejected(tmp_path: Path) -> None:
    target = script(tmp_path, "target")
    try:
        (tmp_path / "train.py").symlink_to(target)
    except OSError:
        pytest.skip("Symlink creation requires platform privileges.")
    definition(tmp_path, entry("train"))
    report = validate(tmp_path)
    assert report.status == "invalid"
    assert "symlink" in report.checks[-2].message


@pytest.mark.parametrize(
    "content",
    [
        "schema_version = 2\nsteps = []",
        "schema_version = true\nsteps = []",
        "schema_version = 1\nsteps = []",
        "not valid TOML",
        "schema_version = 1\nunknown = 1",
        "schema_version = 1\n" + entry("train") * 2,
    ],
)
def test_invalid_definition(tmp_path: Path, content: str) -> None:
    (tmp_path / "mlef.toml").write_text(content, encoding="utf-8")
    assert validate(tmp_path).status == "invalid"


def test_missing_definition_and_syntax_errors(tmp_path: Path) -> None:
    assert validate(tmp_path).status == "invalid"
    definition(tmp_path, entry("train"))
    (tmp_path / "train.py").write_text("this is not valid Python!", encoding="utf-8")
    assert validate(tmp_path).status == "invalid"
    script(tmp_path, "train").write_text(
        'step = Step(name="wrong", outputs=[Output(name="data", type="file", format="text")])',
        encoding="utf-8",
    )
    assert validate(tmp_path).status == "invalid"


def test_deferred_producer_preserves_graph_without_claiming_compatibility(tmp_path: Path) -> None:
    (tmp_path / "ingest.py").write_text("step = factory()", encoding="utf-8")
    script(tmp_path, "train", inputs='Input(name="raw", type="file")')
    definition(tmp_path, entry("train", 'raw = "ingest.data"') + entry("ingest"))
    report = validate(tmp_path)
    assert report.status == "deferred"
    assert report.order == ["ingest", "train"]
    assert any(item.code == "binding" and item.status == "deferred" for item in report.checks)
    definition(tmp_path, entry("train", 'raw = "missing.data"') + entry("ingest"))
    assert validate(tmp_path).status == "invalid"


def test_toml_dates_are_invalid_contract_values_not_serialization_crashes(tmp_path: Path) -> None:
    definition(tmp_path, '[[steps]]\nname = "train"\nscript = 2026-01-01\n')
    assert validate(tmp_path).status == "invalid"
    script(tmp_path, "train")
    (tmp_path / "contract.toml").write_text("name = 2026-01-01\n", encoding="utf-8")
    definition(tmp_path, entry("train") + 'contract = "contract.toml"\n')
    assert validate(tmp_path).status == "invalid"


def test_contract_paths_and_nested_definitions_use_project_root(tmp_path: Path) -> None:
    script(tmp_path, "train")
    (tmp_path / "config").mkdir()
    path = definition(tmp_path, entry("train"))
    path.rename(tmp_path / "config/pipeline.toml")
    assert validate(tmp_path, Path("config/pipeline.toml")).status == "verified"
    assert validate(tmp_path, Path("../outside.toml")).status == "invalid"
    definition(tmp_path, entry("train") + 'contract = "../outside.toml"\n')
    assert validate(tmp_path).status == "invalid"


def test_api_and_cli_json_exit_codes(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    script(project, "train")
    definition(project, entry("train"))
    assert api.validate(root=project).status == "verified"
    assert main(["validate", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "verified"
    assert (
        PipelineDefinition.model_validate_json(json.dumps(payload["definition"])).steps[0].name
        == "train"
    )
    (project / "train.py").write_text("step = factory()", encoding="utf-8")
    assert main(["validate", "--json"]) == 3
    assert json.loads(capsys.readouterr().out)["status"] == "deferred"
    (project / "train.py").unlink()
    assert main(["validate", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "invalid"
    assert not (project / ".mlef").exists()
