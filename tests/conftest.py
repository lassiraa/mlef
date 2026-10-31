import json
import subprocess
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path

import pytest

from mlef import api
from mlef.environment import ImageIdentity


@pytest.fixture
def docker_images(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "mlef.runner.image_identity",
        lambda image, log: ImageIdentity(
            id="sha256:test", digests=["python@sha256:test"], os="linux", architecture="amd64"
        ),
    )


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "mlef test project"
    root.mkdir()
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    monkeypatch.chdir(root)
    return root


@pytest.fixture
def cli() -> Callable[..., subprocess.CompletedProcess[str]]:
    def invoke(*arguments: str, expected: int = 0) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [sys.executable, "-m", "mlef", *arguments],
            capture_output=True,
            text=True,
            timeout=240,
        )
        assert result.returncode == expected, result.stdout + result.stderr
        return result

    return invoke


@pytest.fixture
def evaluation_project(project: Path) -> Callable[[str], tuple[Path, list[str]]]:
    api.init(scaffold=False)
    for name in ("pyproject.toml", "uv.lock"):
        (project / name).write_bytes(
            (Path(__file__).resolve().parents[1] / "examples/pipeline" / name).read_bytes()
        )
    (project / "mlef.toml").write_text(
        '[[steps]]\nname="data"\nscript="data.py"\n'
        '[[steps]]\nname="train"\nscript="train.py"\nbindings={data="data.data"}\n'
    )
    scripts = {
        "data.py": """
            from mlef import Artifact, ArtifactType, Output, Step
            from mlef.artifacts import Column, DatasetMetadata
            step = Step(name="data", outputs=[Output(
                name="data", type=ArtifactType.DATASET, format="application/json")])
            with step.context() as context:
                path = context.output_path("data", "data.json")
                path.write_text('{"id":1,"value":2}')
                step.save([Artifact(name="data", type=ArtifactType.DATASET,
                    format="application/json", path=path,
                    metadata=DatasetMetadata(rows=1,
                        columns=[Column(name="id", dtype="int")], row_id=["id"]))],
                    context=context)
        """,
        "train.py": """
            import json
            from mlef import Artifact, ArtifactType, Input, Output, Step
            from mlef.artifacts import ModelMetadata
            step = Step(schema_version=2, kind="train", name="train",
                inputs=[Input(name="data", type=ArtifactType.DATASET)],
                outputs=[Output(name="baseline", type=ArtifactType.MODEL,
                             format="application/json"),
                         Output(name="candidate", type=ArtifactType.MODEL,
                             format="application/json")])
            with step.context() as context:
                artifacts = []
                for name, value in [("baseline", 1), ("candidate", 2)]:
                    path = context.output_path(name, "model.json")
                    path.write_text(json.dumps({"prediction":value}))
                    artifacts.append(Artifact(name=name, type=ArtifactType.MODEL,
                        format="application/json", path=path,
                        metadata=ModelMetadata(framework="constant", serialization="json",
                            training_dataset=context.input_reference("data"))))
                step.save(artifacts, context=context)
        """,
        "evaluate.py": """
            import json
            from mlef import Artifact, ArtifactType, Input, Output, Step, Metrics, Scalar
            from mlef.metrics import Series
            step = Step(schema_version=2, kind="evaluate", name="evaluate",
                inputs=[Input(name="model", type=ArtifactType.MODEL),
                        Input(name="aux", type=ArtifactType.MODEL, required=False),
                        Input(name="data", type=ArtifactType.DATASET)],
                outputs=[Output(name="metrics", type=ArtifactType.METRICS,
                                format="application/json"),
                         Output(name="extra", type=ArtifactType.METRICS,
                                format="application/json", required=False)])
            with step.context() as context:
                prediction = json.loads(
                    context.input_path("model", "model.json").read_text())["prediction"]
                if context.parameters.get("fail_baseline", False) and prediction == 1:
                    raise RuntimeError("Deliberate baseline failure")
                value = json.loads(context.input_path("data", "data.json").read_text())["value"]
                model_input = "aux" if context.parameters.get("use_aux", False) else "model"
                metrics = Metrics(model=context.input_reference(model_input),
                    dataset=context.input_reference("data"), scope="test",
                    values={"mae":Scalar(definition="mae",
                                         direction=context.parameters.get("direction", "minimize"),
                                         target=context.parameters.get("target"),
                                         value=abs(value-prediction), sample_count=1),
                            "loss":Series(definition="loss", index="epoch", x=[1], y=[0.5])})
                path = context.output_path("metrics", "metrics.json")
                path.write_text(metrics.model_dump_json())
                artifacts = [Artifact(name="metrics", type=ArtifactType.METRICS,
                    format="application/json", path=path)]
                if context.parameters.get("duplicate", False):
                    extra = context.output_path("extra", "metrics.json")
                    extra.write_text(metrics.model_copy(update={"scope":"validation"})
                                     .model_dump_json())
                    artifacts.append(Artifact(name="extra", type=ArtifactType.METRICS,
                        format="application/json", path=extra))
                step.save(artifacts, context=context)
        """,
    }
    for name, script in scripts.items():
        (project / name).write_text(textwrap.dedent(script), encoding="utf-8")

    def configure(identifier: str) -> tuple[Path, list[str]]:
        from mlef.execution import attempt_directory
        from mlef.sdk import read_step

        directory = project / ".mlef/runs" / identifier / "artifacts/pipeline"
        reference = read_step(attempt_directory(directory, "data")).artifacts["data"].producer
        assert reference is not None
        spec = project / "evaluation.toml"
        lines = [
            'script="evaluate.py"',
            "[bindings.data]",
            f'run_id="{identifier}"',
            "[bindings.data.artifact]",
        ]
        lines.extend(f"{key}={json.dumps(value)}" for key, value in reference.model_dump().items())
        spec.write_text("\n".join(lines))
        versions = [
            api.register_model(identifier, "train", output, name="constant")
            for output in ("baseline", "candidate")
        ]
        return spec.relative_to(project), [
            f"{version.name}@{version.version}" for version in versions
        ]

    return configure
