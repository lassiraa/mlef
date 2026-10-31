"""Custom identities are inspectable without executing their registered codecs."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from mlef import (
    ArtifactCodec,
    ArtifactManifest,
    ArtifactRegistry,
    ArtifactType,
    CustomType,
    Input,
    Output,
    Step,
    load_artifacts,
    read_step,
)
from mlef.artifacts import ArtifactFile, Column, DatasetMetadata
from mlef.pipeline import validate

PAYLOAD = CustomType(name="example.positive_integer", version=1)
FORMAT = "application/json"


def serialize(value: object, path: Path) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def positive_integer(path: Path) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not int or value <= 0:
        raise ValueError("Expected a positive integer.")


def registry() -> ArtifactRegistry:
    result = ArtifactRegistry()
    result.register(
        ArtifactCodec(type=PAYLOAD, format=FORMAT, serializer=serialize, validator=positive_integer)
    )
    return result


def step() -> Step:
    return Step(name="producer", outputs=[Output(name="data", type=PAYLOAD, format=FORMAT)])


@pytest.mark.parametrize("name", ["file", "Example.value", "example..value", "example/value"])
def test_custom_names_require_lowercase_namespaces(name: str) -> None:
    with pytest.raises(ValidationError):
        CustomType(name=name, version=1)


@pytest.mark.parametrize("version", [True, "1", 0, -1])
def test_custom_versions_are_positive_integers(version: object) -> None:
    with pytest.raises(ValidationError):
        CustomType.model_validate({"name": PAYLOAD.name, "version": version})


def test_custom_manifest_roundtrip_and_versioned_identity() -> None:
    original = ArtifactManifest(
        name="data",
        type=PAYLOAD,
        format=FORMAT,
        files=[ArtifactFile(path="data.json", size_bytes=1, sha256="a" * 64)],
    )
    assert ArtifactManifest.model_validate_json(original.model_dump_json()) == original
    assert original.type == CustomType.model_validate_json(PAYLOAD.model_dump_json())
    changed = ArtifactManifest.model_validate(
        original.model_dump() | {"type": CustomType(name=PAYLOAD.name, version=2)}
    )
    assert changed.content_sha256() != original.content_sha256()
    with pytest.raises(ValidationError, match="disagree"):
        ArtifactManifest(
            name="data",
            type=PAYLOAD,
            format=FORMAT,
            files=original.files,
            metadata=DatasetMetadata(
                rows=1, columns=[Column(name="id", dtype="int")], row_id=["id"]
            ),
        )
    with pytest.raises(ValidationError, match="frozen"):
        PAYLOAD.__setattr__("version", 2)


def test_registry_is_explicit_and_rejects_replacement(tmp_path: Path) -> None:
    codecs = registry()
    with pytest.raises(ValueError, match="already registered"):
        codecs.register(
            ArtifactCodec(
                type=PAYLOAD, format=FORMAT, serializer=serialize, validator=positive_integer
            )
        )
    next_version = CustomType(name=PAYLOAD.name, version=2)
    with pytest.raises(ValueError, match="exact"):
        codecs.validate_payload(next_version, FORMAT, tmp_path / "absent")
    with pytest.raises(ValueError, match="exact"):
        codecs.validate_payload(PAYLOAD, "other-format", tmp_path / "absent")
    with pytest.raises(ValueError, match="exact"):
        ArtifactRegistry().artifact(
            name="data", artifact_type=PAYLOAD, media_format=FORMAT, value=1, path=tmp_path / "data"
        )
    codecs.register(
        ArtifactCodec(
            type=next_version, format=FORMAT, serializer=serialize, validator=positive_integer
        )
    )
    with pytest.raises(ValueError, match="existing"):
        codecs.validate_payload(next_version, FORMAT, tmp_path / "absent")


def test_registered_save_and_verified_handoff(tmp_path: Path) -> None:
    codecs = registry()
    producer = step()
    with producer.local(tmp_path / "producer") as context:
        artifact = codecs.artifact(
            name="data",
            artifact_type=PAYLOAD,
            media_format=FORMAT,
            value=42,
            path=context.output_path("data", "data.json"),
        )
        producer.save([artifact], context=context)
        assert "serializer" not in artifact.model_dump()
    source = load_artifacts(tmp_path / "producer")["data"]
    consumer = Step(
        name="consumer",
        inputs=[Input(name="raw", type=PAYLOAD, format=FORMAT)],
        outputs=[Output(name="data", type=PAYLOAD, format=FORMAT)],
    )
    with consumer.local(tmp_path / "consumer", inputs={"raw": source}) as context:
        copied = context.input_path("raw", "data.json")
        codecs.validate_payload(PAYLOAD, FORMAT, copied)
        consumer.save(
            [
                codecs.artifact(
                    name="data",
                    artifact_type=PAYLOAD,
                    media_format=FORMAT,
                    value=json.loads(copied.read_text()),
                    path=context.output_path("data", "data.json"),
                )
            ],
            context=context,
        )
    assert (
        read_step(tmp_path / "consumer").inputs["raw"]
        == read_step(source.directory).artifacts["data"].producer
    )
    other = CustomType(name=PAYLOAD.name, version=2)
    incompatible = Step(
        name="consumer", inputs=[Input(name="raw", type=other)], outputs=consumer.outputs
    )
    with pytest.raises(ValueError, match="Incompatible"):
        incompatible.local(tmp_path / "incompatible", inputs={"raw": source}).prepare()
    (source.directory / "outputs/data/data.json").write_text("99")
    with pytest.raises(ValueError, match="changed"):
        read_step(source.directory)


@pytest.mark.parametrize("failure", ["serializer", "validator"])
def test_codec_failure_never_publishes_partial_outputs(tmp_path: Path, failure: str) -> None:
    def broken(value: object, path: Path) -> None:
        path.write_text("partial")
        raise RuntimeError("Serialization failed.")

    codecs = ArtifactRegistry()
    codecs.register(
        ArtifactCodec(
            type=PAYLOAD,
            format=FORMAT,
            serializer=broken if failure == "serializer" else serialize,
            validator=positive_integer,
        )
    )
    producer = step()
    directory = tmp_path / "failed"
    with pytest.raises((RuntimeError, ValueError)), producer.local(directory) as context:
        producer.save(
            [
                codecs.artifact(
                    name="data",
                    artifact_type=PAYLOAD,
                    media_format=FORMAT,
                    value=-1,
                    path=context.output_path("data", "data.json"),
                )
            ],
            context=context,
        )
    assert json.loads((directory / "step.json").read_text())["state"] == "failed"
    assert (directory / "outputs/data/data.json").exists()
    with pytest.raises(ValueError, match="incomplete"):
        read_step(directory)


def test_registry_free_inspection_and_explicit_semantic_validation(tmp_path: Path) -> None:
    producer = step()
    codecs = registry()
    with producer.local(tmp_path / "producer") as context:
        producer.save(
            [
                codecs.artifact(
                    name="data",
                    artifact_type=PAYLOAD,
                    media_format=FORMAT,
                    value=1,
                    path=context.output_path("data", "data.json"),
                )
            ],
            context=context,
        )

    def fail(path: Path) -> None:
        pytest.fail("Inspection executed custom code.")

    observer = ArtifactRegistry()
    observer.register(
        ArtifactCodec(type=PAYLOAD, format=FORMAT, serializer=serialize, validator=fail)
    )
    assert read_step(tmp_path / "producer").artifacts["data"].type == PAYLOAD
    assert load_artifacts(tmp_path / "producer")["data"].name == "data"


@pytest.mark.parametrize("version", [1, 2])
def test_static_custom_bindings_compare_versions_without_importing_code(
    tmp_path: Path, version: int
) -> None:
    for name in ("producer", "consumer"):
        inputs = (
            f'Input(name="raw", type=CustomType(name="{PAYLOAD.name}", version={version}))'
            if name == "consumer"
            else ""
        )
        (tmp_path / f"{name}.py").write_text(
            'raise RuntimeError("must not execute")\n'
            f'step = Step(name="{name}", inputs=[{inputs}], outputs=['
            f'Output(name="data", type=CustomType(name="{PAYLOAD.name}", version=1), '
            f'format="{FORMAT}")])\n'
        )
    (tmp_path / "mlef.toml").write_text(
        '[[steps]]\nname="producer"\nscript="producer.py"\n'
        '[[steps]]\nname="consumer"\nscript="consumer.py"\nbindings={raw="producer.data"}\n'
    )
    report = validate(tmp_path)
    assert report.status == ("verified" if version == 1 else "invalid")
    assert report.contracts["producer"].outputs[0].type == PAYLOAD


def test_builtin_identities_do_not_depend_on_registration() -> None:
    manifest = ArtifactManifest(
        name="data",
        type=ArtifactType.FILE,
        format="text/plain",
        files=[ArtifactFile(path="data", size_bytes=1, sha256="a" * 64)],
    )
    assert json.loads(manifest.model_dump_json())["type"] == "file"
    assert "CustomType" not in manifest.model_dump_json()
