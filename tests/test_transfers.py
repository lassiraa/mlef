import importlib
import io
import json
import os
import shutil
from collections.abc import Generator
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4

import pytest
from typing_extensions import override

from mlef import api
from mlef.cli import main
from mlef.locking import run_lock
from mlef.project import initialize
from mlef.records import RunRecord
from mlef.runner import now, save
from mlef.storage import QueryError
from mlef.transfers import BlobStore, LocalStore, RunTransfer, S3Store, fetch, publish


@pytest.fixture
def saved_run(project: Path) -> str:
    initialize(project, scaffold=False)
    identifier = str(uuid4())
    directory = project / ".mlef/runs" / identifier
    (directory / "artifacts/empty").mkdir(parents=True)
    (directory / "artifacts/result.bin").write_bytes(b"\x00saved result\xff")
    (directory / "stderr.log").write_text("diagnostic evidence")
    (directory / "Z.txt").write_text("uppercase sorts before lowercase on every host")
    (directory / "a.txt").write_text("lowercase")
    save(
        directory / "run.json",
        RunRecord(
            id=identifier,
            state="succeeded",
            created_at=now(),
            finished_at=now(),
            execution_exit_code=0,
            command=["python", "train.py"],
        ),
    )
    return identifier


@pytest.fixture(params=["local", "s3"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Generator[BlobStore]:
    if request.param == "local":
        yield LocalStore(tmp_path / "store")
    else:
        with importlib.import_module("moto").mock_aws():
            client = importlib.import_module("boto3").client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="mlef-transfers")
            yield S3Store("s3://mlef-transfers/experiments", client=client)


def test_roundtrip_is_verified_idempotent_and_self_contained(
    project: Path, saved_run: str, store: BlobStore, tmp_path: Path
) -> None:
    original = project / ".mlef/runs" / saved_run
    manifest = publish(project, saved_run, store)
    assert publish(project, saved_run, store) == manifest
    assert not any("active.lock" in item.path for item in manifest.files)
    expected = (original / "run.json").read_bytes()
    shutil.rmtree(original)
    assert fetch(project, saved_run, store) == manifest
    assert (original / "run.json").read_bytes() == expected
    assert (original / "artifacts/result.bin").read_bytes() == b"\x00saved result\xff"
    assert (original / "artifacts/empty").is_dir()
    assert api.status(saved_run, root=project).state == "succeeded"
    with pytest.raises(ValueError, match="overwrites"):
        fetch(project, saved_run, store)
    (original / "stderr.log").write_text("changed")
    with pytest.raises(ValueError, match="immutable"):
        publish(project, saved_run, store)


class InterruptedStore:
    def __init__(self, wrapped: BlobStore) -> None:
        self.wrapped = wrapped
        self.remaining = 1

    def read(self, key: str, destination: Path, limit: int) -> bool:
        return self.wrapped.read(key, destination, limit)

    def put(self, key: str, source: Path) -> None:
        if self.remaining == 0:
            raise OSError("interrupted upload")
        self.remaining -= 1
        self.wrapped.put(key, source)


def test_interrupted_upload_is_unpublished_and_retryable(
    project: Path, saved_run: str, store: BlobStore, tmp_path: Path
) -> None:
    interrupted = InterruptedStore(store)
    with pytest.raises(OSError, match="interrupted"):
        publish(project, saved_run, interrupted)
    with pytest.raises(ValueError, match="not published"):
        fetch(tmp_path / "fresh", saved_run, store)
    manifest = publish(project, saved_run, store)
    assert fetch(tmp_path / "fresh", saved_run, store) == manifest


@pytest.mark.parametrize("damage", ["corrupt", "missing", "oversized", "manifest"])
def test_fetch_never_exposes_partial_or_corrupt_runs(
    project: Path, saved_run: str, tmp_path: Path, damage: str
) -> None:
    remote = tmp_path / "store"
    store = LocalStore(remote)
    publish(project, saved_run, store)
    manifest = remote / "runs" / saved_run / "manifest.json"
    if damage == "manifest":
        value = json.loads(manifest.read_bytes())
        value["files"][0]["path"] = "../../escaped"
        manifest.write_text(json.dumps(value))
    else:
        payload = next((remote / "runs" / saved_run / "snapshots").glob("*/artifacts/result.bin"))
        if damage == "missing":
            payload.unlink()
        else:
            payload.write_bytes(b"x" * (100 if damage == "oversized" else 2))
    destination = tmp_path / "fresh"
    with pytest.raises(ValueError, match=r"relative|payload|size limit"):
        fetch(destination, saved_run, store)
    assert not (destination / ".mlef/runs" / saved_run).exists()
    assert not (tmp_path / "escaped").exists()


def test_publish_requires_terminal_cleanup_complete_unowned_run(
    project: Path, saved_run: str, tmp_path: Path
) -> None:
    store = LocalStore(tmp_path / "store")
    directory = project / ".mlef/runs" / saved_run
    with run_lock(directory), pytest.raises(QueryError, match="active"):
        publish(project, saved_run, store)
    path = directory / "run.json"
    value = json.loads(path.read_bytes())
    value["state"] = "running"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="terminal"):
        publish(project, saved_run, store)
    value["state"] = "failed"
    value["runtime"] = {
        "token": "test",
        "project": str(project),
        "container_name": "owned",
        "probe_name": "probe",
        "cleanup_pending": True,
    }
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="cleanup"):
        publish(project, saved_run, store)
    assert not (tmp_path / "store").exists()


@pytest.mark.parametrize("name", ["../x", "AUX.txt", "a/../b", "a.", "x\\y", "/abs", "x:y"])
def test_manifest_rejects_nonportable_or_escaping_paths(saved_run: str, name: str) -> None:
    with pytest.raises(ValueError, match="paths"):
        RunTransfer.model_validate(
            {"run_id": saved_run, "files": [{"path": name, "sha256": "a" * 64, "size_bytes": 0}]}
        )


def test_manifest_rejects_case_collisions_and_file_directory_conflicts(saved_run: str) -> None:
    for names in [["run.json", "RUN.JSON"], ["run.json", "run.json/child"]]:
        with pytest.raises(ValueError, match=r"unique|disjoint"):
            RunTransfer.model_validate(
                {
                    "run_id": saved_run,
                    "files": [
                        {"path": name, "sha256": "a" * 64, "size_bytes": 0} for name in names
                    ],
                }
            )


def test_cli_and_api_use_same_transfer_services(
    project: Path, saved_run: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    capsys.readouterr()
    destination = str(tmp_path / "store")
    assert main(["publish", saved_run, "--to", destination, "--json"]) == 0
    manifest = RunTransfer.model_validate_json(capsys.readouterr().out)
    assert api.publish_run(saved_run, destination, root=project) == manifest
    shutil.rmtree(project / ".mlef/runs" / saved_run)
    assert main(["fetch", saved_run, "--from", destination, "--json"]) == 0
    assert RunTransfer.model_validate_json(capsys.readouterr().out) == manifest
    with pytest.raises(api.MlefError, match="overwrites"):
        api.fetch_run(saved_run, destination, root=project)


class MissingObjectError(Exception):
    response: ClassVar = {"Error": {"Code": "NoSuchKey"}}


class DeniedError(Exception):
    response: ClassVar = {"Error": {"Code": "AccessDenied"}}


class Client:
    def __init__(self, *, missing: bool = False) -> None:
        self.missing = missing

    def get_object(self, **kwargs: object) -> dict[str, Any]:
        raise MissingObjectError if self.missing else DeniedError

    def put_object(self, **kwargs: object) -> dict[str, Any]:
        raise DeniedError


def test_s3_permission_errors_are_not_missing_objects(tmp_path: Path) -> None:
    (tmp_path / "payload").write_bytes(b"payload")
    store = S3Store("s3://bucket/prefix", client=Client())
    with pytest.raises(QueryError, match="AccessDenied"):
        store.read("key", tmp_path / "payload", 10)
    with pytest.raises(QueryError, match="AccessDenied"):
        store.put("key", tmp_path / "payload")
    missing = S3Store("s3://bucket", client=Client(missing=True))
    assert not missing.read("key", tmp_path / "payload", 10)


def test_s3_stream_is_closed_on_oversize(tmp_path: Path) -> None:
    body = io.BytesIO(b"too long")

    class StreamingClient(Client):
        @override
        def get_object(self, **kwargs: object) -> dict[str, Any]:
            return {"Body": body, "ETag": "not-a-sha256"}

    with pytest.raises(ValueError, match="size limit"):
        S3Store("s3://bucket", client=StreamingClient()).read("key", tmp_path / "payload", 1)
    assert body.closed


def test_conditional_writes_never_replace_existing_objects(
    store: BlobStore, tmp_path: Path
) -> None:
    payload = tmp_path / "payload"
    payload.write_bytes(b"first")
    store.put("immutable", payload)
    payload.write_bytes(b"later")
    store.put("immutable", payload)
    assert store.read("immutable", payload, 5)
    assert payload.read_bytes() == b"first"


def test_streaming_network_errors_are_typed_and_close_the_response(tmp_path: Path) -> None:
    class BrokenBody(io.BytesIO):
        @override
        def read(self, size: int | None = -1) -> bytes:
            raise RuntimeError("private transport details")

    body = BrokenBody()

    class StreamingClient(Client):
        @override
        def get_object(self, **kwargs: object) -> dict[str, Any]:
            return {"Body": body}

    with pytest.raises(QueryError, match="payload read failed") as caught:
        S3Store("s3://bucket", client=StreamingClient()).read("key", tmp_path / "payload", 10)
    assert "private" not in str(caught.value)
    assert body.closed


def test_download_interrupt_can_retry_without_exposing_a_run(
    project: Path, saved_run: str, store: BlobStore, tmp_path: Path
) -> None:
    manifest = publish(project, saved_run, store)

    class InterruptedDownload(InterruptedStore):
        @override
        def read(self, key: str, destination: Path, limit: int) -> bool:
            if "/snapshots/" in key:
                destination.write_bytes(b"partial")
                raise OSError("interrupted download")
            return self.wrapped.read(key, destination, limit)

    destination = tmp_path / "fresh"
    with pytest.raises(OSError, match="interrupted download"):
        fetch(destination, saved_run, InterruptedDownload(store))
    assert not (destination / ".mlef/runs" / saved_run).exists()
    assert fetch(destination, saved_run, store) == manifest


def test_hardlinks_cannot_escape_the_published_inventory(
    project: Path, saved_run: str, tmp_path: Path
) -> None:
    external = tmp_path / "private"
    external.write_text("do not upload")
    os.link(external, project / ".mlef/runs" / saved_run / "artifacts/linked")
    with pytest.raises(ValueError, match="hard links"):
        publish(project, saved_run, LocalStore(tmp_path / "store"))
    assert not (tmp_path / "store").exists()


def test_local_store_cannot_overlap_source_run(project: Path, saved_run: str) -> None:
    for destination in [project / ".mlef", project / ".mlef/runs" / saved_run / "remote"]:
        with pytest.raises(ValueError, match="overlap"):
            publish(project, saved_run, LocalStore(destination))


@pytest.mark.skipif(not bool(os.environ.get("MLEF_TEST_S3_URI")), reason="Set MLEF_TEST_S3_URI")
def test_live_s3_roundtrip(project: Path, saved_run: str, tmp_path: Path) -> None:
    uri = os.environ["MLEF_TEST_S3_URI"].rstrip("/") + "/mlef-tests/" + str(uuid4())
    store = S3Store(uri)
    manifest = publish(project, saved_run, store)
    assert fetch(tmp_path / "fresh", saved_run, store) == manifest
