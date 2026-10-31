"""Explicit local/S3 run transfers with a final immutable publication manifest."""

import hashlib
import importlib
import json
import os
import shutil
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any, BinaryIO, Literal, Protocol, Self
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator

from . import storage
from .artifacts import ArtifactFile, safe_path
from .environment import file_hash
from .locking import file_lock, run_lock
from .records import Contract, read_json, read_run, run_id
from .sdk import _checked

MAX_MANIFEST_BYTES = 16 * 1024 * 1024


def portable_path(value: str) -> str:
    safe_path(value)
    reserved = {"con", "prn", "aux", "nul"} | {
        prefix + str(index) for prefix in ("com", "lpt") for index in range(1, 10)
    }
    if any(
        part.rstrip(" .") != part
        or part.split(".")[0].lower() in reserved
        or any(character in '<>"|?*' or ord(character) < 32 for character in part)
        for part in value.split("/")
    ):
        raise ValueError("Run transfer paths must be portable relative paths.")
    return value


class RunTransfer(Contract):
    schema_version: Literal[1] = 1
    run_id: str
    files: list[ArtifactFile] = Field(min_length=1, max_length=100_000)
    directories: list[str] = Field(default_factory=list, max_length=100_000)

    _run_id = field_validator("run_id")(run_id)

    @model_validator(mode="after")
    def inventory(self) -> Self:
        names = [item.path for item in self.files] + self.directories
        for name in names:
            portable_path(name)
            if "active.lock" in name.split("/"):
                raise ValueError("Ownership locks cannot be transferred.")
        if len({name.casefold() for name in names}) != len(names):
            raise ValueError("Transfer paths must be unique, including case-insensitive names.")
        files = {item.path.casefold() for item in self.files}
        if "run.json" not in {item.path for item in self.files} or any(
            parent.as_posix().casefold() in files
            for name in names
            for parent in Path(name).parents
            if parent.as_posix() != "."
        ):
            raise ValueError("Transfer requires run.json and disjoint file/directory paths.")
        return self

    def canonical(self) -> bytes:
        values = self.model_dump(mode="json")
        values["files"] = sorted(values["files"], key=lambda item: item["path"])
        values["directories"] = sorted(self.directories)
        return json.dumps(values, sort_keys=True, separators=(",", ":")).encode()


class BlobStore(Protocol):
    def read(self, key: str, destination: Path, limit: int) -> bool: ...

    def put(self, key: str, source: Path) -> None: ...


def _copy(source: BinaryIO, destination: Path, limit: int) -> None:
    with destination.open("wb") as target:
        remaining = limit
        while chunk := source.read(min(1024 * 1024, remaining + 1)):
            if len(chunk) > remaining:
                raise ValueError("Remote object exceeds its declared size limit.")
            target.write(chunk)
            remaining -= len(chunk)


class LocalStore:
    def __init__(self, root: Path) -> None:
        self.root = _checked(root.absolute())

    def read(self, key: str, destination: Path, limit: int) -> bool:
        path = _checked(self.root / safe_path(key))
        if not path.exists():
            return False
        with path.open("rb") as stream:
            _copy(stream, destination, limit)
        return True

    def put(self, key: str, source: Path) -> None:
        path = _checked(self.root / safe_path(key))
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            shutil.copyfile(source, temporary)
            with suppress(FileExistsError):
                os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


class S3Client(Protocol):
    def get_object(self, **kwargs: object) -> dict[str, Any]: ...

    def put_object(self, **kwargs: object) -> dict[str, Any]: ...


class S3Store:
    def __init__(self, uri: str, *, client: S3Client | None = None) -> None:
        parsed = urlsplit(uri)
        if (
            parsed.scheme != "s3"
            or not parsed.netloc
            or parsed.netloc != parsed.hostname
            or parsed.query
            or parsed.fragment
            or ":" in parsed.netloc
        ):
            raise ValueError("Expected s3://bucket/optional-prefix without credentials or query.")
        self.bucket = parsed.netloc
        self.prefix = parsed.path.strip("/")
        if self.prefix:
            safe_path(self.prefix)
        if client is None:
            try:
                client = importlib.import_module("boto3").client("s3")
            except ImportError as error:
                raise ValueError("S3 transfers require the optional mlef[s3] extra.") from error
            except Exception as error:
                raise storage.QueryError(
                    "remote_storage", "Cannot initialize S3 client."
                ) from error
        self.client = client

    def _key(self, key: str) -> str:
        return "/".join(part for part in (self.prefix, safe_path(key)) if part)

    def read(self, key: str, destination: Path, limit: int) -> bool:
        try:
            result = self.client.get_object(Bucket=self.bucket, Key=self._key(key))
        except Exception as error:
            code = getattr(error, "response", {}).get("Error", {}).get("Code", "unknown")
            if code in {"NoSuchKey", "404"}:
                return False
            raise storage.QueryError("remote_storage", f"S3 read failed ({code}).") from error
        body = result["Body"]
        try:
            _copy(body, destination, limit)
        except (OSError, ValueError):
            raise
        except Exception as error:
            raise storage.QueryError(
                "remote_storage", "S3 payload read failed; retry the transfer."
            ) from error
        finally:
            body.close()
        return True

    def put(self, key: str, source: Path) -> None:
        try:
            with source.open("rb") as stream:
                self.client.put_object(
                    Bucket=self.bucket, Key=self._key(key), Body=stream, IfNoneMatch="*"
                )
        except Exception as error:
            code = getattr(error, "response", {}).get("Error", {}).get("Code", "unknown")
            if code != "PreconditionFailed":
                raise storage.QueryError("remote_storage", f"S3 write failed ({code}).") from error


def backend(location: str) -> BlobStore:
    if location.startswith("s3://"):
        return S3Store(location)
    if "://" in location:
        raise ValueError("Run stores require a local directory or s3:// URI.")
    return LocalStore(Path(location))


def _terminal(directory: Path, identifier: str) -> None:
    record = read_run(_checked(directory / "run.json"))
    if record.id != identifier or record.state not in {"succeeded", "failed", "cancelled"}:
        raise ValueError("Run transfers require a terminal run with matching identity.")
    if record.runtime is not None and record.runtime.cleanup_pending:
        raise ValueError("Run transfers require completed resource cleanup.")


def _inventory(directory: Path, identifier: str) -> RunTransfer:
    files, directories = [], []
    for path in sorted(
        directory.rglob("*"), key=lambda item: item.relative_to(directory).as_posix()
    ):
        _checked(path)
        name = path.relative_to(directory).as_posix()
        if path.name == "active.lock":
            continue
        if path.is_dir():
            directories.append(name)
        else:
            files.append(
                ArtifactFile(path=name, size_bytes=path.stat().st_size, sha256=file_hash(path))
            )
    return RunTransfer(run_id=identifier, files=files, directories=directories)


def _verified(store: BlobStore, key: str, source: Path, scratch: Path) -> None:
    size, digest = source.stat().st_size, file_hash(source)
    if not store.read(key, scratch, size):
        store.put(key, source)
        if not store.read(key, scratch, size):
            raise ValueError("Remote publication disappeared before verification.")
    if scratch.stat().st_size != size or file_hash(scratch) != digest:
        raise ValueError("Immutable remote object conflicts with local bytes.")


def _workspace(root: Path) -> Path:
    path = _checked(root.absolute() / ".mlef/transfers")
    path.mkdir(parents=True, exist_ok=True)
    return path


def publish(root: Path, identifier: str, store: BlobStore) -> RunTransfer:
    source = _checked(storage.directory(root, identifier).absolute())
    if isinstance(store, LocalStore):
        namespace = store.root / "runs" / identifier
        if namespace.is_relative_to(source) or source.is_relative_to(namespace):
            raise ValueError("A run store cannot overlap its source run.")
    with run_lock(source), tempfile.TemporaryDirectory(dir=_workspace(root)) as temporary:
        _terminal(source, identifier)
        manifest = _inventory(source, identifier)
        staging = Path(temporary)
        snapshot = staging / "snapshot"
        snapshot.mkdir()
        for name in manifest.directories:
            (snapshot / name).mkdir(parents=True, exist_ok=True)
        for item in manifest.files:
            target = snapshot / item.path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_checked(source / item.path), target)
        if (
            _inventory(snapshot, identifier) != manifest
            or _inventory(source, identifier) != manifest
        ):
            raise ValueError("Run bytes changed while staging publication.")
        _terminal(snapshot, identifier)
        content = manifest.canonical()
        if len(content) > MAX_MANIFEST_BYTES:
            raise ValueError("Run publication manifest exceeds 16 MiB.")
        identity = hashlib.sha256(content).hexdigest()
        prefix = f"runs/{identifier}/snapshots/{identity}"
        scratch = staging / "verification"
        if (
            store.read(f"runs/{identifier}/manifest.json", scratch, MAX_MANIFEST_BYTES)
            and scratch.read_bytes() != content
        ):
            raise ValueError("Published run identity is immutable; local run bytes changed.")
        for item in manifest.files:
            _verified(store, f"{prefix}/{item.path}", snapshot / item.path, scratch)
        path = staging / "manifest.json"
        path.write_bytes(content)
        _verified(store, f"runs/{identifier}/manifest.json", path, scratch)
        return manifest


def fetch(root: Path, identifier: str, store: BlobStore) -> RunTransfer:
    run_id(identifier)
    root = root.absolute()
    workspace = _workspace(root)
    target = _checked(root / ".mlef/runs" / identifier)
    target.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(workspace / f"{identifier}.lock"):
        if target.exists():
            raise ValueError("Fetch never overwrites an existing local run.")
        with tempfile.TemporaryDirectory(dir=workspace) as temporary:
            staging = Path(temporary)
            path = staging / "manifest.json"
            if not store.read(f"runs/{identifier}/manifest.json", path, MAX_MANIFEST_BYTES):
                raise ValueError("Run is not published; no committed manifest exists.")
            manifest = RunTransfer.model_validate(read_json(path))
            if manifest.run_id != identifier:
                raise ValueError("Remote manifest identifies a different run.")
            identity = hashlib.sha256(manifest.canonical()).hexdigest()
            prefix = f"runs/{identifier}/snapshots/{identity}"
            snapshot = staging / "snapshot"
            snapshot.mkdir()
            for name in manifest.directories:
                (snapshot / name).mkdir(parents=True, exist_ok=True)
            for item in manifest.files:
                payload = snapshot / item.path
                payload.parent.mkdir(parents=True, exist_ok=True)
                if (
                    not store.read(f"{prefix}/{item.path}", payload, item.size_bytes)
                    or payload.stat().st_size != item.size_bytes
                    or file_hash(payload) != item.sha256
                ):
                    raise ValueError("Remote payload is missing, truncated or has changed bytes.")
            _terminal(snapshot, identifier)
            if _inventory(snapshot, identifier) != manifest:
                raise ValueError("Downloaded run inventory differs from the publication manifest.")
            snapshot.rename(target)
            return manifest
