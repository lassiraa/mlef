"""Kernel-held ownership locks; lock files remain, process death releases ownership."""

import os
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path

from .storage import QueryError


def native_lock(descriptor: int, release: bool) -> None:
    if os.name == "nt":
        import msvcrt

        assert hasattr(msvcrt, "locking")
        assert hasattr(msvcrt, "LK_NBLCK")
        assert hasattr(msvcrt, "LK_UNLCK")
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK if release else msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        assert hasattr(fcntl, "flock")
        assert hasattr(fcntl, "LOCK_EX")
        assert hasattr(fcntl, "LOCK_NB")
        assert hasattr(fcntl, "LOCK_UN")
        fcntl.flock(descriptor, fcntl.LOCK_UN if release else fcntl.LOCK_EX | fcntl.LOCK_NB)


@contextmanager
def file_lock(path: Path) -> Generator[None, None, None]:
    if path.is_symlink():
        raise QueryError("unsafe_lock", "Ownership lock must not be a symbolic link.")
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            native_lock(stream.fileno(), False)
        except OSError as error:
            raise QueryError(
                "run_active", "Run has an active runner, supervisor or recovery owner."
            ) from error
        try:
            yield
        finally:
            stream.seek(0)
            native_lock(stream.fileno(), True)


def run_lock(run_dir: Path) -> AbstractContextManager[None]:
    return file_lock(run_dir / "active.lock")


def supervisor_lock(root: Path, identifier: str) -> AbstractContextManager[None]:
    requests = root / ".mlef/requests"
    requests.mkdir(exist_ok=True)
    return file_lock(requests / f"{identifier}.lock")
