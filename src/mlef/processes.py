"""Drain subprocess pipes through enforced byte budgets instead of direct file descriptors."""

import contextlib
import io
import subprocess
import threading
import time
from typing import BinaryIO

from typing_extensions import Buffer, override


class LogLimitError(RuntimeError):
    pass


class Budget:
    def __init__(self, limit: int | None) -> None:
        if limit is not None and (type(limit) is not int or limit <= 0):
            raise ValueError("Log budgets must be positive integer byte counts.")
        self.limit = limit
        self.used = 0
        self.lock = threading.Lock()
        self.exceeded = threading.Event()

    def take(self, chunk: bytes) -> bytes:
        with self.lock:
            allowed = (
                len(chunk)
                if self.limit is None
                else max(0, min(len(chunk), self.limit - self.used))
            )
            self.used += allowed
            if allowed < len(chunk):
                self.exceeded.set()
            return chunk[:allowed]


class BoundedLog(io.BufferedWriter):
    def __init__(self, path: str, limit: int | None) -> None:
        self.budget = Budget(limit)
        super().__init__(io.FileIO(path, "ab"))

    @override
    def write(self, value: Buffer, /) -> int:
        allowed = self.budget.take(bytes(value))
        written = super().write(allowed)
        super().flush()
        if self.budget.exceeded.is_set():
            raise LogLimitError("Runner log byte budget exceeded.")
        return written


def command(
    arguments: list[str], log: BinaryIO | None, *, capture: bool = False, timeout: float = 30
) -> subprocess.CompletedProcess[bytes]:
    process = subprocess.Popen(
        arguments,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )
    output = bytearray()
    errors: list[OSError | LogLimitError] = []
    captured = Budget(4 * 1024 * 1024)

    def drain(source: io.BufferedReader, collect: bool) -> None:
        try:
            while chunk := source.read1(65536):
                if collect:
                    output.extend(captured.take(chunk))
                    if captured.exceeded.is_set():
                        raise LogLimitError("Captured command output byte budget exceeded.")
                elif log is not None:
                    log.write(chunk)
                    log.flush()
        except (OSError, LogLimitError) as error:
            errors.append(error)
        finally:
            source.close()

    assert isinstance(process.stdout, io.BufferedReader)
    assert isinstance(process.stderr, io.BufferedReader)
    threads = [
        threading.Thread(target=drain, args=(process.stdout, capture), daemon=True),
        threading.Thread(target=drain, args=(process.stderr, False), daemon=True),
    ]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + timeout
    try:
        while process.poll() is None:
            if errors:
                raise errors[0]
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(arguments, timeout)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=0.05)
    except BaseException:
        if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
            from .supervision import kill_tree

            kill_tree(process)
        else:
            process.kill()
            process.wait(timeout=5)
        raise
    finally:
        for thread in threads:
            thread.join(timeout=5)
    if errors:
        raise errors[0]
    return subprocess.CompletedProcess(arguments, process.returncode, bytes(output))
