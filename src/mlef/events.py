"""Bounded worker output delivery for synchronous Python callers."""

import io
import subprocess
import threading
from collections.abc import Callable
from typing import Literal

from pydantic import ConfigDict

from .processes import Budget
from .records import Contract


class LogEvent(Contract):
    model_config = ConfigDict(ser_json_bytes="base64")
    stream: Literal["stdout", "stderr"]
    data: bytes


class Delivery:
    def __init__(self, callback: Callable[[LogEvent], None] | None, limit: int) -> None:
        self.callback = callback
        self.budget = Budget(limit)
        self.errors: list[BaseException] = []
        self.threads: list[threading.Thread] = []
        self.lock = threading.Lock()
        self.diagnostics = bytearray()

    def start(self, process: subprocess.Popen[bytes]) -> None:
        def drain(source: io.BufferedReader, stream: Literal["stdout", "stderr"]) -> None:
            try:
                while chunk := source.read1(4096):
                    if stream == "stderr":
                        self.diagnostics.extend(chunk[: max(0, 4096 - len(self.diagnostics))])
                    if self.callback is not None:
                        allowed = self.budget.take(chunk)
                        if allowed:
                            with self.lock:
                                self.callback(LogEvent(stream=stream, data=allowed))
            except BaseException as error:
                self.errors.append(error)
            finally:
                source.close()

        assert isinstance(process.stdout, io.BufferedReader)
        assert isinstance(process.stderr, io.BufferedReader)
        self.threads = [
            threading.Thread(target=drain, args=(process.stdout, "stdout"), daemon=True),
            threading.Thread(target=drain, args=(process.stderr, "stderr"), daemon=True),
        ]
        for thread in self.threads:
            thread.start()

    def finish(self) -> None:
        for thread in self.threads:
            thread.join(timeout=1)
