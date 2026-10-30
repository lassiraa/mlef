"""Validated runtime requests and Docker's applied control values."""

import re
from decimal import Decimal
from typing import Literal

from pydantic import Field, field_validator

from .records import Contract


class Limits(Contract):
    cpus: float | None = Field(default=None, ge=1e-9, allow_inf_nan=False)
    memory_bytes: int | None = Field(default=None, gt=0, le=2**63 - 1)
    network: Literal["default", "none"] = "default"

    @field_validator("cpus")
    @classmethod
    def nano_precision(cls, value: float | None) -> float | None:
        if value is not None:
            nano = Decimal(str(value)) * 1_000_000_000
            if nano != nano.to_integral_value() or nano > 2**63 - 1:
                raise ValueError(
                    "CPU limits require at most nine decimal places within int64 nanocpus."
                )
        return value

    def capabilities(self, info: dict[str, object]) -> None:
        if self.cpus is not None and info.get("CpuCfsQuota") is not True:
            raise RuntimeError("Docker reports CPU quota enforcement is unavailable.")
        if self.memory_bytes is not None and (
            info.get("MemoryLimit") is not True or info.get("SwapLimit") is not True
        ):
            raise RuntimeError("Docker reports memory/swap enforcement is unavailable.")

    def docker_arguments(self) -> list[str]:
        arguments: list[str] = []
        if self.cpus is not None:
            arguments += ["--cpus", str(self.cpus)]
        if self.memory_bytes is not None:
            arguments += [
                "--memory",
                str(self.memory_bytes),
                "--memory-swap",
                str(self.memory_bytes),
            ]
        if self.network == "none":
            arguments += ["--network", "none"]
        return arguments

    def applied(self, host: dict[str, object]) -> dict[str, object]:
        if self.cpus is not None and host.get("NanoCpus") != round(self.cpus * 1_000_000_000):
            raise RuntimeError("Docker did not apply the requested CPU limit.")
        if self.memory_bytes is not None and host.get("Memory") != self.memory_bytes:
            raise RuntimeError("Docker did not apply the requested memory limit.")
        if self.memory_bytes is not None and host.get("MemorySwap") != self.memory_bytes:
            raise RuntimeError("Docker did not apply the requested swap restriction.")
        if self.network == "none" and host.get("NetworkMode") != "none":
            raise RuntimeError("Docker did not apply runtime network-disable.")
        return {key: host.get(key) for key in ["NanoCpus", "Memory", "MemorySwap", "NetworkMode"]}


def memory_bytes(value: str | None) -> int | None:
    if value is None:
        return None
    match = re.fullmatch(r"([1-9][0-9]*)([kmgt]?)", value.lower())
    if match is None:
        raise ValueError("Memory must be positive bytes or an integer with k/m/g/t suffix.")
    factors = {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}
    return int(match[1]) * factors[match[2]]
