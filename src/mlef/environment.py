"""Locked dependency inputs and the environment actually used by Docker."""

import hashlib
import json
import platform
import tomllib
from pathlib import Path
from typing import BinaryIO, Literal, Self

from pydantic import BaseModel, Field, model_validator

from . import __version__
from .processes import command
from .records import Contract

UV_IMAGE = "ghcr.io/astral-sh/uv:0.13.6"
INSTALL_COMMAND = [
    "uv",
    "sync",
    "--locked",
    "--no-default-groups",
    "--no-dev",
    "--no-install-project",
    "--python",
    "/usr/local/bin/python",
    "--no-managed-python",
    "--no-python-downloads",
    "--no-build",
]


class ProjectMetadata(BaseModel):
    dependencies: list[str] = Field(default_factory=list)
    dynamic: list[str] = Field(default_factory=list)
    requires_python: str | None = Field(default=None, alias="requires-python")


class UvSettings(BaseModel):
    workspace: dict[str, object] | None = None
    sources: dict[str, object] | None = None


class ToolSettings(BaseModel):
    uv: UvSettings = Field(default_factory=UvSettings)
    poetry: dict[str, object] | None = None
    pdm: dict[str, object] | None = None


class ProjectFile(BaseModel):
    project: ProjectMetadata = Field(default_factory=ProjectMetadata)
    tool: ToolSettings = Field(default_factory=ToolSettings)


class LockedPackage(BaseModel):
    source: dict[str, str]


class LockFile(BaseModel):
    version: int = Field(strict=True)
    package: list[LockedPackage] = Field(min_length=1)


class DependencyInputs(BaseModel):
    hashes: dict[str, str] = Field(default_factory=dict)
    install_command: list[str] | None = None


def file_hash(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def dependency_inputs(workspace: Path) -> DependencyInputs:
    unsupported = [
        path.name
        for path in workspace.iterdir()
        if path.name
        in {
            "Pipfile",
            "Pipfile.lock",
            "poetry.lock",
            "pdm.lock",
            "uv.toml",
            "setup.py",
            "setup.cfg",
        }
        or (path.name.startswith("requirements") and path.suffix in {".txt", ".in"})
    ]
    if unsupported:
        raise ValueError(f"Unsupported dependency inputs: {', '.join(sorted(unsupported))}")
    project_path, lock_path = workspace / "pyproject.toml", workspace / "uv.lock"
    if not project_path.exists():
        if lock_path.exists():
            raise ValueError("uv.lock requires pyproject.toml.")
        return DependencyInputs()
    project = ProjectFile.model_validate(tomllib.loads(project_path.read_text(encoding="utf-8")))
    metadata, tools = project.project, project.tool
    uv = tools.uv
    if (
        tools.poetry is not None
        or tools.pdm is not None
        or uv.workspace is not None
        or uv.sources is not None
    ):
        raise ValueError(
            "Only single-project registry dependencies are supported; no workspace/sources."
        )
    hashes = {"pyproject.toml": file_hash(project_path)}
    if not lock_path.exists():
        if metadata.dependencies or metadata.dynamic or metadata.requires_python is not None:
            raise ValueError("Project dependencies require an existing uv.lock; run uv lock first.")
        return DependencyInputs(hashes=hashes)
    lock = LockFile.model_validate(tomllib.loads(lock_path.read_text(encoding="utf-8")))
    for package in lock.package:
        source = package.source
        if not (set(source) == {"registry"} or source in ({"virtual": "."}, {"editable": "."})):
            raise ValueError(
                "Only registry packages and the root project are supported in uv.lock."
            )
    hashes["uv.lock"] = file_hash(lock_path)
    return DependencyInputs(hashes=hashes, install_command=INSTALL_COMMAND)


class ImageIdentity(BaseModel):
    id: str
    digests: list[str]
    os: str
    architecture: str


class SystemInfo(BaseModel):
    os: str
    architecture: str
    python: str


class ContainerInfo(SystemInfo):
    os_release: dict[str, str]
    packages: dict[str, str]
    uv_version: str | None


class SdkWheel(Contract):
    version: str = Field(pattern=r"^[A-Za-z0-9_.]+$")
    filename: str = Field(pattern=r"^mlef-[A-Za-z0-9_.]+-py3-none-any\.whl$")
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    pipeline_reuse: bool = False

    @model_validator(mode="after")
    def filename_version(self) -> Self:
        if self.filename != f"mlef-{self.version}-py3-none-any.whl":
            raise ValueError("SDK wheel filename and version disagree.")
        return self


class Environment(BaseModel):
    schema_version: Literal[1] = 1
    mlef_version: str = __version__
    host: SystemInfo = Field(
        default_factory=lambda: SystemInfo(
            os=platform.platform(),
            architecture=platform.machine(),
            python=platform.python_version(),
        )
    )
    container: ContainerInfo | None = None
    base_image: ImageIdentity | None = None
    uv_image: ImageIdentity | None = None
    execution_image: ImageIdentity | None = None
    dependencies: DependencyInputs | None = None
    sdk_wheel: SdkWheel | None = None
    warnings: list[str] = Field(default_factory=lambda: ["Environment capture is incomplete."])


def query(arguments: list[str], log: BinaryIO) -> str:
    result = command(["docker", *arguments], log, capture=True, timeout=60)
    if result.returncode:
        raise RuntimeError(
            f"docker {arguments[0]} failed during environment capture; see runner.log."
        )
    return result.stdout.decode("utf-8")


def image_identity(image: str, log: BinaryIO) -> ImageIdentity:
    records = json.loads(query(["image", "inspect", image], log))
    if not isinstance(records, list) or not records or not isinstance(records[0], dict):
        raise ValueError("Docker returned invalid image inspection data.")
    record = records[0]
    return ImageIdentity.model_validate(
        {
            "id": record.get("Id"),
            "digests": record.get("RepoDigests") or [],
            "os": record.get("Os"),
            "architecture": record.get("Architecture"),
        }
    )


PROBE = """import importlib.metadata, json, platform, shutil, subprocess
uv = shutil.which('uv')
print(json.dumps(dict(os=platform.platform(), architecture=platform.machine(),
    python=platform.python_version(),
    os_release=platform.freedesktop_os_release(),
    packages={d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
    uv_version=subprocess.check_output([uv, '--version'], text=True).strip() if uv else None)))
"""


def container_info(
    image: str,
    log: BinaryIO,
    user: str | None = None,
    *,
    name: str | None = None,
    labels: dict[str, str] | None = None,
    resource_arguments: list[str] | None = None,
) -> ContainerInfo:
    arguments = ["run", "--rm"]
    arguments += list(resource_arguments or [])
    if "--network" not in arguments:
        arguments += ["--network", "none"]
    if name is not None:
        arguments += ["--name", name]
    for key, value in (labels or {}).items():
        arguments += ["--label", f"{key}={value}"]
    if user is not None:
        arguments += ["--user", user]
    arguments += ["--entrypoint", "python", image, "-I", "-c", PROBE]
    return ContainerInfo.model_validate_json(query(arguments, log))


def build_recipe(base: str, uv: str | None, user: str | None, sdk: SdkWheel | None = None) -> str:
    ownership = f"--chown={user} " if user is not None else ""
    recipe = f"FROM {base}\nWORKDIR /workspace\nCOPY {ownership}workspace/ /workspace/\n"
    if uv is not None:
        recipe += f"COPY --from={uv} /uv /usr/local/bin/uv\n"
        recipe += "ENV UV_PROJECT_ENVIRONMENT=/opt/mlef-venv\n"
        recipe += "RUN " + " ".join(INSTALL_COMMAND) + "\n"
        recipe += 'ENV PATH="/opt/mlef-venv/bin:$PATH"\n'
    if sdk is not None:
        if uv is None:
            raise ValueError("Pipeline SDK installation requires a locked experiment environment.")
        path = f"/opt/mlef-sdk/{sdk.filename}"
        recipe += "COPY sdk/ /opt/mlef-sdk/\n"
        recipe += (
            'RUN python -I -c "import hashlib; from pathlib import Path; '
            f"assert hashlib.sha256(Path('{path}').read_bytes()).hexdigest() == '{sdk.sha256}'"
            '"\n'
        )
        recipe += (
            "RUN uv pip install --python /opt/mlef-venv/bin/python --no-deps --no-index "
            f"--reinstall {path}\n"
            "RUN uv pip check --python /opt/mlef-venv/bin/python\n"
        )
    if user is not None:
        recipe += f"RUN chown {user} /workspace\n"
    return recipe
