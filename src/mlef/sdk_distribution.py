"""Capture the installed pure-Python SDK as an immutable, installable wheel."""

import base64
import csv
import hashlib
import io
import zipfile
from importlib.metadata import distribution
from pathlib import Path

from . import __version__
from .environment import SdkWheel, file_hash


def capture_sdk(directory: Path) -> SdkWheel:
    installed = distribution("mlef")
    if installed.version != __version__:
        raise ValueError("Installed SDK metadata does not match the loaded package version.")
    version = installed.version
    metadata = installed.read_text("METADATA")
    if metadata is None:
        raise ValueError("The installed SDK has no distribution metadata.")
    identity = SdkWheel(
        version=version,
        filename=f"mlef-{version}-py3-none-any.whl",
        sha256="0" * 64,
        pipeline_reuse=True,
        pipeline_cache=True,
    )
    package = Path(__file__).parent
    contents = {}
    for path in sorted(package.rglob("*")):
        relative = path.relative_to(package)
        if "__pycache__" in relative.parts:
            continue
        if path.suffix != ".py" and not (
            relative.parent == Path("templates") and path.suffix in {".txt", ".toml"}
        ):
            continue
        if path.is_symlink():
            raise ValueError("SDK files must not be symbolic links.")
        if path.is_file():
            contents[f"mlef/{relative.as_posix()}"] = path.read_bytes()
    prefix = f"mlef-{version}.dist-info"
    contents[f"{prefix}/METADATA"] = metadata.encode("utf-8")
    contents[f"{prefix}/WHEEL"] = (
        b"Wheel-Version: 1.0\nGenerator: mlef\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    )
    entry_points = installed.read_text("entry_points.txt")
    if entry_points is not None:
        contents[f"{prefix}/entry_points.txt"] = entry_points.encode("utf-8")
    for item in installed.files or []:
        if ".dist-info/licenses/" in item.as_posix():
            suffix = item.as_posix().split(".dist-info/", 1)[1]
            contents[f"{prefix}/{suffix}"] = Path(str(installed.locate_file(item))).read_bytes()
    inventory = io.StringIO(newline="")
    writer = csv.writer(inventory)
    for name, content in sorted(contents.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
        writer.writerow([name, f"sha256={digest}", len(content)])
    writer.writerow([f"{prefix}/RECORD", "", ""])
    contents[f"{prefix}/RECORD"] = inventory.getvalue().encode("utf-8")
    directory.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(directory / identity.filename, "x", zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(contents.items()):
            entry = zipfile.ZipInfo(name)
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, content)
    identity.sha256 = file_hash(directory / identity.filename)
    return identity
