"""Explicit, process-local custom codecs; inspection never imports user code."""

from collections.abc import Callable
from pathlib import Path

from pydantic import ConfigDict, Field, PrivateAttr

from .artifacts import CustomType
from .records import Contract
from .sdk import Artifact, _checked, _inventory


class ArtifactCodec(Contract):
    model_config = ConfigDict(frozen=True)
    type: CustomType
    format: str = Field(min_length=1)
    serializer: Callable[[object, Path], None] = Field(exclude=True)
    validator: Callable[[Path], None] = Field(exclude=True)


class ArtifactRegistry(Contract):
    _codecs: dict[tuple[CustomType, str], ArtifactCodec] = PrivateAttr(default_factory=dict)

    def register(self, codec: ArtifactCodec) -> None:
        key = (codec.type, codec.format)
        if key in self._codecs:
            raise ValueError("A codec is already registered for this type/version/format.")
        self._codecs[key] = codec

    def _codec(self, artifact_type: CustomType, media_format: str) -> ArtifactCodec:
        try:
            return self._codecs[(artifact_type, media_format)]
        except KeyError as error:
            raise ValueError("No codec registered for this exact type/version/format.") from error

    def validate_payload(self, artifact_type: CustomType, media_format: str, path: Path) -> None:
        codec = self._codec(artifact_type, media_format)
        path = _checked(path.absolute())
        if path.is_dir():
            _inventory(path)
        elif not path.is_file():
            raise ValueError("Custom payload must be an existing file or directory.")
        codec.validator(path)

    def artifact(
        self, *, name: str, artifact_type: CustomType, media_format: str, value: object, path: Path
    ) -> Artifact:
        codec = self._codec(artifact_type, media_format)

        def serialize(target: Path) -> None:
            codec.serializer(value, target)
            self.validate_payload(artifact_type, media_format, target)

        return Artifact(
            name=name, type=artifact_type, format=media_format, path=path, serializer=serialize
        )
