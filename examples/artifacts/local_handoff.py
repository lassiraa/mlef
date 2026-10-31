"""Run two ordinary local script steps with verified, typed artifact handoff."""

import argparse
from pathlib import Path

from mlef import Artifact, ArtifactType, Input, Output, Step, load_artifacts, read_step


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Parent for two new attempt directories.")
    args = parser.parse_args()
    ingest = Step(
        name="ingest", outputs=[Output(name="raw", type=ArtifactType.TEXT, format="text/plain")]
    )
    with ingest.local(args.directory / "ingest") as context:
        path = context.output_path("raw", "message.txt")
        path.write_text("ordinary scripts, reproducible evidence", encoding="utf-8")
        ingest.save(
            [Artifact(name="raw", type=ArtifactType.TEXT, format="text/plain", path=path)],
            context=context,
        )

    transform = Step(
        name="transform",
        inputs=[Input(name="raw", type=ArtifactType.TEXT, format="text/plain")],
        outputs=[Output(name="report", type=ArtifactType.TEXT, format="text/plain")],
    )
    inputs = load_artifacts(args.directory / "ingest")
    with transform.local(args.directory / "transform", inputs=inputs) as context:
        text = context.input_path("raw", "message.txt").read_text(encoding="utf-8")
        path = context.output_path("report", "report.txt")
        path.write_text(text.upper(), encoding="utf-8")
        transform.save(
            [Artifact(name="report", type=ArtifactType.TEXT, format="text/plain", path=path)],
            context=context,
        )
    record = read_step(args.directory / "transform")
    print(record.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
