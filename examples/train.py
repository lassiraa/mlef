"""A dependency-free experiment for trying mlef."""

import argparse
import json
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--accuracy", type=float, default=0.9)
parser.add_argument("--fail", action="store_true", help="Fail after writing partial outputs.")
args = parser.parse_args()
output = Path(os.environ["MLEF_OUTPUT_DIR"])
output.mkdir(parents=True, exist_ok=True)
(output / "metrics.json").write_text(json.dumps({"accuracy": args.accuracy}) + "\n")
(output / "model.bin").write_bytes(b"example model\n")
print("Training complete", flush=True)
if args.fail:
    print("Requested example failure", file=sys.stderr, flush=True)
    raise SystemExit(7)
