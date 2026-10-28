"""A tiny experiment that requires a locked third-party dependency."""

import json
import os
from pathlib import Path

from packaging.version import Version

versions = sorted([Version("1.10"), Version("1.2")])
output = Path(os.environ["MLEF_OUTPUT_DIR"])
(output / "metrics.json").write_text(json.dumps({"versions_checked": len(versions)}))
print(f"Sorted versions: {versions}")
