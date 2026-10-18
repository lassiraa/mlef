"""Opt-in checks against a real Docker engine; no mock container execution."""

import json
import os
import signal
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from test_cli import project

from mlef import runner
from mlef.cli import init

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/train.py"


@unittest.skipUnless(os.environ.get("MLEF_TEST_DOCKER") == "1", "Set MLEF_TEST_DOCKER=1")
class DockerTests(unittest.TestCase):
    def test_nonroot_container_can_write_workspace_and_artifacts(self):
        with project() as root:
            init(root)
            (root / "train.py").write_text(
                "import os\nfrom pathlib import Path\n"
                "Path('results').mkdir()\n"
                "Path('results/value.txt').write_text('workspace')\n"
                "Path(os.environ['MLEF_OUTPUT_DIR'], 'value.txt').write_text('artifact')\n"
            )
            uid = os.getuid() if hasattr(os, "getuid") else 1000
            gid = os.getgid() if hasattr(os, "getgid") else 1000
            with patch("mlef.runner.os.getuid", return_value=uid, create=True):
                with patch("mlef.runner.os.getgid", return_value=gid, create=True):
                    self.assertEqual(
                        runner.run(root, Path("train.py"), [], [], [Path("results")]), 0
                    )
            (directory,) = (root / ".mlef/runs").iterdir()
            self.assertEqual((directory / "artifacts/results/value.txt").read_text(), "workspace")
            self.assertEqual((directory / "artifacts/value.txt").read_text(), "artifact")

    def test_success_failure_arguments_and_declared_outputs(self):
        with project() as root:
            self.cli("init")
            (root / "train.py").write_bytes(EXAMPLE.read_bytes())
            success = self.cli("run", "train.py", "--", "--accuracy", "0.95")
            self.assertIn("Training complete", success.stdout)
            failure = self.cli("run", "train.py", "--", "--fail", expected=1)
            self.assertIn("Requested example failure", failure.stderr)
            directories = list((root / ".mlef/runs").iterdir())
            self.assertEqual(len(directories), 2)
            runs = {
                json.loads((path / "run.json").read_text())["state"]: path for path in directories
            }
            self.assertEqual(set(runs), {"succeeded", "failed"})
            for state, directory in runs.items():
                metadata = json.loads((directory / "run.json").read_text())
                self.assertEqual(metadata["execution_exit_code"], 0 if state == "succeeded" else 7)
                self.assertTrue(metadata["started_at"])
                self.assertTrue(metadata["finished_at"])
                self.assertIn("Training complete", (directory / "stdout.log").read_text())
                self.assertTrue((directory / "artifacts/model.bin").exists())
            metrics = json.loads((runs["succeeded"] / "artifacts/metrics.json").read_text())
            self.assertEqual(metrics["accuracy"], 0.95)
            (root / "train.py").write_text(
                "from pathlib import Path\n"
                "Path('results').mkdir()\n"
                "Path('results/value.txt').write_text('collected')\n"
            )
            self.cli("run", "train.py", "--output", "results")
            collected = list((root / ".mlef/runs").glob("*/artifacts/results/value.txt"))
            self.assertEqual(len(collected), 1)
            self.assertEqual(collected[0].read_text(), "collected")
            self.assertFalse((root / "results").exists())

            # Missing required outputs must not silently turn a run into a success.
            self.cli("run", "train.py", "--output", "missing", expected=1)
            collection_errors = [
                json.loads(path.read_text())
                for path in (root / ".mlef/runs").glob("*/run.json")
                if (json.loads(path.read_text()).get("error") or {}).get("category") == "collection"
            ]
            self.assertEqual(len(collection_errors), 1)
            self.assertEqual(collection_errors[0]["execution_exit_code"], 0)

    def test_interrupt_handler_with_real_container(self):
        # Inject KeyboardInterrupt at the wait boundary to cover the handler on Windows too.
        with project() as root:
            init(root)
            (root / "wait.py").write_text(
                "import os, time\nfrom pathlib import Path\n"
                "Path(os.environ['MLEF_OUTPUT_DIR'], 'partial.txt').write_text('saved')\n"
                "print('ready', flush=True)\ntime.sleep(120)\n"
            )
            real_popen = subprocess.Popen

            def popen(command, *args, **kwargs):
                process = real_popen(command, *args, **kwargs)
                if command[:3] == ["docker", "start", "--attach"]:
                    original_wait = process.wait
                    interrupted = False

                    def wait(timeout=None):
                        nonlocal interrupted
                        if not interrupted:
                            interrupted = True
                            deadline = time.monotonic() + 30
                            while not list((root / ".mlef/runs").glob("*/artifacts/partial.txt")):
                                if time.monotonic() > deadline:
                                    raise RuntimeError("Container did not produce a partial output")
                                time.sleep(0.1)
                            raise KeyboardInterrupt
                        return original_wait(timeout)

                    process.wait = wait
                return process

            with patch("mlef.runner.subprocess.Popen", side_effect=popen):
                self.assertEqual(runner.run(root, Path("wait.py"), [], [], []), 130)
            (directory,) = (root / ".mlef/runs").iterdir()
            metadata = json.loads((directory / "run.json").read_text())
            self.assertEqual(metadata["state"], "cancelled")
            self.assertTrue(metadata["finished_at"])
            self.assertEqual((directory / "artifacts/partial.txt").read_text(), "saved")
            self.assertIn("ready", (directory / "stdout.log").read_text())

    @unittest.skipIf(sys.platform == "win32", "SIGINT subprocess delivery checked on POSIX CI")
    def test_cancellation_preserves_partial_outputs(self):
        with project() as root:
            self.cli("init")
            (root / "wait.py").write_text(
                "import os, time\nfrom pathlib import Path\n"
                "Path(os.environ['MLEF_OUTPUT_DIR'], 'partial.txt').write_text('saved')\n"
                "print('ready', flush=True)\ntime.sleep(120)\n"
            )
            process = subprocess.Popen(
                [sys.executable, "-m", "mlef", "run", "wait.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                deadline = time.monotonic() + 120
                while not list((root / ".mlef/runs").glob("*/artifacts/partial.txt")):
                    if process.poll() is not None:
                        self.fail(str(process.communicate()))
                    if time.monotonic() > deadline:
                        self.fail("Container never wrote its partial output")
                    time.sleep(0.2)
                process.send_signal(signal.SIGINT)
                stdout, stderr = process.communicate(timeout=30)
                self.assertEqual(process.returncode, 130, stdout + stderr)
                (directory,) = (root / ".mlef/runs").iterdir()
                metadata = json.loads((directory / "run.json").read_text())
                self.assertEqual(metadata["state"], "cancelled")
                self.assertEqual((directory / "artifacts/partial.txt").read_text(), "saved")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()

    def cli(self, *arguments, expected=0):
        result = subprocess.run(
            [sys.executable, "-m", "mlef", *arguments],
            capture_output=True,
            text=True,
            timeout=240,
        )
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result


if __name__ == "__main__":
    unittest.main()
