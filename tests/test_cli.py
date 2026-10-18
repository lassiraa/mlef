import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlef.cli import main
from mlef.runner import relative_path, run, source_files


@contextlib.contextmanager
def project():
    with tempfile.TemporaryDirectory(prefix="mlef test ") as directory:
        root = Path(directory).resolve()
        subprocess.run(["git", "init", "--quiet", str(root)], check=True)
        before = Path.cwd()
        try:
            os.chdir(root)
            yield root
        finally:
            os.chdir(before)


class CliTests(unittest.TestCase):
    def test_init_is_idempotent_and_preserves_ignore_file(self):
        with project() as root, contextlib.redirect_stdout(io.StringIO()):
            (root / ".gitignore").write_text("existing-entry", encoding="utf-8")
            self.assertEqual(main(["init"]), 0)
            self.assertEqual(main(["init"]), 0)
            self.assertEqual((root / ".gitignore").read_text(), "existing-entry\n/.mlef/\n")
            self.assertTrue((root / ".mlef/runs").is_dir())
            result = subprocess.run(
                ["git", "check-ignore", ".mlef/runs/test"], capture_output=True, check=False
            )
            self.assertEqual(result.returncode, 0)

    def test_arguments_after_separator_are_forwarded_verbatim(self):
        with project() as root, patch("mlef.cli.run", return_value=0) as execute:
            self.assertEqual(main(["run", "train.py", "--", "--output", "a b", "$(text)"]), 0)
            execute.assert_called_once_with(
                root, Path("train.py"), ["--output", "a b", "$(text)"], [], []
            )

    def test_nested_directory_and_missing_project_are_usage_errors(self):
        with project() as root, contextlib.redirect_stderr(io.StringIO()):
            (root / "nested").mkdir()
            os.chdir(root / "nested")
            self.assertEqual(main(["init"]), 2)
        with tempfile.TemporaryDirectory() as directory:
            before = Path.cwd()
            try:
                os.chdir(directory)
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(main(["init"]), 2)
            finally:
                os.chdir(before)

    def test_bad_script_does_not_create_a_run(self):
        with project() as root, contextlib.redirect_stdout(io.StringIO()):
            main(["init"])
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["run", "missing.py"]), 2)
            self.assertEqual(list((root / ".mlef/runs").iterdir()), [])

    def test_source_selection_and_path_boundaries(self):
        with project() as root:
            (root / "train.py").write_text("print('hello')")
            (root / "helper.py").write_text("value = 1")
            (root / "ignored.py").write_text("secret = 1")
            subprocess.run(["git", "add", "helper.py"], check=True)
            selected = source_files(root, Path("train.py"), [])
            self.assertEqual(set(selected), {Path("train.py"), Path("helper.py")})
            for path in [Path("../escape"), root / "absolute", Path(".git/config"), Path(".mlef")]:
                with self.assertRaises(ValueError):
                    relative_path(root, path)
            (root / ".env").write_text("SECRET=never-copy")
            with self.assertRaises(ValueError):
                source_files(root, Path("train.py"), [Path(".env")])

    def test_missing_docker_preserves_failed_metadata_and_logs(self):
        with project() as root, contextlib.redirect_stdout(io.StringIO()):
            main(["init"])
            (root / "train.py").write_text("print('hello')")
            with patch("mlef.runner.shutil.which", return_value=None):
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(run(root, Path("train.py"), [], [], []), 1)
            (directory,) = (root / ".mlef/runs").iterdir()
            metadata = json.loads((directory / "run.json").read_text())
            self.assertEqual(metadata["state"], "failed")
            self.assertIsNone(metadata["execution_exit_code"])
            self.assertIsNotNone(metadata["finished_at"])
            self.assertIn("Docker executable not found", metadata["error"]["message"])
            for name in ["stdout.log", "stderr.log", "runner.log"]:
                self.assertTrue((directory / name).exists())

    def test_build_failure_is_recorded(self):
        with project() as root, contextlib.redirect_stdout(io.StringIO()):
            main(["init"])
            (root / "train.py").write_text("print('hello')")

            def fail_build(arguments, log, **kwargs):
                if arguments[0] == "build":
                    raise RuntimeError("Image build failed")
                return subprocess.CompletedProcess(arguments, 0)

            with patch("mlef.runner.shutil.which", return_value="docker"):
                with patch("mlef.runner.docker", side_effect=fail_build):
                    with contextlib.redirect_stderr(io.StringIO()):
                        self.assertEqual(run(root, Path("train.py"), [], [], []), 1)
            (directory,) = (root / ".mlef/runs").iterdir()
            metadata = json.loads((directory / "run.json").read_text())
            self.assertEqual(metadata["state"], "failed")
            self.assertIn("Image build failed", (directory / "runner.log").read_text())
            self.assertEqual(list(directory.glob("build-*")), [])


if __name__ == "__main__":
    unittest.main()
