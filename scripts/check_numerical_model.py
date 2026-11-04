"""Run numerical model checks in an allocation, optionally with the full suite."""

import subprocess
import sys
import tarfile
import tempfile
import os
from pathlib import Path


def main():
    """Run pytest using the interpreter selected by Poetry."""
    paths = sys.argv[1:] or [
        "quack/test_model_numeric.py",
        "quack/test_numerical_workflow.py",
        "quack/test_evaluation_numeric.py",
        "quack/test_numerical_task_parity.py",
    ]
    test_temp = (
        Path.cwd()
        / "outputs/numerical_data"
        / f"pytest-{os.environ.get('SLURM_JOB_ID', 'local')}"
    )
    test_temp.parent.mkdir(parents=True, exist_ok=True)
    test_options = ["--basetemp", str(test_temp)]
    if paths[0] == "--baseline":
        root = Path.cwd()
        output_root = root / "outputs" / "numerical_data"
        output_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=output_root, prefix="baseline-"
        ) as directory:
            snapshot = Path(directory)
            archive_path = snapshot / "source.tar"
            with archive_path.open("wb") as archive:
                subprocess.run(
                    [
                        "git",
                        "archive",
                        "HEAD",
                        "syntheticicu",
                        "quack",
                        "config",
                        "scripts",
                    ],
                    stdout=archive,
                    check=True,
                )
            with tarfile.open(archive_path) as archive:
                archive.extractall(snapshot, filter="data")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    *test_options,
                    *(paths[1:] or ["quack"]),
                ],
                cwd=snapshot,
                check=True,
            )
        return
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *test_options, *paths], check=True
    )


if __name__ == "__main__":
    main()
