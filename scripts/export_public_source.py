"""Export only committed source files, excluding worktree artifacts and Git history."""

import argparse
from gzip import GzipFile
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile


ROOT_FILES = {".gitignore", "LICENSE", "README.md", "poetry.lock", "pyproject.toml"}
SOURCE_DIRECTORIES = {
    ".github",
    "config",
    "docs",
    "examples",
    "experiments",
    "quack",
    "scripts",
    "syntheticicu",
}
SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".md", ".svg", ".sh"}
ARTIFICIAL_FIXTURES = {"examples/toy_spec.json", "quack/fixtures/dtw_l1_golden.json"}
CREDENTIAL_PATTERNS = [
    r"ghp_[A-Za-z0-9]{30,}",
    r"github_pat_[A-Za-z0-9_]{30,}",
    r"AKIA[0-9A-Z]{16}",
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
]


def validate_source(name, content):
    """Restrict the release to reviewed source formats and artificial fixtures."""
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or ".git" in path.parts:
        raise ValueError(f"Unsafe source archive path: {name}")
    if name not in ROOT_FILES:
        if path.parts[0] not in SOURCE_DIRECTORIES:
            raise ValueError(f"Unexpected source directory: {name}")
        if name not in ARTIFICIAL_FIXTURES and path.suffix not in SOURCE_SUFFIXES:
            raise ValueError(f"Unexpected source artifact type: {name}")
    if len(content) > 2 * 1024 * 1024 or b"\0" in content:
        raise ValueError(f"Binary or oversized source artifact: {name}")
    text = content.decode("utf-8")
    if any(re.search(pattern, text) for pattern in CREDENTIAL_PATTERNS):
        raise ValueError(f"Possible credential in source: {name}")
    return dict(
        path=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest()
    )


def export_source(repository, destination, revision="HEAD"):
    """Read blobs from one Git tree; neither untracked files nor ancestors are copied."""
    repository = Path(repository).resolve()
    destination = Path(destination).resolve()
    if not destination.is_relative_to(repository / "outputs"):
        raise ValueError(
            "Publication exports must remain under the repository outputs directory"
        )
    dirty = subprocess.check_output(
        ["git", "diff", "HEAD", "--name-only"], cwd=repository, text=True
    )
    if dirty.strip():
        raise ValueError(
            "Commit reviewed source changes before creating a publication export"
        )
    archive = subprocess.check_output(
        ["git", "archive", "--format=tar", revision], cwd=repository
    )
    records, files = [], []
    with tarfile.open(fileobj=BytesIO(archive), mode="r:") as source:
        for member in source.getmembers():
            if member.isdir() or member.name == "PUBLIC_SOURCE_INVENTORY.json":
                continue
            if not member.isfile():
                raise ValueError(
                    f"Links and special files cannot be published: {member.name}"
                )
            content = source.extractfile(member).read()
            records.append(validate_source(member.name, content))
            files.append((member.name, content))
    if not files:
        raise ValueError("The source tree is empty")
    inventory = json.dumps(
        dict(scope="committed source tree only; no Git history", files=records),
        indent=2,
    ).encode()
    files.append(("PUBLIC_SOURCE_INVENTORY.json", inventory))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        destination.open("wb") as stream,
        GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as compressed,
    ):
        with tarfile.open(fileobj=compressed, mode="w") as target:
            for name, content in sorted(files):
                member = tarfile.TarInfo(name)
                member.size = len(content)
                member.mode = 0o644
                target.addfile(member, BytesIO(content))
    checksum = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_name(destination.name + ".sha256").write_text(
        f"{checksum}  {destination.name}\n"
    )
    return len(records), checksum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="outputs/public_source/RECOMB27-source.tar.gz"
    )
    options = parser.parse_args()
    count, checksum = export_source(Path.cwd(), options.output)
    print(f"Exported {count} committed source files; SHA256 {checksum}")


if __name__ == "__main__":
    main()
