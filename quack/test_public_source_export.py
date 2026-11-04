"""Publication exports exclude local artifacts and deleted historical content."""

import subprocess
import tarfile

import pytest

from scripts.export_public_source import export_source, validate_source


def commit(root):
    subprocess.run(["git", "add", "--all"], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Artificial Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "test fixture",
        ],
        cwd=root,
        check=True,
    )


def test_export_uses_only_the_current_committed_tree(tmp_path):
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    (tmp_path / "README.md").write_text("Artificial source example.\n")
    (tmp_path / "deleted.csv").write_text("artificial historical fixture\n")
    commit(tmp_path)
    (tmp_path / "deleted.csv").unlink()
    commit(tmp_path)
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs/local.csv").write_text("artificial local fixture\n")
    destination = tmp_path / "outputs/public.tar.gz"
    count, checksum = export_source(tmp_path, destination)
    with tarfile.open(destination) as archive:
        assert set(archive.getnames()) == {"README.md", "PUBLIC_SOURCE_INVENTORY.json"}
        assert (
            archive.extractfile("README.md").read() == b"Artificial source example.\n"
        )
    assert count == 1
    assert len(checksum) == 64
    assert destination.with_name(destination.name + ".sha256").is_file()


@pytest.mark.parametrize(
    "name,content",
    [
        ("inputs/private.csv", b"a,b\n"),
        ("scripts/link", b"x"),
        ("scripts/payload.py", b"\0"),
        ("../README.md", b"x"),
    ],
)
def test_unreviewed_artifact_shapes_are_rejected(name, content):
    with pytest.raises(ValueError):
        validate_source(name, content)


def test_export_rejects_uncommitted_tracked_edits(tmp_path):
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    (tmp_path / "README.md").write_text("Artificial source example.\n")
    commit(tmp_path)
    (tmp_path / "README.md").write_text("Unreviewed source edit.\n")
    with pytest.raises(ValueError, match="Commit reviewed"):
        export_source(tmp_path, tmp_path / "outputs/public.tar.gz")


def test_source_links_are_never_exported(tmp_path):
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    (tmp_path / "README.md").write_text("Artificial source example.\n")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/linked.py").symlink_to("../README.md")
    commit(tmp_path)
    with pytest.raises(ValueError, match="Links and special files"):
        export_source(tmp_path, tmp_path / "outputs/public.tar.gz")


def test_reexport_refreshes_the_generated_inventory_without_duplicate_members(tmp_path):
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    (tmp_path / "README.md").write_text("Artificial source example.\n")
    (tmp_path / "PUBLIC_SOURCE_INVENTORY.json").write_text("{}")
    commit(tmp_path)
    destination = tmp_path / "outputs/public.tar.gz"
    count, _ = export_source(tmp_path, destination)
    with tarfile.open(destination) as archive:
        names = archive.getnames()
    assert count == 1
    assert names.count("PUBLIC_SOURCE_INVENTORY.json") == 1
