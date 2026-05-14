"""Smoke tests for drive module — uses mocked filesystem layout."""
from __future__ import annotations

from pathlib import Path

import pytest

from kaggle_playground_utils.drive import (
    restore_from_drive,
    sync_file_to_drive,
    sync_to_drive,
)


# -------------------------- restore_from_drive --------------------------


def test_restore_copies_missing_dir(tmp_path):
    drive = tmp_path / "drive"
    local = tmp_path / "local"
    (drive / "probs" / "v1").mkdir(parents=True)
    (drive / "probs" / "v1" / "oof.npy").write_bytes(b"fake")
    local.mkdir()

    report = restore_from_drive(drive, local, subdirs=["probs"])
    assert (local / "probs" / "v1" / "oof.npy").exists()
    assert report["probs"] >= 1


def test_restore_skips_existing_dir(tmp_path):
    drive = tmp_path / "drive"
    local = tmp_path / "local"
    (drive / "probs" / "v1").mkdir(parents=True)
    (drive / "probs" / "v1" / "oof.npy").write_bytes(b"drive-version")
    (local / "probs" / "v1").mkdir(parents=True)
    (local / "probs" / "v1" / "oof.npy").write_bytes(b"local-version")

    # Restore shouldn't overwrite existing local content
    restore_from_drive(drive, local, subdirs=["probs"])
    assert (local / "probs" / "v1" / "oof.npy").read_bytes() == b"local-version"


def test_restore_merges_nested_files(tmp_path):
    """When local has empty subdir + drive has nested files, files get merged in."""
    drive = tmp_path / "drive"
    local = tmp_path / "local"
    # Drive has harvest/v18/{kernel}/submission.csv
    (drive / "harvest" / "v18" / "kernel_a").mkdir(parents=True)
    (drive / "harvest" / "v18" / "kernel_a" / "submission.csv").write_text("id,p\n1,0.5\n")
    (drive / "harvest" / "v18" / "manifest.json").write_text('[]')

    # Local has only the v18 dir + manifest (git clone leftover)
    (local / "harvest" / "v18").mkdir(parents=True)
    (local / "harvest" / "v18" / "manifest.json").write_text('[]')

    restore_from_drive(drive, local, subdirs=["harvest"])
    assert (local / "harvest" / "v18" / "kernel_a" / "submission.csv").exists()


def test_restore_handles_missing_drive_subdir(tmp_path):
    """Doesn't error when a subdir is missing on Drive."""
    drive = tmp_path / "drive"
    drive.mkdir()
    local = tmp_path / "local"
    local.mkdir()

    report = restore_from_drive(drive, local, subdirs=["probs", "harvest"])
    assert report["probs"] == 0
    assert report["harvest"] == 0


# -------------------------- sync_to_drive --------------------------


def test_sync_copies_local_files_to_drive(tmp_path):
    drive = tmp_path / "drive"
    local = tmp_path / "local"
    (local / "submissions").mkdir(parents=True)
    (local / "submissions" / "v1.csv").write_text("id,p\n1,0.5\n")

    report = sync_to_drive(local, drive, subdirs=["submissions"])
    assert (drive / "submissions" / "v1.csv").exists()
    assert report["submissions"] >= 1


def test_sync_respects_pattern(tmp_path):
    drive = tmp_path / "drive"
    local = tmp_path / "local"
    (local / "submissions").mkdir(parents=True)
    (local / "submissions" / "v18.001.csv").write_text("a")
    (local / "submissions" / "v17_NM.csv").write_text("b")

    sync_to_drive(local, drive, subdirs=["submissions"], pattern="v18.*")
    assert (drive / "submissions" / "v18.001.csv").exists()
    assert not (drive / "submissions" / "v17_NM.csv").exists()


# -------------------------- sync_file_to_drive --------------------------


def test_sync_file_creates_parent(tmp_path):
    local_file = tmp_path / "releases.jsonl"
    local_file.write_text('{"version": "v1"}\n')
    drive_path = tmp_path / "drive" / "releases.jsonl"

    ok = sync_file_to_drive(local_file, drive_path)
    assert ok is True
    assert drive_path.exists()


def test_sync_file_skip_when_identical(tmp_path):
    local_file = tmp_path / "f.txt"
    local_file.write_text("same")
    drive_path = tmp_path / "drive" / "f.txt"
    drive_path.parent.mkdir(parents=True)
    drive_path.write_text("same")

    ok = sync_file_to_drive(local_file, drive_path)
    assert ok is False  # no copy needed
