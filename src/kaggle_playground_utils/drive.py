"""Google Drive restore/sync — Colab persistent storage helpers.

When running on Colab, /content is ephemeral (wiped on session end). Drive
is persistent. These helpers move artifacts between the two:

  - restore_from_drive: pull missing items Drive → /content
  - sync_to_drive: push fresh artifacts /content → Drive

Both use merge semantics (don't overwrite existing destination), so they're
safe to re-run.
"""
from __future__ import annotations

import shutil
from pathlib import Path


def restore_from_drive(
    drive_root: Path,
    local_root: Path,
    subdirs: list[str] | None = None,
) -> dict[str, int]:
    """Copy missing items from Drive to local /content.

    Args:
        drive_root: Drive folder (e.g., /content/drive/MyDrive/Colab Notebooks/kaggle/s6e5)
        local_root: Local working dir (e.g., /content/playground-s6e5)
        subdirs: which subdirs to mirror. Default: ['probs', 'harvest', 'submissions'].

    Returns:
        dict of {subdir_name: count_of_items_copied}.
    """
    subdirs = subdirs or ["probs", "harvest", "submissions"]
    report: dict[str, int] = {}

    for sub in subdirs:
        src_dir = drive_root / sub
        dst_dir = local_root / sub
        if not src_dir.exists():
            report[sub] = 0
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for item in src_dir.iterdir():
            target = dst_dir / item.name
            if item.is_dir():
                if not target.exists():
                    shutil.copytree(item, target)
                    count += 1
                else:
                    # Recurse one level for nested subdirs (e.g., harvest/v18/{kernel}/)
                    for nested in item.iterdir():
                        nested_target = target / nested.name
                        if nested.is_dir() and not nested_target.exists():
                            shutil.copytree(nested, nested_target)
                            count += 1
                        elif nested.is_file() and (
                            not nested_target.exists()
                            or nested_target.stat().st_size != nested.stat().st_size
                        ):
                            shutil.copyfile(nested, nested_target)
                            count += 1
            elif item.is_file():
                if not target.exists() or target.stat().st_size != item.stat().st_size:
                    shutil.copyfile(item, target)
                    count += 1
        report[sub] = count
    return report


def sync_to_drive(
    local_root: Path,
    drive_root: Path,
    subdirs: list[str] | None = None,
    pattern: str | None = None,
) -> dict[str, int]:
    """Copy fresh artifacts from local to Drive.

    Args:
        local_root: Local working dir
        drive_root: Drive folder
        subdirs: which subdirs to sync. Default: ['probs', 'submissions'].
        pattern: optional glob filter (e.g., 'v18.*') — only sync matching items.

    Returns:
        dict of {subdir: count_synced}.
    """
    subdirs = subdirs or ["probs", "submissions"]
    report: dict[str, int] = {}

    for sub in subdirs:
        src_dir = local_root / sub
        dst_dir = drive_root / sub
        if not src_dir.exists():
            report[sub] = 0
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        items = (
            list(src_dir.glob(pattern)) if pattern
            else list(src_dir.iterdir())
        )
        for item in items:
            target = dst_dir / item.name
            if item.is_dir():
                if not target.exists():
                    shutil.copytree(item, target)
                    count += 1
                # else: skip (subdir already exists; avoid re-syncing large content)
            elif item.is_file():
                if not target.exists() or target.stat().st_size != item.stat().st_size:
                    shutil.copyfile(item, target)
                    count += 1
        report[sub] = count
    return report


def sync_file_to_drive(local_path: Path, drive_path: Path) -> bool:
    """Sync a single file (e.g., experiments.jsonl, releases.jsonl).

    Returns True if a copy was performed (file was missing or differed in size).
    """
    if not local_path.exists():
        return False
    drive_path.parent.mkdir(parents=True, exist_ok=True)
    if drive_path.exists() and drive_path.stat().st_size == local_path.stat().st_size:
        return False
    shutil.copyfile(local_path, drive_path)
    return True
