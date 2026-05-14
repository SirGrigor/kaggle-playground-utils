"""Environment detection + Kaggle CLI auth — Colab and local.

Use these helpers from any pipeline script. They:
  - Detect whether we're running on Colab (Drive mounted) or local
  - Set up Kaggle CLI credentials from Colab Secrets OR existing kaggle.json
  - Provide canonical paths for Drive-backed storage
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal


def detect_environment() -> Literal["colab", "local"]:
    """Return 'colab' if Drive is mounted (or detectable), else 'local'."""
    if Path("/content/drive/MyDrive").exists() or Path("/content").exists():
        return "colab"
    return "local"


def get_drive_path(*subpath_parts: str) -> Path | None:
    """Return the Drive folder path on Colab, or None on local.

    Args:
        subpath_parts: optional subpath components appended to the Drive root.
            e.g., get_drive_path("Colab Notebooks", "kaggle", "s6e5")
                → /content/drive/MyDrive/Colab Notebooks/kaggle/s6e5
    """
    if detect_environment() != "colab":
        return None
    base = Path("/content/drive/MyDrive")
    if not base.exists():
        return None
    return base.joinpath(*subpath_parts) if subpath_parts else base


def setup_kaggle_auth(
    username: str | None = None,
    api_token: str | None = None,
    from_colab_userdata: bool = True,
) -> bool:
    """Configure Kaggle CLI credentials. Returns True if successful.

    Lookup order:
      1. Explicit username + api_token args
      2. Colab userdata secrets (KAGGLE_USERNAME, KAGGLE_API_TOKEN)
      3. Existing env vars (KAGGLE_USERNAME, KAGGLE_KEY)
      4. Existing ~/.kaggle/kaggle.json

    Auto-unwraps JSON if api_token is the full kaggle.json content.

    Sets os.environ['KAGGLE_USERNAME'] + os.environ['KAGGLE_KEY'] which the
    Kaggle CLI picks up natively.
    """
    if username is not None and api_token is not None:
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = _maybe_unwrap_json_token(api_token)
        return True

    if from_colab_userdata:
        try:
            from google.colab import userdata  # type: ignore
            uname = userdata.get("KAGGLE_USERNAME")
            token = userdata.get("KAGGLE_API_TOKEN")
            if uname and token:
                os.environ["KAGGLE_USERNAME"] = uname
                os.environ["KAGGLE_KEY"] = _maybe_unwrap_json_token(token)
                return True
        except Exception:
            pass

    # Already in env or kaggle.json?
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    if (Path.home() / ".kaggle" / "kaggle.json").exists():
        return True
    return False


def _maybe_unwrap_json_token(token: str) -> str:
    """If token is the full kaggle.json content, extract the 'key' field."""
    token = token.strip()
    if token.startswith("{"):
        try:
            parsed = json.loads(token)
            if isinstance(parsed, dict) and "key" in parsed:
                return parsed["key"]
        except json.JSONDecodeError:
            pass
    return token
