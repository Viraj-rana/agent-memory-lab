"""Create/update the Hugging Face Space for Agent Memory Lab.

Usage (PowerShell):
  $env:HF_TOKEN = "hf_..."   # write-access token from huggingface.co/settings/tokens
  .\.venv\Scripts\python deploy_hf.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, login, whoami

ROOT = Path(__file__).resolve().parent
SPACE_NAME = "agent-memory-lab"
AUTHOR = "Viraj Rana"

INCLUDE = [
    "app.py",
    "requirements.txt",
    "README.md",
    "LICENSE",
    "AUTHORS.md",
    "Dockerfile",
    "pyproject.toml",
    "src",
    "data",
]


def main() -> int:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print(
            "Missing HF_TOKEN.\n"
            "1) Create a token (Write) at https://huggingface.co/settings/tokens\n"
            "2) $env:HF_TOKEN = 'hf_...'\n"
            "3) python deploy_hf.py",
            file=sys.stderr,
        )
        return 1

    login(token=token, add_to_git_credential=False)
    info = whoami()
    username = info["name"]
    repo_id = f"{username}/{SPACE_NAME}"
    api = HfApi(token=token)

    print(f"Deploying as {AUTHOR} → https://huggingface.co/spaces/{repo_id}")
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        private=False,
        exist_ok=True,
    )

    allow = []
    for item in INCLUDE:
        path = ROOT / item
        if path.is_file():
            allow.append(item)
        elif path.is_dir():
            allow.append(f"{item}/**")

    api.upload_folder(
        folder_path=str(ROOT),
        repo_id=repo_id,
        repo_type="space",
        allow_patterns=allow,
        ignore_patterns=[
            ".venv/**",
            ".git/**",
            "__pycache__/**",
            ".pytest_cache/**",
            "notebooks/**",
            "tests/**",
            "deploy_hf.py",
            "spaces_readme.md",
            "*.png",
            "PR_BODY.md",
        ],
    )

    url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"\nPermanent recruiter link:\n{url}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
