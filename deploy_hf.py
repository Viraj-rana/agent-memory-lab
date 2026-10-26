"""Hugging Face Gradio/Docker Spaces currently require PRO (402 on free CPU).

Prefer Streamlit Community Cloud instead — see README.md.
This script is kept only for users with HF PRO.
"""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "Hugging Face free tier no longer hosts Gradio/Docker Spaces (HTTP 402).\n"
        "Use Streamlit Community Cloud:\n"
        "  1) Push this repo to GitHub\n"
        "  2) https://share.streamlit.io → New app → streamlit_app.py\n"
        "Author: Viraj Rana (ML Engineer)",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
