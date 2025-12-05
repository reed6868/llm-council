import os
from pathlib import Path


def test_start_sh_forwards_cli_arguments():
    """start.sh should forward any CLI args to backend.cli."""
    script = Path("start.sh")
    assert script.exists(), "start.sh should exist at project root"

    text = script.read_text(encoding="utf-8")

    # The script should invoke backend.cli and forward arguments.
    assert "uv run python -m backend.cli" in text
    assert '"$@"' in text or "$@" in text

