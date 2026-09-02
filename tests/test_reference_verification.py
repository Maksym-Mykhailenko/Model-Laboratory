from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_reference_verification_suite_passes() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "verification" / "run_reference_verification.py")],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    assert "Reference verification: 49/49 passed" in completed.stdout
