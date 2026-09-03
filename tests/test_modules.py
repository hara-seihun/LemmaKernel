"""Repository-wide checks: every module is complete and every derived file is current."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_manifests_are_consistent_and_generated_files_current():
    proc = subprocess.run([sys.executable, str(ROOT / "tools" / "manifest.py"), "check"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_every_module_contract_compiles():
    """`lake build` covers each module's Reference and Contract (the latter against Mathlib)."""
    proc = subprocess.run(["lake", "build"], cwd=ROOT, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
