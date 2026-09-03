"""Repository-wide checks: every module is complete and every derived file is current."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_manifests_are_consistent_and_generated_files_current():
    proc = subprocess.run([sys.executable, str(ROOT / "tools" / "manifest.py"), "check"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_every_module_bench_record_is_current():
    """A module's bench.json fingerprints the module tree and the modules it includes. Touching
    either without rerunning `tools/bench.py --module NAME` leaves numbers nobody measured."""
    sys.path.insert(0, str(ROOT))
    from tools import harness as H
    stale = [f"{m.name} ({why})" for m in H.modules() for why in [H.bench_staleness(m)]
             if why is not None and why != "runtime changed"]
    assert not stale, "stale bench records, run tools/bench.py --module NAME for: " + ", ".join(stale)


def test_every_module_contract_compiles():
    """`lake build` covers each module's Reference and Contract (the latter against Mathlib)."""
    proc = subprocess.run(["lake", "build"], cwd=ROOT, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]


def test_split_search_pieces_add_up(tmp_path):
    """`tools/split_search.py` partitions a search by prefix and resumes from its state file;
    the pieces must sum to the direct count (here the 4 optimal Golomb rulers with 5 marks)."""
    args = [sys.executable, str(ROOT / "tools" / "split_search.py"), "--op", "sidon", "--size", "5", "--top", "11",
            "--depth", "2", "--state", str(tmp_path / "s.json")]
    first = subprocess.run(args + ["--budget", "0"], capture_output=True, text=True, timeout=120)
    assert first.returncode == 2, first.stdout + first.stderr
    rest = subprocess.run(args + ["--budget", "60"], capture_output=True, text=True, timeout=120)
    assert rest.returncode == 0, rest.stdout + rest.stderr
    assert "complete, count 4," in rest.stdout, rest.stdout
