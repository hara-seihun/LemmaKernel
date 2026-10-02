"""Failed candidate smokes must leave the live release untouched."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import lemmakernel as lk  # noqa: E402


def test_failed_smoke_preserves_current(tmp_path):
    dest = tmp_path / "releases"
    live = dest / "working"
    live.mkdir(parents=True)
    (live / "COMMIT").write_text("working\n")
    (dest / "current").symlink_to(live.name)
    build = tmp_path / "build"
    build.mkdir()
    (build / "liblemmakernel.so").symlink_to(Path(lk._lib._name).resolve())
    commands = tmp_path / "bin"
    commands.mkdir()
    # Reuse the test suite's built library; this test is about publication, not compilation.
    for name in ("cmake", "ninja"):
        stub = commands / name
        stub.write_text("#!/bin/sh\nexit 0\n")
        stub.chmod(0o755)
    python = commands / "python"
    python.write_text(f'''#!{sys.executable}
import os
import sys
if sys.argv[1] == "-c":
    assert "/.stage." in os.environ["PYTHONPATH"]
    assert os.environ["LEMMAKERNEL_LIB"] == ""
    sys.exit(73)
os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
''')
    python.chmod(0o755)
    proc = subprocess.run([str(ROOT / "deploy")], env=dict(
        os.environ, PATH=f"{commands}:{os.environ['PATH']}", PYTHON=str(python),
        LEMMAKERNEL_BUILD_DIR=str(build), LEMMAKERNEL_DEPLOY_ROOT=str(dest),
        LEMMAKERNEL_BUILD_SHARD=""), capture_output=True, text=True, timeout=20)
    assert proc.returncode == 73, proc.stdout + proc.stderr
    assert (dest / "current").readlink() == Path("working")
    assert (dest / "current" / "COMMIT").read_text() == "working\n"
    assert set(dest.iterdir()) == {live, dest / "current"}
