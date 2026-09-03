"""Check claims against a module's Lean reference with `decide +kernel`.

A test does not store expected answers. It runs the kernel on an input, states
`example : <reference term> = <kernel answer> := by decide +kernel`, and asks Lean to accept
the file. The Lean kernel evaluates the reference definition; if the kernel's answer differs,
the example fails and the test fails. The expected output therefore lives nowhere but in the
reference definition itself.

    from tools.leancheck import LeanCheck
    lc = LeanCheck("gfp_small", imports=["Gfp.Reference"], opens=["Gfp", "Lk"])
    lc.claim("run .rank (.grassmannian 2 4 2) .histogram", ".histogram 35 [0, 0, 35]", label="G(2,4,2) rank")
    lc.verify()          # raises AssertionError listing the failed claims

Each claim becomes one `example`. `verify` runs `lean` once per LeanCheck (the toolchain and
search path are resolved from `lake env` once per process, since that alone costs 0.45 s; Lean
itself starts in ~0.1 s). Kernel checking is single-threaded within a process, so parallelism
comes from many small files, not one big one. Kernel evaluation costs roughly 10 ms per matrix
member in gfp; keep oracle inputs to a few dozen members.
"""
from __future__ import annotations

import functools
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "leancheck"


@functools.cache
def lean_command() -> tuple[str, dict]:
    """The `lean` binary and environment that `lake env lean` would use, resolved once."""
    env = dict(os.environ)
    out = subprocess.run(["lake", "env", "env", "-0"], cwd=ROOT, capture_output=True, text=True).stdout
    for entry in out.split("\0"):
        if entry.startswith(("LEAN", "PATH=")):
            key, _, value = entry.partition("=")
            env[key] = value
    return env.get("LEAN", "lean"), env


class LeanCheck:
    def __init__(self, name: str, imports: list[str], opens: list[str] = ()):
        self.name = name
        self.imports = list(imports)
        self.opens = list(opens)
        self.claims: list[tuple[str, str, str]] = []

    def claim(self, lhs: str, rhs: str, label: str = "") -> None:
        self.claims.append((lhs, rhs, label or f"claim {len(self.claims)}"))

    def render(self) -> str:
        lines = [f"import {m}" for m in self.imports]
        if self.opens:
            lines.append("open " + " ".join(self.opens))
        lines.append("set_option maxRecDepth 1000000")
        lines.append("")
        for i, (lhs, rhs, label) in enumerate(self.claims):
            lines.append(f"-- [{i}] {label}")
            lines.append(f"example : {lhs} = {rhs} := by decide +kernel")
            lines.append("")
        return "\n".join(lines)

    def verify(self, timeout: float = 600) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / f"{self.name}.lean"
        text = self.render()
        path.write_text(text)
        lean, env = lean_command()
        proc = subprocess.run([lean, "--tstack=1000000", str(path)], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=timeout)
        if proc.returncode == 0 and not proc.stdout.strip():
            return
        # Map error lines back to claim labels.
        line_to_label = {}
        for n, line in enumerate(text.splitlines(), start=1):
            m = re.match(r"-- \[(\d+)\] (.*)", line)
            if m:
                line_to_label[n + 1] = m.group(2)
        failures = []
        for line in (proc.stdout + proc.stderr).splitlines():
            m = re.match(rf"{re.escape(str(path))}:(\d+):\d+: (.*)", line)
            if m:
                failures.append(f"{line_to_label.get(int(m.group(1)), '?')}: {m.group(2)}")
        raise AssertionError(f"Lean rejected {len(failures)} claim(s) in {path}:\n" + "\n".join(failures)
                             + ("\n" + (proc.stdout + proc.stderr)[-2000:] if not failures else ""))
