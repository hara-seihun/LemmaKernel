"""Run a long sum_free_and_additive search in resumable pieces.

    tools/split_search.py --op ap_free --size 33 --top 136 --depth 2 --budget 40 --state r3_33.json
    tools/split_search.py --op sidon --size 14 --top 126 --depth 2 --budget 40 --state ogr14.json

Counts the `size`-element sets in [0, top] with the property (`sum_free`, `sidon`, `ap_free`;
`--length` for progressions, `--modulus` for Z/n), split by their `depth` smallest elements:
each piece is one `extends_*` request whose context is the prefix, so the pieces partition
the family and their counts add up. Pieces that the span table refutes at the root cost
microseconds, so a small depth (2 or 3) is usually right; the widest pieces are the first ones.

The state file maps each prefix to its count and seconds. A run works through the pieces
still missing until the budget is spent, then reports; rerun with the same arguments to
continue, from any session. `--threads` caps the kernel's threads.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import lemmakernel as lk  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--op", required=True, choices=["sum_free", "sidon", "ap_free"])
    ap.add_argument("--size", type=int, required=True, help="elements per set")
    ap.add_argument("--top", type=int, required=True, help="sets live in [0, top]")
    ap.add_argument("--depth", type=int, default=2, help="prefix length that names a piece")
    ap.add_argument("--budget", type=float, default=40.0, help="seconds of search per run")
    ap.add_argument("--modulus", type=int, default=0)
    ap.add_argument("--length", type=int, default=3)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--state", required=True, help="JSON progress file")
    a = ap.parse_args()
    if not 0 < a.depth < a.size:
        ap.error("--depth must be between 1 and size - 1")

    ctx = lk.Context()
    if a.threads:
        ctx.threads = a.threads
    args = {"modulus": a.modulus}
    if a.op == "ap_free":
        args["length"] = a.length
    path = Path(a.state)
    key = {"op": a.op, "size": a.size, "top": a.top, "depth": a.depth, "modulus": a.modulus, "length": a.length}
    state = json.loads(path.read_text()) if path.exists() else {"search": key, "pieces": {}}
    if state["search"] != key:
        ap.error(f"{path} belongs to another search: {state['search']}")
    pieces = state["pieces"]

    # Every prefix with the property; the kernel lists them in canonical order.
    heads = ctx.value(f"sum_free_and_additive.is_{a.op}", ctx.subsets_of(ctx.range(0, a.top + 1), a.depth),
                      "hits", limit=1 << 40, **args)
    prefixes = [[int(v[0]) for v in row] for row in heads.members.tolist()] if heads.total else []
    prefixes = [p for p in prefixes if a.top - p[-1] >= a.size - a.depth]  # room for the rest
    todo = [p for p in prefixes if json.dumps(p) not in pieces]
    t0 = time.perf_counter()
    for p in todo:
        if time.perf_counter() - t0 > a.budget:
            break
        t1 = time.perf_counter()
        r = ctx.value(f"sum_free_and_additive.extends_{a.op}",
                      ctx.subsets_of(ctx.range(p[-1] + 1, a.top + 1), a.size - a.depth), "count",
                      context=ctx.naturals([p]), **args)
        pieces[json.dumps(p)] = [int(r.value), round(time.perf_counter() - t1, 3)]
        path.write_text(json.dumps(state))
    done = len(pieces)
    seconds = sum(v[1] for v in pieces.values())
    count = sum(v[0] for v in pieces.values())
    status = "complete" if done == len(prefixes) else f"{done}/{len(prefixes)} pieces"
    print(f"{a.op} {a.size}-sets in [0, {a.top}]: {status}, count {count}, {seconds:.1f}s of search")
    return 0 if done == len(prefixes) else 2


if __name__ == "__main__":
    sys.exit(main())
