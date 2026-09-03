#!/usr/bin/env python3
"""Generate everything that derives from module manifests, or check that it is current.

    tools/manifest.py generate     rewrite runtime/generated/*, python/lemmakernel/_manifest.py, lakefile.toml, BENCHMARKS.md
    tools/manifest.py check        exit 1 if any generated file is stale or a module is incomplete

modules/<name>/manifest.toml is the single declaration of a module. This script is the only
producer of the C++ registry data, the runtime description JSON, the CMake source list, the Lake
configuration and the Python-side manifest.
"""
from __future__ import annotations

import json
import pprint
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODULES = ROOT / "modules"
RUNTIME_MANIFEST = ROOT / "runtime" / "manifest.toml"
GENERATED_CPP = ROOT / "runtime" / "generated" / "manifest.cpp"
GENERATED_CMAKE = ROOT / "runtime" / "generated" / "modules.cmake"
GENERATED_BENCH = ROOT / "BENCHMARKS.md"
GENERATED_PY = ROOT / "python" / "lemmakernel" / "_manifest.py"
GENERATED_LAKE = ROOT / "lakefile.toml"

REQUIRED_FILES = ("lean", "contract", "reference", "naive", "cases")
ARG_TYPES = {"int", "vector", "vectors", "perms", "group", "family"}
FEATURE_FLAGS = {
    "avx512bw": ["-mavx512f", "-mavx512bw", "-mavx512vl", "-mavx512dq"],
}


def load_modules() -> list[dict]:
    out = []
    for manifest in sorted(MODULES.glob("*/manifest.toml")):
        data = tomllib.loads(manifest.read_text())
        data["_dir"] = manifest.parent
        if data["module"]["name"] != manifest.parent.name:
            raise SystemExit(f"{manifest}: module name {data['module']['name']!r} != directory {manifest.parent.name!r}")
        out.append(data)
    return out


def load_runtime() -> dict:
    return tomllib.loads(RUNTIME_MANIFEST.read_text())


def validate(runtime: dict, modules: list[dict]) -> list[str]:
    problems = []
    reductions = {r["name"]: r for r in runtime.get("reductions", [])}
    for m in modules:
        if "reductions" in m:
            problems.append(f"{m['module']['name']}: reductions are declared in runtime/manifest.toml, not per module")
        name = m["module"]["name"]
        for key in REQUIRED_FILES:
            rel = m["module"].get(key)
            if not rel:
                problems.append(f"{name}: manifest lacks module.{key}")
            elif not (m["_dir"] / rel).exists():
                problems.append(f"{name}: {key} file {rel} does not exist")
        lean_root = m["_dir"] / m["module"].get("lean", "lean") / (lean_lib(name) + ".lean")
        if not lean_root.exists():
            problems.append(f"{name}: Lean root module {lean_root.relative_to(ROOT)} does not exist")
        for b in m.get("backends", []):
            for src in b["sources"]:
                if not (m["_dir"] / src).exists():
                    problems.append(f"{name}: backend {b['name']} source {src} does not exist")
            for feature in b.get("requires", []):
                if feature not in FEATURE_FLAGS:
                    problems.append(f"{name}: backend {b['name']} requires unknown feature {feature}")
        if "families" in m:
            problems.append(f"{name}: families are declared in runtime/manifest.toml, not per module")
        kinds = ({k["name"] for k in runtime.get("kinds", [])} |
                 {k["name"] for mm in modules for k in mm.get("kinds", [])})
        values = kinds | {"integer", "boolean"}
        family_names = {f["name"] for f in runtime.get("families", [])}
        op_names = set()
        for op in m.get("operations", []):
            op_names.add(op["name"])
            for fam in op.get("families", []):
                if fam not in family_names:
                    problems.append(f"{name}: operation {op['name']} restricts to unknown family {fam}")
            if op["value"] not in values:
                problems.append(f"{name}: operation {op['name']} has unknown value type {op['value']}")
            kind_decls = runtime.get("kinds", []) + [k for mm in modules for k in mm.get("kinds", [])]
            if op["value"] in kinds and not any(k.get("lean") for k in kind_decls if k["name"] == op["value"]):
                problems.append(f"{name}: operation {op['name']} produces kind {op['value']}, which declares no Lean value constructor")
            if not any(op["value"] in r["accepts"] or "*" in r["accepts"] for r in reductions.values()):
                problems.append(f"{name}: no reduction accepts operation {op['name']}")
            for arg, typ in arg_items(op):
                if typ not in ARG_TYPES:
                    problems.append(f"{name}: operation {op['name']} argument {arg} has unknown type {typ}")
        for rej in m.get("rejections", []):
            if "case" not in rej or "error" not in rej:
                problems.append(f"{name}: every rejection needs `case` and `error`")
            if "op" in rej and rej["op"] not in op_names:
                problems.append(f"{name}: rejection names unknown operation {rej['op']}")
    for r in runtime.get("reductions", []):
        for arg, typ in arg_items(r):
            if typ not in ARG_TYPES:
                problems.append(f"runtime: reduction {r['name']} argument {arg} has unknown type {typ}")
    for f in runtime.get("families", []):
        if not f.get("lean"):
            problems.append(f"runtime: family {f['name']} declares no Lean constructor")
    return problems


def arg_items(decl: dict) -> list[tuple[str, str]]:
    """(name, type) pairs of an operation's or reduction's arguments, sorted by name.

    Sorted, not as declared: this is the order every generated file and the Lean claim renderer
    use, so it is also the order a reference's constructor takes its arguments in. `validate`
    asks manifests to declare them that way so that the file reads the way it behaves."""
    args = decl.get("args", {})
    if isinstance(args, list):
        raise SystemExit(f"arguments must be typed: args = {{ name = \"type\" }} (got {args})")
    return sorted(args.items())


def lean_lib(module: str) -> str:
    return module[:1].upper() + module[1:]


def render_lakefile(modules: list[dict]) -> str:
    lines = ["# Generated by tools/manifest.py from modules/*/manifest.toml. Do not edit.",
             "# Lk is the runtime's reference (families, reductions) and contract; then one library",
             "# per module: its executable reference and contract (against Mathlib).",
             'name = "LemmaKernel"',
             f"defaultTargets = {json.dumps(['Lk'] + [lean_lib(m['module']['name']) for m in modules])}",
             "", "[[require]]", 'name = "mathlib"', 'scope = "leanprover-community"', 'rev = "v4.33.0"',
             "", "[[lean_lib]]", 'name = "Lk"', 'srcDir = "runtime/lean"', 'roots = ["Lk"]']
    for m in modules:
        name = m["module"]["name"]
        src = (m["_dir"] / m["module"]["lean"]).relative_to(ROOT).as_posix()
        lines += ["", "[[lean_lib]]", f'name = "{lean_lib(name)}"', f'srcDir = "{src}"', f'roots = ["{lean_lib(name)}"]']
    return "\n".join(lines) + "\n"


def cstr(s: str) -> str:
    return json.dumps(s)


def c_array(name: str, items: list[str]) -> str:
    body = ", ".join(cstr(i) for i in items)
    return f"static const char *const {name}[] = {{{body}{', ' if items else ''}nullptr}};\n"


def render_cpp(runtime: dict, modules: list[dict]) -> str:
    lines = ["// Generated by tools/manifest.py from modules/*/manifest.toml. Do not edit.",
             '#include "../src/registry.hpp"', "", "namespace lk {", "namespace {", ""]
    ops, reds, backs = [], [], []
    for m in modules:
        mod = m["module"]["name"]
        for op in m.get("operations", []):
            arr = f"args_{mod}_{op['name']}"
            lines.append(c_array(arr, [a for a, _ in arg_items(op)]))
            fams = f"families_{mod}_{op['name']}"
            lines.append(c_array(fams, op.get("families", [])))
            ops.append(f"    {{{cstr(mod)}, {cstr(op['name'])}, {cstr(op['value'])}, {arr}, {fams}}},")
        for b in m.get("backends", []):
            req = f"requires_{mod}_{b['name']}"
            lines.append(c_array(req, b.get("requires", [])))
            backs.append(f"    {{{cstr(mod)}, {cstr(b['name'])}, {req}}},")
    for r in runtime.get("reductions", []):
        acc = f"accepts_{r['name']}"
        arg = f"redargs_{r['name']}"
        lines.append(c_array(acc, r["accepts"]))
        lines.append(c_array(arg, [a for a, _ in arg_items(r)]))
        reds.append(f"    {{{cstr(r['name'])}, {acc}, {arg}}},")
    lines += ["} // namespace", "",
              "const std::vector<ManifestOperation> &manifest_operations() {",
              "    static const std::vector<ManifestOperation> v{", *ops, "    };", "    return v;", "}", "",
              "const std::vector<ManifestReduction> &manifest_reductions() {",
              "    static const std::vector<ManifestReduction> v{", *reds, "    };", "    return v;", "}", "",
              "const std::vector<ManifestBackend> &manifest_backends() {",
              "    static const std::vector<ManifestBackend> v{", *backs, "    };", "    return v;", "}", ""]
    public = []
    for m in modules:
        entry = {k: v for k, v in m.items() if not k.startswith("_")}
        public.append(entry)
    describe = json.dumps({"runtime": runtime, "modules": public}, indent=None, sort_keys=True)
    lines += ["const char *manifest_describe_json() {", f"    return {cstr(describe)};", "}", "", "} // namespace lk", ""]
    return "\n".join(lines)


def render_cmake(modules: list[dict]) -> str:
    lines = ["# Generated by tools/manifest.py from modules/*/manifest.toml. Do not edit.", ""]
    for m in modules:
        mod = m["module"]["name"]
        for b in m.get("backends", []):
            for src in b["sources"]:
                path = (m["_dir"] / src).relative_to(ROOT).as_posix()
                lines.append(f"target_sources(lemmakernel PRIVATE ${{CMAKE_SOURCE_DIR}}/{path})")
                flags = [f for feat in b.get("requires", []) for f in FEATURE_FLAGS[feat]]
                if flags:
                    lines.append(f"set_source_files_properties(${{CMAKE_SOURCE_DIR}}/{path} PROPERTIES COMPILE_OPTIONS \"{';'.join(flags)}\")")
        lines.append(f"# module {mod}: backends {[b['name'] for b in m.get('backends', [])]}")
    return "\n".join(lines) + "\n"


def render_python(runtime: dict, modules: list[dict]) -> str:
    public = [{k: v for k, v in m.items() if not k.startswith("_")} for m in modules]
    return ("# Generated by tools/manifest.py from runtime/manifest.toml and modules/*/manifest.toml. Do not edit.\n"
            f"RUNTIME = {pprint.pformat(runtime, width=110, sort_dicts=True)}\n"
            f"MODULES = {pprint.pformat(public, width=110, sort_dicts=True)}\n")


def _seconds(s: float, extrapolated: bool = False) -> str:
    text = f"{s:.3g}" if s < 100 or s >= 1e7 else f"{s:,.0f}"
    return text + " s" + ("~" if extrapolated else "")


def _ratio(x: float) -> str:
    return f"{x:.2g}×" if x >= 1e7 else f"{x:,.0f}×" if x >= 10 else f"{x:.1f}×"


def render_benchmarks(modules: list[dict]) -> str:
    records = []
    missing = []
    for m in modules:
        p = ROOT / "modules" / m["module"]["name"] / "bench.json"
        (records if p.exists() else missing).append(json.loads(p.read_text()) if p.exists() else m["module"]["name"])
    hosts = sorted({(r["machine"]["host"], r["machine"]["cpu"], r["machine"]["threads"]) for r in records})
    out = ["# Benchmarks", "",
           "Generated by `tools/manifest.py generate` from every `modules/*/bench.json`, which `tools/bench.py` "
           "writes; do not edit. Each row is the module's generic backend against its naive Python implementation "
           "on the same family, with byte-for-byte agreement of the answers. A `~` marks a naive time "
           "extrapolated from a prefix of members. A module's record is rerun when its sources, the modules it "
           "includes, or the runtime change (`tools/bench.py --status`).", ""]
    for host, cpu, threads in hosts:
        out.append(f"Measured on `{host}` ({cpu}, {threads} threads).")
    out += ["", "## Best single-thread ratio per module", "",
            "| module | case | members | naive | kernel ×1 | speedup ×1 | speedup all threads |",
            "|---|---|---:|---:|---:|---:|---:|"]
    best = [(r["module"], max(r["rows"], key=lambda x: x["speedup_1_thread"])) for r in records if r["rows"]]
    for name, r in sorted(best, key=lambda x: -x[1]["speedup_1_thread"]):
        out.append(f"| {name} | {r['case']} | {r['members']:,} | {_seconds(r['naive_s'], r['naive_extrapolated'])} | "
                   f"{_seconds(r['kernel_1_thread_s'])} | {_ratio(r['speedup_1_thread'])} | {_ratio(r['speedup_all_threads'])} |")
    out += ["", "## Every bench case", "",
            "| module | case | what | members | naive | kernel ×1 | kernel ×all | speedup ×1 | speedup ×all |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for rec in sorted(records, key=lambda r: r["module"]):
        for r in rec["rows"]:
            out.append(f"| {rec['module']} | {r['case']} | {r['what']} | {r['members']:,} | "
                       f"{_seconds(r['naive_s'], r['naive_extrapolated'])} | {_seconds(r['kernel_1_thread_s'])} | "
                       f"{_seconds(r['kernel_all_threads_s'])} | {_ratio(r['speedup_1_thread'])} | {_ratio(r['speedup_all_threads'])} |")
    without = sorted([r["module"] for r in records if not r["rows"]])
    if without:
        out += ["", "Modules with a record but no bench case: " + ", ".join(f"`{n}`" for n in without) + "."]
    if missing:
        out += ["", "Modules without a bench record yet: " + ", ".join(f"`{n}`" for n in missing) + "."]
    return "\n".join(out) + "\n"


def outputs(runtime: dict, modules: list[dict]) -> dict[Path, str]:
    return {GENERATED_CPP: render_cpp(runtime, modules), GENERATED_CMAKE: render_cmake(modules),
            GENERATED_PY: render_python(runtime, modules), GENERATED_LAKE: render_lakefile(modules),
            GENERATED_BENCH: render_benchmarks(modules)}


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) > 1 else "check"
    runtime = load_runtime()
    modules = load_modules()
    problems = validate(runtime, modules)
    if problems:
        print("\n".join(problems))
        return 1
    files = outputs(runtime, modules)
    if mode == "generate":
        for path, text in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        print(f"generated {len(files)} files for {len(modules)} module(s)")
        return 0
    if mode == "check":
        stale = [str(p.relative_to(ROOT)) for p, text in files.items() if not p.exists() or p.read_text() != text]
        if stale:
            print("stale generated files (run tools/manifest.py generate): " + ", ".join(stale))
            return 1
        print(f"manifests consistent: {', '.join(m['module']['name'] for m in modules)}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
