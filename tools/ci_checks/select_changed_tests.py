#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Derive which operator tests/benchmarks a PR should run on the GPU.

This is the FlagTrain analogue of FlagGems' ``tools/test-op.sh`` classification
(the step that maps a diff onto the operator tests it touches), specialised for
FlagTrain's flat operator space. FlagGems fans a changed file out three ways --
``tests/test*.py`` -> functional, ``benchmark/test*`` -> perf, an ops impl file
-> a pytest marker selected with ``-m`` -- because its operators and their test
files do not share a name. Here they do, so the mapping collapses to a 1:1 rule
by filename stem:

    src/flag_train/deepspeed/<op>.py                       -> op (generic = nvidia)
    src/flag_train/runtime/backend/_<v>/ops/deepspeed/<op>.py -> op (vendor override)
    tests/deepspeed/test_<op>.py                           -> op (functional)
    benchmark/deepspeed/test_<op>.py                       -> op (performance)

Outputs (written to GITHUB_OUTPUT when present, otherwise printed):

    run_gpu       true|false   whether the GPU job should run at all
    test_targets  space-separated pytest paths, or __ALL__
    bench_targets space-separated pytest paths, or __ALL__ ("" = no perf for
                  the touched operators, so the perf stage is skipped)
    changed_ops   space-separated operator ids, or "all" for a full run

Decision matrix:

    workflow_dispatch / push / no diff computable  -> full run (__ALL__)
    docs-only change                               -> run_gpu=false
    one or more operators touched                  -> those operators only
    no operator touched but infra changed          -> full run (fail open: an
        infra change can affect every operator)
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DEEPSPEED_DIR = ROOT / "src" / "flag_train" / "deepspeed"
DEEPSPEED_INIT = "src/flag_train/deepspeed/__init__.py"
TESTS_DIR = ROOT / "tests" / "deepspeed"
BENCH_DIR = ROOT / "benchmark" / "deepspeed"

DOC_RE = re.compile(r"\.(md|rst)$|^docs/")
GENERIC_OP_RE = re.compile(r"^src/flag_train/deepspeed/([^/]+)\.py$")
VENDOR_OP_RE = re.compile(
    r"^src/flag_train/runtime/backend/_[^/]+/ops/deepspeed/([^/]+)\.py$"
)
TEST_RE = re.compile(r"^tests/deepspeed/test_(.+)\.py$")
BENCH_RE = re.compile(r"^benchmark/deepspeed/test_(.+)\.py$")


def operator_ids() -> list[str]:
    """Operator inventory, derived from the generic deepspeed impl files.

    The generic implementations are the NVIDIA ones, and every operator there
    has a corresponding ``tests/deepspeed/test_<op>.py``; this list stays in
    sync automatically as operators are added (e.g. evoformer_attn on another
    branch), so nothing here is hard-coded.
    """
    return sorted(p.stem for p in DEEPSPEED_DIR.glob("*.py") if p.stem != "__init__")


def git_diff_files(base: str, head: str) -> list[str]:
    """Fall back to a three-dot git diff when no file list is supplied."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=ACMR", f"{base}...{head}"],
            capture_output=True,
            text=True,
            check=True,
        )
        return [f for f in result.stdout.splitlines() if f.strip()]
    except subprocess.CalledProcessError:
        return []


def existing_paths(ops: set[str], kind: str) -> list[str]:
    """Relative pytest paths that exist for the given operators."""
    base = TESTS_DIR if kind == "test" else BENCH_DIR
    paths = []
    for op in sorted(ops):
        path = base / f"test_{op}.py"
        if path.exists():
            paths.append(str(path.relative_to(ROOT)))
        else:
            print(
                f"::warning::{kind} file {path.relative_to(ROOT)} for changed "
                f"operator '{op}' does not exist; skipping it",
                file=sys.stderr,
            )
    return paths


def decide(
    event: str, changed: list[str], op_ids: list[str]
) -> tuple[str, str, str, str]:
    """Return (run_gpu, test_targets, bench_targets, changed_ops)."""
    # Anything that is not a pull request has no base to scope against, and a
    # merge to main/dev must exercise every operator anyway: run everything.
    if event != "pull_request":
        return "true", "__ALL__", "__ALL__", "all"

    # No diff computed: fail open, run everything.
    if not changed:
        return "true", "__ALL__", "__ALL__", "all"

    ops: set[str] = set()
    docs = 0
    init_changed = False

    for path in changed:
        if DOC_RE.search(path):
            docs += 1
            continue
        if path == DEEPSPEED_INIT:
            init_changed = True
            continue
        m = (
            GENERIC_OP_RE.match(path)
            or VENDOR_OP_RE.match(path)
            or TEST_RE.match(path)
            or BENCH_RE.match(path)
        )
        if m:
            stem = m.group(1)
            if stem == "__init__":
                init_changed = True
            else:
                ops.add(stem)
            continue
        # Anything else (workflow, docker, tools, runtime, backends.yaml,
        # pytest.ini, ...) is infra; it falls through to the fail-open branch.

    # A change whose non-doc files are all absent -> docs-only -> skip.
    if len(changed) - docs == 0:
        return "false", "", "", ""

    # Touching the deepspeed __init__ (the public export surface) can change
    # what any operator resolves to: treat it as touching every operator.
    if init_changed:
        ops = set(op_ids)

    # No operator touched (infra-only, or an unrecognised path): fail open.
    if not ops:
        return "true", "__ALL__", "__ALL__", "all"

    test_targets = " ".join(existing_paths(ops, "test")) or "__ALL__"
    bench_targets = " ".join(existing_paths(ops, "bench"))
    return "true", test_targets, bench_targets, " ".join(sorted(ops))


def emit(name: str, value: str) -> None:
    out_file = os.environ.get("GITHUB_OUTPUT")
    if out_file:
        with open(out_file, "a") as f:
            f.write(f"{name}={value}\n")
    print(f"{name}={value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event-name", default="pull_request", help="GitHub event name"
    )
    parser.add_argument(
        "--changed-files",
        default="",
        help="Space/newline-separated changed file paths (from tj-actions/changed-files).",
    )
    parser.add_argument(
        "--base", default="main", help="Base ref for the git-diff fallback"
    )
    parser.add_argument(
        "--head", default="HEAD", help="Head ref for the git-diff fallback"
    )
    args = parser.parse_args()

    changed = [f for f in args.changed_files.split() if f.strip()]
    if not changed:
        changed = git_diff_files(args.base, args.head)

    run_gpu, test_targets, bench_targets, changed_ops = decide(
        args.event_name, changed, operator_ids()
    )

    print(f"event={args.event_name}")
    print(f"changed files ({len(changed)}): {', '.join(changed[:10]) or '<none>'}")
    print(f"run_gpu={run_gpu}")
    print(f"test_targets={test_targets}")
    print(f"bench_targets={bench_targets}")
    print(f"changed_ops={changed_ops}")

    emit("run_gpu", run_gpu)
    emit("test_targets", test_targets)
    emit("bench_targets", bench_targets)
    emit("changed_ops", changed_ops)


if __name__ == "__main__":
    main()
