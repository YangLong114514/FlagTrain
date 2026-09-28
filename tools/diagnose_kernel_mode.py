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
"""Report whether a kernel-mode reading is real, or leaked host dispatch time.

``triton.testing.do_bench`` -- what ``benchmark/`` uses in ``--mode kernel`` --
records an event pair around each ``fn()`` call, after a ``clear_cache`` kernel:

    clear_cache(cache)      # zero the L2 flush buffer
    start_event.record()
    fn()
    end_event.record()

The events are enqueued asynchronously, so the window between them is a
*GPU stream* interval. The flush kernel gives the CPU a head start: while the GPU
is still zeroing, the CPU can enqueue ``fn()``'s launches, and the window then
contains only GPU execution. If ``fn()``'s host-side dispatch is longer than the
flush masks, the queue drains, the GPU idles inside the window, and the reading
becomes host time:

    reading ~= max(real_kernel_time, host_dispatch - effective_masking)

Note that the budget is ``effective_masking``, not the flush kernel's own GPU
time: part of the flush elapses before the start event is enqueued. On the Hygon
BW box a 256 MB flush costs ~220-300us of GPU time but masks only ~140-200us.

So a kernel-mode number is trustworthy only when the flush outlasts the host
dispatch. This script measures the reading at several flush sizes, takes the
converged value as the real kernel time, and reports the configured flush's
reading against it.

Note that ``--mode cudagraph`` is *not* a substitute for the absolute number:
``do_bench_cudagraph`` never flushes L2, so the working set stays cache-resident
and the reading is optimistic -- by ~3x on an A100, less on some other parts.
It is meaningful for ranking two implementations against each other, not as a
kernel time.

Usage:
    python tools/diagnose_kernel_mode.py --op lamb
"""

import argparse
import statistics
import time

import torch

import flag_train
from benchmark import consts


def _host_cost(fn, it=400, warm=50):
    """Wall-clock cost of enqueuing one call, GPU drained afterwards."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(it):
        fn()
    end = time.perf_counter()
    torch.cuda.synchronize()
    return (end - start) / it * 1e6


def _flush_time(mb, device, it=30):
    """GPU time of the L2 flush kernel itself, at a given buffer size."""
    if mb <= 0:
        return 0.0
    buf = torch.empty(mb * 1024 * 1024 // 4, dtype=torch.int, device=device)
    buf.zero_()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(it):
        buf.zero_()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / it * 1e6


def _do_bench_flush(fn, flush_mb, device, n_repeat=60, warm=25):
    """``triton.testing.do_bench``'s loop, with the flush size made explicit."""
    cache = None
    if flush_mb > 0:
        cache = torch.empty(flush_mb * 1024 * 1024 // 4, dtype=torch.int, device=device)
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    for i in range(n_repeat):
        if cache is not None:
            cache.zero_()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in zip(starts, ends)) * 1000


# Past this flush size the reading has stopped falling on every device measured
# (Hygon BW, A100); larger sizes only cost wall-clock.
CONVERGED_MB = 384


def _reading(fn, flush_mb, device, n=3):
    """A do_bench reading, median of ``n`` runs.

    Single readings on this box scatter by ~20% at small sizes, which is enough
    to make an operator look inflated when it is not (or the reverse), so the
    number this script reports is never one sample.
    """
    return statistics.median(_do_bench_flush(fn, flush_mb, device) for _ in range(n))


def diagnose(fn, label, device, flush_sizes):
    host = _host_cost(fn)
    rows = []
    for mb in flush_sizes:
        ft = _flush_time(mb, device)
        rows.append((mb, ft, _reading(fn, mb, device)))

    # The real kernel time is the reading once the flush is past convergence:
    # there the window is GPU-only, so what it contains is the kernel.  Taking a
    # median over the converged sizes rather than one sample keeps clock noise
    # from deciding whether an operator looks inflated.
    converged = [rd for mb, _, rd in rows if mb >= CONVERGED_MB]
    plateau = statistics.median(converged) if converged else rows[-1][2]

    bench_mb = consts.DEFAULT_L2_FLUSH_MB
    bench = next((r for r in rows if r[0] == bench_mb), None)
    if bench is None:
        bench = (
            bench_mb,
            _flush_time(bench_mb, device),
            _reading(fn, bench_mb, device),
        )
        rows.append(bench)
        rows.sort()

    print(f"\n=== {label} ===")
    print("  host dispatch per call   : %10.2f us  (what has to be masked)" % host)
    print(
        "  real kernel time (cold L2): %8.2f us  (median of flush >= %d MB)"
        % (plateau, CONVERGED_MB)
    )
    print(
        "  benchmark reading        : %10.2f us  (at %d MB, the configured flush)"
        % (bench[2], bench_mb)
    )
    if plateau > 0:
        factor = bench[2] / plateau
        print(
            "  exaggeration factor      : %10.1fx  %s"
            % (
                factor,
                (
                    "OK, reading is kernel time"
                    if factor < 1.15
                    else "INFLATED: reading is host time, not kernel time"
                ),
            )
        )
        # Effective masking, inferred rather than assumed: the flush's own GPU
        # time is not the budget, because part of it elapses before the start
        # event is enqueued.  ``reading = max(kernel, host - masking)`` makes
        # ``host - reading`` an equality while the op leaks and a lower bound
        # once it is masked, so it is reported as the bound it always is.
        print(
            "  masking (lower bound)    : %10.2f us  (= host - reading)"
            % (host - bench[2])
        )

    print("\n  flush_mb   flush_us   reading_us")
    for mb, ft, rd in rows:
        mark = "  <- benchmark's flush" if mb == bench_mb else ""
        print("  %6d   %8.1f   %10.2f%s" % (mb, ft, rd, mark))
    return host, plateau, bench[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--op", default="lamb", choices=["lamb"])
    ap.add_argument(
        "--shape",
        type=int,
        default=1024,
        help="flattened parameter count; small sizes expose host cost",
    )
    ap.add_argument(
        "--flushes", type=int, nargs="+", default=[0, 64, 256, 384, 512, 768]
    )
    args = ap.parse_args()
    if consts.DEFAULT_L2_FLUSH_MB not in args.flushes:
        args.flushes = sorted(set(args.flushes) | {consts.DEFAULT_L2_FLUSH_MB})

    from benchmark.deepspeed.test_lamb import lamb_input_fn, torch_op, train_op

    device = flag_train.device
    print(f"vendor={flag_train.vendor_name} device={device} shape=({args.shape},)")

    tensors = next(iter(lamb_input_fn((args.shape,), torch.float32, device)))
    cases = (("baseline (torch_op)", torch_op), ("under test (train_op)", train_op))
    results = {}
    for label, op in cases:
        call = lambda op=op: op(*tensors)  # noqa: E731
        results[label] = diagnose(call, label, device, args.flushes)

    _, base_kernel, base_reading = results["baseline (torch_op)"]
    _, test_kernel, test_reading = results["under test (train_op)"]
    print("\n=== verdict ===")
    print(
        "  real kernel speedup      : %.3fx  (trustworthy)"
        % (base_kernel / test_kernel)
    )
    print(
        "  kernel-mode speedup      : %.3fx  (what the benchmark reports)"
        % (base_reading / test_reading)
    )
    for label, (host, kernel, reading) in results.items():
        if kernel > 0 and reading > kernel * 1.15:
            print(
                f"  WARNING: {label} reads {reading / kernel:.1f}x its real kernel "
                f"time -- {host:.0f} us of host dispatch is leaking into the "
                f"{consts.DEFAULT_L2_FLUSH_MB} MB flush window."
            )


if __name__ == "__main__":
    main()
