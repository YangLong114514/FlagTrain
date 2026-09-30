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
"""Hygon-tuned blocked flash attention.

The generic implementation is ``flag_train.deepspeed.blocked_flash`` and is what
every other backend runs. What is kept here is the part of that operator's tuning
that only pays on Hygon's parts, because the same code costs the NVIDIA ones
measurably: capping the KV tile at 32 is worth a second resident program per
compute unit at Hygon's 64 KB shared-memory budget, and costs a second
online-softmax update per cache block everywhere. On an A100 that single
difference moves the benchmark average from 0.92 to 0.81.

Four things differ from the generic kernel:

* a narrower KV tile (32 against 64), with the shared-memory budget read from the
  device rather than assumed, and a clear error when even the narrowest tile does
  not fit;
* ``num_warps`` sized by *threads* rather than warps -- Hygon's warps are 64
  threads against NVIDIA's 32, so the same count would be twice the CTA;
* the K tile loaded pre-transposed, which avoids a shared-memory layout
  conversion inside the KV loop. Worth ~79% of the kernel's time on Hygon; on an
  A100 Triton already folds the transpose and it measures as nothing;
* a flat 1-D grid with the head index varying fastest and the atoms numbered
  widest-first, worth 20-30% on Hygon's large shapes.

Everything else shared is imported from the generic module so the two cannot
drift: the atom-field constants (``_ATOM_*``, which are the kernel's view of the
layout), ``_LOG2E``, and ``_check_launch_arguments`` -- the input checks are one
implementation used by both, so a backend cannot accept something another
rejects. ``AtomBuilder`` and ``RaggedBatchWrapper`` are *not* imported: they live
in the test files now, and neither implementation of this kernel calls them.

Registration is by name. ``_hygon/ops/deepspeed/__init__.py`` re-exports this
module's ``blocked_flash``, and ``SpecOpRegistrar`` writes it over the same-named
generic function in ``flag_train.deepspeed`` when that package is imported -- the
namespace the tests and the benchmark read the operator from.
"""

import functools
import logging

import triton
import triton.language as tl

from flag_train.deepspeed.blocked_flash import (
    _ATOM_BLOCK_OFFSET,
    _ATOM_GLOBAL_Q_IDX,
    _ATOM_KV_BLOCKS,
    _ATOM_Q_LEN,
    _ATOM_Q_START,
    _ATOM_STRIDE,
    _ATOM_TOTAL_EXTENT,
    _LOG2E,
    _check_launch_arguments,
)
from flag_train.utils import libentry

logger = logging.getLogger(__name__)


def _shared_memory_for(block_m, block_n, block_d, element_size):
    """Shared memory one program needs, in bytes, for a given tiling.

    Triton's allocator decides the real figure -- it reuses buffers whose live
    ranges do not overlap and stages the pipelined KV loop -- so this is not
    derived from the kernel's source but measured against it: it reproduces the
    allocation exactly for every tiling this kernel can be launched with on the
    devices checked, and never reports below it. That is what makes it usable as
    a limit test; an approximation that under-reported would let a launch
    through to fail deep inside the compiler.

    The two terms are the query tile and the KV loop's live set -- a key tile, a
    value tile, and the compute-dtype copy of the scores the second dot consumes
    -- and the allocator needs room for whichever is larger.
    """
    return (
        max(block_m * block_d, 2 * block_n * block_d + block_m * block_n) * element_size
    )


def _widest_query_tile(block_d, element_size, limit):
    """The largest power-of-two query tile a device can hold, for an advisory.

    Used only to tell the caller what ``q_block_size`` to retry with when even the
    narrowest KV tile does not fit; it is the bare query tile at ``block_n = 1``,
    so it is an upper bound on what will actually launch.
    """
    block_m = 1
    while _shared_memory_for(block_m * 2, 1, block_d, element_size) <= limit:
        block_m *= 2
    return block_m


@functools.lru_cache(maxsize=None)
def _device_properties(device_index):
    """The device's property dict, or ``None`` if it cannot be read.

    Cached, and it has to be: the driver call costs about **1 ms**, against
    kernels that run in tens of microseconds, and this host function asks for it
    twice per invocation. Uncached it does not size the tiling, it *is* the
    timing -- 0.92 average speedup down to 0.14, measured on an A100. Device
    properties do not change under a running process, so nothing to invalidate.
    """
    try:
        from triton.runtime.driver import driver as triton_driver

        return triton_driver.active.utils.get_device_properties(device_index)
    except Exception:
        return None


def _max_shared_memory(device_index):
    """The device's shared-memory limit in bytes, or ``None`` if unreadable.

    ``None`` means "cannot tell", and the caller then leaves the tiling alone
    rather than guessing a limit -- the same behaviour as before this was
    queryable at all.
    """
    props = _device_properties(device_index)
    limit = props.get("max_shared_mem") if props else None
    return int(limit) if limit else None


def _warp_size(device_index):
    """Threads per warp, or 32 when the device will not say.

    A warp is not a fixed width across devices -- 32 on the NVIDIA parts this
    was first tuned on, 64 here -- and the per-CTA sizing below is a count of
    *threads*, so it has to know which. 32 is the value that sizing already
    assumed before it could ask, so an unreadable device keeps that behaviour
    rather than getting a new one.
    """
    props = _device_properties(device_index)
    size = props.get("warpSize") if props else None
    return int(size) if size else 32


@libentry()
@triton.jit
def _blocked_flash_fwd_kernel(
    out_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    atoms_ptr,
    block_idx_ptr,
    qk_scale,
    q_row_stride,
    o_row_stride,
    k_row_stride,
    h_h_k_ratio,
    n_heads_q,
    IS_CAUSAL: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Which (atom, head) pair this program runs is a scheduling decision, and it
    # is worth a quarter of the kernel: the programs are dispatched in linear-id
    # order, so the order they are numbered in decides which of them run at the
    # same time. Two effects, both measured together at 20-30% on every large
    # shape, and both order-only -- permuting independent programs cannot change
    # a result:
    #
    #   * heads vary fastest. The programs resident at any moment are then the
    #     same atom on different heads, which read the same cache blocks; inside
    #     a block the heads are contiguous, so memory sees one sequential stream
    #     instead of ``n_heads_q`` scattered ones.
    #   * atoms run from the widest to the narrowest. An atom's work grows with
    #     its index -- under the causal single-sequence layout atom *i* walks
    #     2(i+1) cache blocks -- so numbering them backwards hands the long
    #     programs to the device first and leaves it the short ones to drain on.
    #     That ordering is the one the benchmark's atoms are built in; for atoms
    #     from anywhere else the reversal is an arbitrary permutation, which is
    #     why it is safe to apply unconditionally.
    #
    # The flat 1-D grid is what lets both hold at once. A 2-D ``(heads, atoms)``
    # grid would need ``num_atoms`` to fit in grid dimension 1, which caps a
    # batch at 65535 atoms; flattening keeps the fast-varying axis under our
    # control while leaving the grid bound at 2**31 programs, as the original
    # ``(atoms, heads)`` grid had.
    pid = tl.program_id(0)
    hq = pid % n_heads_q
    atom = tl.num_programs(0) // n_heads_q - 1 - pid // n_heads_q

    # --- atom metadata -----------------------------------------------------
    atom_base = atoms_ptr + atom * _ATOM_STRIDE
    block_offset = tl.load(atom_base + _ATOM_BLOCK_OFFSET)
    q_start = tl.load(atom_base + _ATOM_Q_START)
    q_len = tl.load(atom_base + _ATOM_Q_LEN)
    kv_blocks = tl.load(atom_base + _ATOM_KV_BLOCKS)
    total_extent = tl.load(atom_base + _ATOM_TOTAL_EXTENT)
    global_q_idx = tl.load(atom_base + _ATOM_GLOBAL_Q_IDX)

    # Grouped-query attention: query heads are grouped in contiguous runs over one
    # KV head, so dividing maps 32 query heads and 8 KV heads to runs of four
    # (heads 0-3 read KV head 0, and so on). At h_h_k_ratio == 1 the division is
    # the identity and this is plain multi-head attention.
    hk = hq // h_h_k_ratio

    # --- queries -----------------------------------------------------------
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < q_len
    d_mask = offs_d < HEAD_SIZE

    q_rows = q_start + offs_m
    q_ptrs = q_ptr + q_rows[:, None] * q_row_stride + hq * HEAD_SIZE + offs_d[None, :]
    # Kept in the input dtype so it matches ``k`` in the dot; the scale is applied
    # to the fp32 scores instead, which is where it belongs numerically.
    q = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)

    # A query token's position in the sequence, not in the tensor: a continuation
    # starts at global_q_idx, which is what the causal mask compares against.
    global_q_pos = global_q_idx + offs_m

    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for b in range(0, kv_blocks):
        block = tl.load(block_idx_ptr + block_offset + b)
        block_row_base = block * k_row_stride * KV_BLOCK_SIZE

        # A KV block holds kv_block_size tokens; walk it in BLOCK_N tiles so the
        # tile does not have to know the cache's block size.
        for t0 in range(0, KV_BLOCK_SIZE, BLOCK_N):
            offs_n = t0 + tl.arange(0, BLOCK_N)
            kv_pos = b * KV_BLOCK_SIZE + offs_n
            n_mask = kv_pos < total_extent

            # ``k`` is loaded already transposed, as (BLOCK_D, BLOCK_N), so the QK
            # dot consumes it directly. Reading it (BLOCK_N, BLOCK_D) and calling
            # ``tl.trans`` gives the same numbers but makes Triton emit a
            # shared-memory layout conversion inside the KV loop; at head_size 64
            # that alone costs about 79% of the kernel's time (1.47 ms -> 0.82 ms
            # on the 4096-token shape), and at head_size 128 with GQA it is worse
            # still. The transposed pointer expression is the memory access the
            # row-major tile would have made anyway, just indexed the other way --
            # the contiguous axis is ``offs_d`` either way.
            k_ptrs = (
                k_ptr
                + block_row_base
                + hk * HEAD_SIZE
                + offs_d[:, None]
                + offs_n[None, :] * k_row_stride
            )
            kT = tl.load(k_ptrs, mask=d_mask[:, None] & n_mask[None, :], other=0.0)

            qk = tl.dot(q, kT) * qk_scale
            if IS_CAUSAL:
                # A query attends to every KV position up to and including its own.
                qk = tl.where(
                    kv_pos[None, :] <= global_q_pos[:, None], qk, float("-inf")
                )
            qk = tl.where(n_mask[None, :], qk, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(qk - m_new[:, None])

            v_ptrs = (
                v_ptr
                + block_row_base
                + offs_n[:, None] * k_row_stride
                + hk * HEAD_SIZE
                + offs_d[None, :]
            )
            v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

            l_i = l_i * alpha + tl.sum(p, 1)
            # ``p`` is rounded into the compute dtype here, which costs about
            # 5e-4 of absolute accuracy. That is deliberate: the bar for this
            # operator is the reference implementation's own tolerance
            # (``inference_test_utils.get_tolerances``: atol 2e-3 for fp16,
            # 3.2e-2 for bf16), which this clears with room to spare. Carrying
            # the rounding residual in a second dot would meet a much tighter
            # atol but costs ~23% of the speedup -- a bad trade against the
            # stated requirement.
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

    # A row whose keys were all masked out has no contribution to renormalize.
    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i[:, None]

    o_ptrs = out_ptr + q_rows[:, None] * o_row_stride + hq * HEAD_SIZE + offs_d[None, :]
    tl.store(
        o_ptrs, acc.to(out_ptr.dtype.element_ty), mask=m_mask[:, None] & d_mask[None, :]
    )


def blocked_flash(
    out,
    q,
    k,
    v,
    attention_atoms,
    kv_block_idx,
    softmax_scale,
    is_causal=True,
    q_block_size=None,
):
    """Flash attention forward over the blocked KV-cache described by ``atoms``.

    Args:
        out (Tensor): output of shape ``[total_q, n_heads_q * head_size]``.
        q (Tensor): queries of shape ``[total_q, n_heads_q * head_size]``.
        k (Tensor): key cache of shape ``[n_blocks, block_size, n_heads_kv, head_size]``,
            contiguous in the last dimension.
        v (Tensor): value cache, same shape and layout as ``k``.
        attention_atoms (Tensor): ``[num_atoms, 8]`` int32; see the module docstring.
        kv_block_idx (Tensor): int32 physical block indices, addressed by each
            atom's ``block_offset`` field.
        softmax_scale (float): scale applied to the query/key dot product.
        is_causal (bool): mask each query against its position in the sequence.
        q_block_size (int, optional): an upper bound on the atoms' ``q_len``,
            normally the ``q_block_size`` the atoms were built with. Supplying it
            avoids a device sync -- see below. ``None`` reads the maximum out of
            ``attention_atoms``, which is always correct and is the default.
            It is a **bound, not a hint**: an atom wider than it would have its
            extra rows masked away, silently.

    Returns:
        Tensor: ``out``, updated in place.
    """
    logger.debug("TRAIN_HYGON BLOCKED_FLASH")

    # The query tile must span the widest atom. Taking that maximum off the
    # device costs a sync (~50 us measured on an A100), which at small shapes is
    # several times the kernel it is sizing -- so a caller who already knows the
    # bound hands it over and the sync never happens.
    head_size = k.size(-1)
    num_heads_q = q.size(-1) // head_size
    num_heads_kv = k.size(-2)
    num_atoms = attention_atoms.size(0)

    _check_launch_arguments(q, k, v, head_size, num_heads_q, num_heads_kv)

    # k is [n_blocks, block_size, n_heads_kv, head_size] with contiguous heads, so
    # stride(0) / stride(1) is the cache's block size -- the same quantity the
    # DeepSpeed launcher leaves implicit.
    kv_block_size = k.stride(0) // k.stride(1)
    if kv_block_size * k.stride(1) != k.stride(0):
        raise ValueError("key cache must be contiguous across the block dimension")

    if q_block_size is None:
        # ``int()`` because the constants are constexpr for the kernel's benefit
        # and torch will not index a tensor with one; ``item()`` is the sync.
        q_len_max = (
            int(attention_atoms[:, int(_ATOM_Q_LEN)].max().item()) if num_atoms else 0
        )
    else:
        q_len_max = q_block_size

    # One program per (atom, query head), so the query tile has to span the widest
    # atom in the batch -- the rest mask off their padding. BLOCK_N is the KV tile
    # walked inside a single cache block. BLOCK_D is the whole head, padded up.
    #
    # BLOCK_N is capped at 32, down from the 128 this used to allow. At a 128-token
    # cache block a 128-wide tile needs 98304 bytes and does not launch on a 64 KB
    # device at all, so the old cap could only ever be reached with the limit check
    # below cutting it back; 32 is the widest tile that has to be considered.
    #
    # 32 rather than 64 is a measurement, and it is really a statement about the
    # pair (BLOCK_N, num_warps) below rather than about the tile by itself. What
    # 32 buys is shared memory: at head 128 it halves the KV loop's live set
    # (49152 -> 32768 bytes) and so buys a second program per compute unit. What
    # it costs is warps per row of the tile, and the two only pay off together.
    #
    # The benchmark's rows say so. Holding num_warps at the 8 the old rule gave
    # head 128, the narrow tile is the *worse* of the two (1.356 against 1.383 on
    # the average, though it wins every 4096-token row). Give a 128x128 tile the
    # 4 warps it actually wants and it reverses hard -- 32 scores 1.458 over three
    # runs against 1.151 for 64, where a 4-warp CTA driving the wide tile falls
    # off a cliff: 3.42 ms against 15.0 on (4096, 128, 64, 8). Narrowing to 16
    # buys nothing anywhere.
    BLOCK_M = max(triton.next_power_of_2(q_len_max), 16)
    BLOCK_N = min(triton.next_power_of_2(kv_block_size), 32)
    BLOCK_D = triton.next_power_of_2(head_size)

    # Shared memory is the one limit that actually bites, and it is a property of
    # the *device*: the same tiling that launches on a 163 KB part dies with
    # OutOfResources on a 64 KB one. So size the tiles against the device rather
    # than against a part.
    #
    # BLOCK_N is the only one of the three that can give. BLOCK_M has to span the
    # widest atom and BLOCK_D the whole head, so shrinking either would silently
    # drop query rows or head dimensions; the KV tile is free, because the inner
    # loop already walks a cache block in BLOCK_N-sized steps, so a narrower tile
    # is the same arithmetic over more passes. Halving keeps it a power of two,
    # which keeps it dividing the block size evenly.
    device_index = q.device.index if q.device.index is not None else 0
    limit = _max_shared_memory(device_index)
    element_size = q.element_size()
    if limit is not None:
        while BLOCK_N > 1 and (
            _shared_memory_for(BLOCK_M, BLOCK_N, BLOCK_D, element_size) > limit
        ):
            BLOCK_N //= 2
        needed = _shared_memory_for(BLOCK_M, BLOCK_N, BLOCK_D, element_size)
        if needed > limit:
            # No KV tile is narrow enough, so the query tile itself is too wide.
            # Only the atom builder sizes that tile, and shrinking it here would
            # drop query rows, so the fix has to go back to the caller.
            raise RuntimeError(
                "blocked flash needs %d bytes of shared memory for a %d-row query "
                "tile at head_size %d, but the device has %d. Build the atoms with "
                "a smaller q_block_size (%d is the widest this head size fits "
                "here)."
                % (
                    needed,
                    BLOCK_M,
                    head_size,
                    limit,
                    _widest_query_tile(BLOCK_D, element_size, limit),
                )
            )

    # Size the CTA to the accumulator. ``acc`` is BLOCK_M x BLOCK_D of fp32 and
    # lives in registers for the whole KV walk, so the per-thread share of it is
    # what decides whether the kernel spills -- and capping registers with
    # ``maxnreg`` to buy occupancy is catastrophic here (128x128 goes from 706 us
    # to 3510 us at maxnreg=128), which says the kernel needs the registers it
    # asks for. Four warps is the floor; above it, the rule targets 64
    # accumulator elements per thread.
    #
    # That target is an A100 measurement -- 128x64 wants 4 warps, 128x128 wants
    # 8 -- and 8 warps of 32 threads is 2048 elements per warp, which is where
    # the old ``// 2048`` came from. The quantity being held fixed is the
    # per-*thread* share, so the constant has to scale with the warp width: this
    # device's warps are 64 threads, and at 64 elements per thread 128x128 wants
    # 4 warps, which is also what it measures best at (4096 tokens, head 128:
    # 1.19 ms at 8 warps against 0.92 at 4). ``BLOCK_M`` of 128 with head 64
    # lands on 4 warps under either arithmetic, so the head-64 tiling keeps the
    # figure the A100 measurement gave it.
    num_warps = max(4, (BLOCK_M * BLOCK_D) // (64 * _warp_size(device_index)))

    _blocked_flash_fwd_kernel[(num_atoms * num_heads_q,)](
        out,
        q,
        k,
        v,
        attention_atoms,
        kv_block_idx,
        # The kernel's softmax runs on exp2, so log2(e) rides on the scale rather
        # than costing an exp() per score.
        softmax_scale * _LOG2E,
        q.stride(0),
        out.stride(0),
        k.stride(1),
        # Query heads that share one KV head; the kernel divides by it to find
        # this program's KV head.
        num_heads_q // num_heads_kv,
        num_heads_q,
        IS_CAUSAL=is_causal,
        # Both are host-side values that the kernel only ever uses as a loop
        # bound and a mask extent. As runtime scalars they force the inner KV
        # loop to stay dynamic and every load mask to carry a term that is
        # statically true; as constexpr they fold away.
        KV_BLOCK_SIZE=kv_block_size,
        HEAD_SIZE=head_size,
        num_warps=num_warps,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    return out
