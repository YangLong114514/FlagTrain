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
"""MetaX-tuned blocked flash attention.

The generic implementation is ``flag_train.deepspeed.blocked_flash`` and is what
every backend without an override runs. What is kept here is the part of that
operator's tuning that only pays on MetaX's parts.

Four things differ from the generic kernel and launcher:

* a flat 1-D grid with the head index varying fastest and the atoms numbered
  widest-first, worth 1.3-2x on its own (see the kernel's comment);
* a KV tile chosen by *head width* rather than by cache block size alone. On this
  part ``BLOCK_N`` at or above 32 with a head wider than 64 lands the QK and PV
  dots on two different MMA layouts, and Triton then emits a layout conversion
  inside the KV loop; 16 is the widest tile that avoids it, and measures 1.8-1.9x
  faster. 16 is also ``tl.dot``'s narrowest N, so it is the floor;
* ``num_warps`` sized by *threads* rather than warps -- MetaX's warps are 64
  threads against NVIDIA's 32, and the part caps a CTA at 512 threads, so the
  count is also clamped to 8;
* the KV tile settled against the device's shared-memory budget *before* the
  launch, from the figure Triton reports for the tiling (``warmup`` compiles
  without running), so the launch that follows is unconditional. No closed form
  is used -- see ``_allocated_shared_memory`` for why one cannot be -- and when
  even the narrowest tile does not fit the error names the exact requirement.

The kernel *body* is the generic kernel's, unchanged: only the program-to-work
mapping and the launch scalars differ. That is deliberate -- the arithmetic is
then identical to the implementation the numeric tests already cover, and this
module is only responsible for the choices that are MetaX's.

Two things Hygon's override does are deliberately *not* carried over, because
they were measured here and did not pay:

* the K tile loaded pre-transposed. Against the flat grid in the same process,
  row-major ``tl.trans`` wins four of five benchmark shapes (e.g. 2008 us against
  2394 us on a 4096-token head-64 case), and never loses by much. Hygon needs it
  because its Triton does not fold the transpose; this one does;
* capping the KV tile at 32 unconditionally. It is one tile too wide here once
  the head passes 64, which is where the layout conversion above appears.

Everything shared is imported from the generic module so the two cannot drift:
the atom-field constants (``_ATOM_*``, which are the kernel's view of the
layout), ``_LOG2E``, and ``_check_launch_arguments`` -- the input checks are one
implementation used by both, so a backend cannot accept something another
rejects. Importing ``_check_launch_arguments`` rather than restating it is also
what keeps the reference-launcher tests working: they monkeypatch the capability
reader in the module it is defined in, and that has to be the generic module
whichever backend dispatched here.

Registration is by name. ``_metax/ops/__init__.py`` re-exports this module's
``blocked_flash`` straight from here, and ``SpecOpRegistrar`` writes it over the
same-named generic function in ``flag_train.deepspeed`` when that package is
imported -- the namespace the tests and the benchmark read the operator from.
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

# ``tl.dot`` requires N to be at least 16, and so does this launcher's narrowing
# loop below: 16 is the narrowest KV tile that compiles, not merely the narrowest
# that fits. Halving past it produces a compilation error rather than a resource
# error, so the loop has to stop here rather than at 1.
_MIN_BLOCK_N = 16

# The widest KV tile the tile-selection policy will ask for, before the device's
# shared-memory budget narrows it further. See the module docstring for why the
# threshold sits at a 64-wide head.
_BLOCK_N_CAP_WIDE_HEAD = 16
_BLOCK_N_CAP_NARROW_HEAD = 32
_NARROW_HEAD_D = 64

# 512 threads per CTA, in warps, at this part's 64-thread warps. Used to clamp
# ``num_warps``; see the launcher.
_MAX_THREADS_PER_CTA = 512


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
    # is worth more than the kernel's other knobs on this part: the programs are
    # dispatched in linear-id order, so the order they are numbered in decides
    # which of them run at the same time. Two effects, both order-only --
    # permuting independent programs cannot change a result:
    #
    #   * heads vary fastest. The programs resident at any moment are then the
    #     same atom on different heads, which read the same cache blocks; inside
    #     a block the heads are contiguous, so memory sees one sequential stream
    #     instead of ``n_heads_q`` scattered ones.
    #   * atoms run from the widest to the narrowest. An atom's work grows with
    #     its index -- under the causal single-sequence layout, with a 128-row
    #     query tile against a 64-token cache block, atom *i* walks 2(i+1) cache
    #     blocks -- so numbering them backwards hands the long programs to the
    #     device first and leaves it the short ones to drain on. That ordering is
    #     the one the benchmark's atoms are built in; for atoms from anywhere else
    #     the reversal is an arbitrary permutation, which is why it is safe to
    #     apply unconditionally.
    #
    # Measured against the generic 2-D ``(atoms, heads)`` grid in one process on
    # a C550, this ordering wins every shape it was measured on: 2008 against
    # 4094 us (4096 tokens, head 64), 10120 against 12683 us (4096, head 128),
    # 1464 against 1996 us (1024, head 128, 32/8 heads). The 2-D grid cannot express
    # either ordering -- its fast axis is the one with the shortest span -- and a
    # ``(heads, atoms)`` grid would need ``num_atoms`` to fit in grid dimension
    # 1, capping a batch at 65535 atoms. Flattening keeps the fast-varying axis
    # under our control while leaving the grid bound at 2**31 programs, as the
    # original ``(atoms, heads)`` grid had.
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

            k_ptrs = (
                k_ptr
                + block_row_base
                + offs_n[:, None] * k_row_stride
                + hk * HEAD_SIZE
                + offs_d[None, :]
            )
            k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

            qk = tl.dot(q, tl.trans(k)) * qk_scale
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


@functools.lru_cache(maxsize=None)
def _device_properties(device_index):
    """The device's property dict, or ``None`` if it cannot be read.

    Cached. The call is cheap here -- about 2 us measured on a C550, against
    kernels that run in tens of microseconds, and it is asked for once per
    launch -- so this is not load-bearing the way it is on parts where the
    property read is milliseconds. It is kept because the read is still on the
    per-call path and device properties do not change under a running process,
    so there is nothing to re-read and nothing to invalidate.
    """
    try:
        from triton.runtime.driver import driver as triton_driver

        return triton_driver.active.utils.get_device_properties(device_index)
    except Exception:
        return None


def _warp_size(device_index):
    """Threads per warp, or 32 when the device will not say.

    A warp is not a fixed width across devices -- 32 on the NVIDIA parts this
    operator was first tuned on, 64 here -- and the per-CTA sizing below is a
    count of *threads*, so it has to know which. 32 is the value that sizing
    already assumed before it could ask, so an unreadable device keeps that
    behaviour rather than getting a new one.
    """
    props = _device_properties(device_index)
    size = props.get("warpSize") if props else None
    return int(size) if size else 32


# The KV tile chosen for each (device, shape) request, so the selection runs once
# per shape rather than once per call. This caches a *decision*, not a trial:
# nothing here records a failure, because nothing fails -- the choice is made from
# the allocator's own number before the launch is issued. Keyed by everything the
# allocation can depend on, including ``KV_BLOCK_SIZE`` -- it is a constexpr loop
# bound, and the pipeliner's buffer count is a function of it.
_SELECTED_BLOCK_N = {}


def _max_shared_memory(device_index):
    """The device's shared-memory limit in bytes, or ``None`` if unreadable.

    ``None`` means "cannot tell", and the caller then leaves the tiling alone
    rather than guessing a limit it did not state -- the same rule the input
    checks follow for a device that will not report a compute capability.
    """
    props = _device_properties(device_index)
    limit = props.get("max_shared_mem") if props else None
    return int(limit) if limit else None


def _allocated_shared_memory(
    out,
    q,
    k,
    v,
    attention_atoms,
    kv_block_idx,
    softmax_scale,
    is_causal,
    kv_block_size,
    head_size,
    h_h_k_ratio,
    n_heads_q,
    num_warps,
    block_m,
    block_n,
    block_d,
):
    """Shared memory Triton allocates for one tiling, in bytes, without launching.

    ``warmup`` compiles and hands back the compiled kernel without running it,
    which is the whole point: a tiling past the device's limit still *compiles* --
    only the launch refuses it, inside ``CompiledKernel._init_handles``. So the
    allocator's own figure is readable for a tiling that could never be launched,
    and the selection below needs neither a failed launch nor a closed form.

    A closed form is what this operator's other backends use, and it is not
    available here: the allocation is the allocator's answer about buffer reuse,
    which is a function of the layouts it picked for the two dots, and that is a
    function of the compiler. Fitting one to this part gave an expression that
    reproduces every narrow-head tiling exactly and *under*-states the wide-head
    ones by 4-15 KB -- wrong in the direction that lets a doomed launch through,
    which is the failure being replaced: an expression that is right until it is
    not.

    The figure is the one the launch itself is checked against, which is what
    makes it usable as a pre-check: a 128-row tile at head 128 with ``BLOCK_N`` of
    64 reports 81920 here and is refused with ``Required: 81920`` there.

    The arguments are the launch's, because the kernel is compiled per signature
    and the strides are part of it; they are not read.
    """
    compiled = _blocked_flash_fwd_kernel.warmup(
        out,
        q,
        k,
        v,
        attention_atoms,
        kv_block_idx,
        softmax_scale,
        q.stride(0),
        out.stride(0),
        k.stride(1),
        h_h_k_ratio,
        n_heads_q,
        grid=(1,),
        IS_CAUSAL=is_causal,
        KV_BLOCK_SIZE=kv_block_size,
        HEAD_SIZE=head_size,
        num_warps=num_warps,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
    )
    # ``warmup`` returns the same ``(kernel, hooks)`` pair a launch does.
    compiled = compiled[0] if isinstance(compiled, tuple) else compiled
    return compiled.metadata.shared


def _block_n_for(block_d, kv_block_size):
    """The KV tile this head size wants, before the device narrows it.

    ``BLOCK_D`` above 64 is the case that matters: at ``BLOCK_N >= 32`` the QK dot
    and the PV dot are assigned different MMA layouts, and Triton inserts a
    ``convert_layout`` between them inside the KV loop. On a C550 that costs
    1.8-1.9x -- 877 against 1677 us at 1024 tokens, 10117 against 18469 us at
    4096 -- and 16 is the widest tile that keeps both dots on one layout. At
    ``BLOCK_D <= 64`` the two agree and the wider tile is the faster one: 32
    against 64 costs 2007 against 4177 us at 4096 tokens.

    The tile is also held to the cache block size. It must never exceed it: the
    inner loop masks a lane by ``kv_pos < total_extent``, which says nothing about
    the block boundary, so a tile wider than the block would load rows belonging
    to the *next* physical block and attend to KV the query was never given. The
    generic launcher's ``next_power_of_2`` does not hold that line for a block
    size that is not a power of two -- 48 rounds up to 64 -- so this takes the
    largest power of two *below* the block size instead.
    """
    cap = (
        _BLOCK_N_CAP_WIDE_HEAD if block_d > _NARROW_HEAD_D else _BLOCK_N_CAP_NARROW_HEAD
    )
    block_n = min(cap, 1 << (int(kv_block_size).bit_length() - 1))
    if block_n < _MIN_BLOCK_N:
        raise ValueError(
            "blocked flash walks a cache block in KV tiles of at least %d rows, "
            "so the key cache's block size must be at least that; this one is %d"
            % (_MIN_BLOCK_N, kv_block_size)
        )
    return block_n


def _select_block_n(block_n, allocated, limit, block_m, head_size):
    """Narrow the KV tile until the device can hold the tiling, before launching.

    ``allocated`` maps a KV tile to the bytes that tiling needs, and is asked
    rather than assumed -- see ``_allocated_shared_memory``. The KV tile is the
    only one of the three free to give: ``BLOCK_M`` must span the widest atom and
    ``BLOCK_D`` the whole head, so shrinking either would silently drop query rows
    or head dimensions. Halving keeps it a power of two, which keeps it dividing
    the cache block evenly, and the inner loop already walks a block in
    ``BLOCK_N``-sized steps, so a narrower tile is the same arithmetic over more
    passes.

    Raises rather than returning a tile that cannot work, with the figures it
    measured: the caller cannot fix an oversized query tile from here (only the
    atom builder sizes it), so the diagnosis has to travel back to them.
    """
    while True:
        needed = allocated(block_n)
        if needed <= limit or block_n <= _MIN_BLOCK_N:
            break
        block_n //= 2
    if needed > limit:
        smaller = (
            " %d rows is the next tile down." % (block_m // 2)
            if block_m > _MIN_BLOCK_N
            else " %d rows is already the narrowest query tile there is." % _MIN_BLOCK_N
        )
        raise RuntimeError(
            "blocked flash needs %d bytes of shared memory for a %d-row query "
            "tile at head_size %d, and the device has %d. That is already at the "
            "narrowest KV tile that compiles (%d rows).%s Build the atoms with a "
            "smaller q_block_size."
            % (needed, block_m, head_size, limit, _MIN_BLOCK_N, smaller)
        )
    return block_n


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
        attention_atoms (Tensor): ``[num_atoms, 8]`` int32; see the generic
            module's docstring for the field layout.
        kv_block_idx (Tensor): int32 physical block indices, addressed by each
            atom's ``block_offset`` field.
        softmax_scale (float): scale applied to the query/key dot product.
        is_causal (bool): mask each query against its position in the sequence.
        q_block_size (int, optional): an upper bound on the atoms' ``q_len``,
            normally the ``q_block_size`` the atoms were built with. Supplying it
            avoids a device sync. ``None`` reads the maximum out of
            ``attention_atoms``, which is always correct and is the default.
            It is a **bound, not a hint**: an atom wider than it would have its
            extra rows masked away, silently.

    Returns:
        Tensor: ``out``, updated in place.
    """
    logger.debug("TRAIN_METAX BLOCKED_FLASH")

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
    # atom in the batch -- the rest mask off their padding. BLOCK_D is the whole
    # head, padded up. BLOCK_M and BLOCK_D together decide whether *any* tiling
    # fits the device, because neither can give: shrinking BLOCK_M would drop
    # query rows and shrinking BLOCK_D would drop head dimensions, both silently.
    BLOCK_M = max(triton.next_power_of_2(q_len_max), 16)
    BLOCK_D = triton.next_power_of_2(head_size)
    BLOCK_N = _block_n_for(BLOCK_D, kv_block_size)

    device_index = q.device.index if q.device.index is not None else 0
    element_size = q.element_size()

    # Size the CTA to the accumulator. ``acc`` is BLOCK_M x BLOCK_D of fp32 and
    # lives in registers for the whole KV walk, so the per-thread share of it is
    # what decides whether the kernel spills -- and capping registers with
    # ``maxnreg`` to buy occupancy is catastrophic here, which says the kernel
    # needs the registers it asks for. The generic rule's ``// 2048`` is an
    # A100 measurement -- 8 warps of 32 threads, holding the per-thread share
    # near 64 -- so the constant has to scale with the warp width: this device's
    # warps are 64 threads. Four warps is the floor. Unlike the generic rule this
    # one is also clamped to the CTA limit: 512 threads here, i.e. 8 warps, and a
    # wider ``BLOCK_M * BLOCK_D`` would ask for more and then fail to launch. At
    # the tilings this operator actually uses -- 128x64, 128x128, 32x256 -- 4
    # warps and 8 measure the same, so the rule is here for the shapes that are
    # not on the benchmark, not for the ones that are.
    warp_size = _warp_size(device_index)
    num_warps = max(4, (BLOCK_M * BLOCK_D) // (64 * warp_size))
    num_warps = min(num_warps, max(1, _MAX_THREADS_PER_CTA // warp_size))

    # Shared memory is the limit that actually bites, and it is the allocator's
    # answer rather than anything derivable from the source. So the KV tile is
    # settled *before* the launch, from the figure Triton reports for the tiling
    # (``_allocated_shared_memory``), which is what makes the launch below
    # unconditional: by the time it is issued the tiling is known to fit, so there
    # is no failure to catch and no second attempt to make. The device is read
    # once per shape -- the answer is memoised -- rather than per call, and when it
    # will not report a limit the tiling is left alone instead of being narrowed
    # against a budget the device never stated.
    key = (device_index, BLOCK_M, BLOCK_D, kv_block_size, element_size, BLOCK_N)
    if key not in _SELECTED_BLOCK_N:
        limit = _max_shared_memory(device_index)

        def allocated(block_n):
            return _allocated_shared_memory(
                out,
                q,
                k,
                v,
                attention_atoms,
                kv_block_idx,
                softmax_scale,
                is_causal,
                kv_block_size,
                head_size,
                num_heads_q // num_heads_kv,
                num_heads_q,
                num_warps,
                BLOCK_M,
                block_n,
                BLOCK_D,
            )

        _SELECTED_BLOCK_N[key] = (
            BLOCK_N
            if limit is None
            else _select_block_n(BLOCK_N, allocated, limit, BLOCK_M, head_size)
        )
    block_n = _SELECTED_BLOCK_N[key]

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
        # The flat grid's fast axis is the head index, so the kernel needs the
        # head count to take the modulus.
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
        BLOCK_N=block_n,
        BLOCK_D=BLOCK_D,
    )
    return out
