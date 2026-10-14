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
"""T-Head Zhenwu (PPU) tuned blocked flash attention.

The generic implementation is ``flag_train.deepspeed.blocked_flash`` and is what
every backend without an override runs. What is kept here is the part of that
operator's tuning that only pays on the PPU, measured on a ZW810E.

Three things differ from the generic kernel and launcher:

* a flat 1-D grid with the head index varying fastest and the atoms numbered
  widest-first, worth 5-24% on its own. The programs are dispatched in linear-id
  order, so the order they are numbered in decides which of them run at the same
  time -- see the kernel's comment;
* the KV walk split in two: tiles that lie entirely before the program's first
  query row carry no causal mask and no extent mask, because the comparison the
  masks would make is already decided. On the PPU that is worth another 3-11%,
  and it is why the flat grid's win survives on the large shapes rather than
  being eaten by mask arithmetic;
* ``num_warps`` from the size of the grid rather than from the tile: 8 when the
  grid cannot fill the device, 4 otherwise. Measured 1.38x on the 128-token
  shape (0.01288 ms at 4 warps against 0.00936 at 8) and 1.08x on the 1024-token
  GQA one, while 4096 tokens at head 64 prefers 4 by 1.30x (0.58654 against
  0.76352).

Two things the other overrides do are deliberately *not* carried over, because
they were measured here and did not pay:

* the K tile loaded pre-transposed. Against this kernel in the same process,
  row-major ``tl.trans`` wins every shape (e.g. 0.06400 ms against 0.06680 on a
  1024-token head-64 case, 1.23192 against 1.36426 on a 4096-token head-128
  one), so Triton's PPU backend already folds the transpose -- unlike Hygon's,
  which needs the transposed form;
* splitting the atom's query tile into 64-row programs. It halves the KV window
  each program walks, but on this part the 64x128 tiling costs far more than the
  saved tiles: 2.12 ms against 1.20 on 4096 tokens at head 128.

The kernel body is shared between *two* entry points, and that is not a
stylistic choice. The PPU runtime loads a compiled kernel under its symbol name,
and two builds of one named kernel that agree on everything but ``num_warps``
collide there: the first one loaded serves both. The two builds this launcher
picks between are exactly that pair -- the same tile, the same shared-memory
figure, only the warp count differing -- so with a single name the rule silently
stops working once a shape of the other kind has run. Measured on a ZW810E,
4096 tokens at head 64: 0.58608 ms when its 4-warp build is the one loaded, and
0.76908 ms -- the 8-warp figure -- when a 128-token shape loaded the 8-warp
build first, which is the order the benchmark happens to run in. Giving each
warp count its own symbol keeps them apart; the body is written once and Triton
inlines it, so the arithmetic is still the single implementation the tests cover.

Everything shared is imported from the generic module so the two cannot drift:
the atom-field constants (``_ATOM_*``, which are the kernel's view of the
layout), ``_LOG2E``, and ``_check_launch_arguments`` -- the input checks are one
implementation used by both, so a backend cannot accept something another
rejects. Importing ``_check_launch_arguments`` rather than restating it is also
what keeps the reference-launcher tests working: they monkeypatch the capability
reader in the module it is defined in, and that has to be the generic module
whichever backend dispatched here.

Registration is by name. ``_thead/ops/__init__.py`` re-exports this module's
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

# The widest KV tile this launcher asks for. The generic cap is 128, which a
# 128-token cache block could in principle feed; on this part 64 measured at
# least as fast as 32 and narrower than that never helped, and 64 keeps the
# pipeliner's live set small enough for two programs per compute unit. The tile
# also has to divide the cache block -- see ``_block_n_for``.
_BLOCK_N_CAP = 64

# ``tl.dot`` requires N to be at least 16, so a cache block narrower than that
# cannot be tiled at all.
_MIN_BLOCK_N = 16

# Grid sizes, as multiples of the compute-unit count, below which a program gets
# 8 warps instead of 4: one program per unit is the hard floor (nothing else can
# hide its latency), and eight units' worth of programs is where the 1024-token
# GQA case stops benefiting.
_WIDE_CTA_GRID_SMS = 1
_WIDE_HEAD_GRID_SMS = 8
_WIDE_HEAD_D = 64


@triton.jit
def _blocked_flash_fwd_body(
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
    # is worth 8-24% on this part, measured against the generic 2-D grid in one
    # process: 0.06400 ms against 0.08370 (1024 tokens, head 64), 0.65608
    # against 0.75724 (4096, head 64), 1.23192 against 1.29284 (4096, head 128),
    # 0.18876 against 0.20294 (1024, head 128, GQA 32/8). The programs are
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

    # The KV walk is cut in two at the diagonal. A tile whose last position is
    # still before this program's *first* query row is visible to every row in
    # the tile and lies inside the extent -- both masks would evaluate to "keep"
    # on every element of it -- so the head loop below carries neither, and only
    # the tail loop pays for the compares and the two ``-inf`` selects. With a
    # 128-row query tile against a 64-token cache block only the last two tiles
    # of an atom fall in the tail, so this is the whole loop for every atom but
    # the first. Measured 3-11%: 0.58694 ms against 0.65796 (4096 tokens, head
    # 64), 0.05880 against 0.06396 (1024, head 64), 1.19740 against 1.23172
    # (4096, head 128).
    #
    # Tile t covers positions ``[t * BLOCK_N, (t + 1) * BLOCK_N)``: the tile walk
    # is over BLOCK_N-sized tiles inside a cache block, so the block index is the
    # tile index divided by the tiles a block holds, and the position is the tile
    # index times BLOCK_N. ``n_full`` is how many of them fit before
    # ``global_q_idx``, the position of this atom's first query token.
    tiles_per_block = KV_BLOCK_SIZE // BLOCK_N
    n_tiles = kv_blocks * tiles_per_block
    n_full = global_q_idx // BLOCK_N

    for t in range(0, n_full):
        b = t // tiles_per_block
        t0 = (t % tiles_per_block) * BLOCK_N
        block = tl.load(block_idx_ptr + block_offset + b)
        block_row_base = block * k_row_stride * KV_BLOCK_SIZE
        offs_n = t0 + tl.arange(0, BLOCK_N)

        k_ptrs = (
            k_ptr
            + block_row_base
            + offs_n[:, None] * k_row_stride
            + hk * HEAD_SIZE
            + offs_d[None, :]
        )
        kk = tl.load(k_ptrs, mask=d_mask[None, :], other=0.0)

        qk = tl.dot(q, tl.trans(kk)) * qk_scale

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
        vv = tl.load(v_ptrs, mask=d_mask[None, :], other=0.0)

        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(vv.dtype), vv)
        m_i = m_new

    for t in range(n_full, n_tiles):
        b = t // tiles_per_block
        t0 = (t % tiles_per_block) * BLOCK_N
        block = tl.load(block_idx_ptr + block_offset + b)
        block_row_base = block * k_row_stride * KV_BLOCK_SIZE
        offs_n = t0 + tl.arange(0, BLOCK_N)
        kv_pos = t * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = kv_pos < total_extent

        k_ptrs = (
            k_ptr
            + block_row_base
            + offs_n[:, None] * k_row_stride
            + hk * HEAD_SIZE
            + offs_d[None, :]
        )
        kk = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

        qk = tl.dot(q, tl.trans(kk)) * qk_scale
        if IS_CAUSAL:
            # A query attends to every KV position up to and including its own.
            qk = tl.where(kv_pos[None, :] <= global_q_pos[:, None], qk, float("-inf"))
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
        vv = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

        l_i = l_i * alpha + tl.sum(p, 1)
        # ``p`` is rounded into the compute dtype here, which costs about
        # 5e-4 of absolute accuracy. That is deliberate: the bar for this
        # operator is the reference implementation's own tolerance
        # (``inference_test_utils.get_tolerances``: atol 2e-3 for fp16,
        # 3.2e-2 for bf16), which this clears with room to spare. Carrying
        # the rounding residual in a second dot would meet a much tighter
        # atol but costs ~23% of the speedup -- a bad trade against the
        # stated requirement.
        acc = acc * alpha[:, None] + tl.dot(p.to(vv.dtype), vv)
        m_i = m_new

    # A row whose keys were all masked out has no contribution to renormalize.
    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i[:, None]

    o_ptrs = out_ptr + q_rows[:, None] * o_row_stride + hq * HEAD_SIZE + offs_d[None, :]
    tl.store(
        o_ptrs, acc.to(out_ptr.dtype.element_ty), mask=m_mask[:, None] & d_mask[None, :]
    )


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
    """The 4-warp program. Its own symbol -- see the module docstring."""
    _blocked_flash_fwd_body(
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
        IS_CAUSAL=IS_CAUSAL,
        KV_BLOCK_SIZE=KV_BLOCK_SIZE,
        HEAD_SIZE=HEAD_SIZE,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )


@libentry()
@triton.jit
def _blocked_flash_fwd_kernel_wide(
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
    """The 8-warp program: the same body under a symbol of its own."""
    _blocked_flash_fwd_body(
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
        IS_CAUSAL=IS_CAUSAL,
        KV_BLOCK_SIZE=KV_BLOCK_SIZE,
        HEAD_SIZE=HEAD_SIZE,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )


@functools.lru_cache(maxsize=None)
def _device_properties(device_index):
    """The device's property dict, or ``None`` if it cannot be read.

    Cached: the read is on the per-call path and device properties do not change
    under a running process, so there is nothing to invalidate. ``None`` means
    "cannot tell", and the warp rule below then keeps the 4 warps that the
    generic launcher's tile arithmetic lands on anyway.
    """
    try:
        from triton.runtime.driver import driver as triton_driver

        return triton_driver.active.utils.get_device_properties(device_index)
    except Exception:
        return None


def _compute_units(device_index):
    """The device's compute-unit count, or ``None`` if unreadable."""
    props = _device_properties(device_index)
    count = props.get("multiprocessor_count") if props else None
    return int(count) if count else None


def _block_n_for(kv_block_size):
    """The KV tile for this cache block size: a power of two that divides it.

    Two constraints, and only one of them is about speed:

    * the tile is capped at 64 rows -- see ``_BLOCK_N_CAP``;
    * it must *divide* the cache block. The walk indexes tiles by their position
      within the block (``t % tiles_per_block``), so a tile that did not divide
      the block would leave the block's last rows unvisited -- a wrong answer
      rather than a slow one. Power-of-two block sizes satisfy this by
      construction, which is what real paged caches use; anything else is
      refused here rather than silently mis-attended.
    """
    block_n = min(_BLOCK_N_CAP, 1 << (int(kv_block_size).bit_length() - 1))
    if block_n < _MIN_BLOCK_N:
        raise ValueError(
            "blocked flash walks a cache block in KV tiles of at least %d rows, "
            "so the key cache's block size must be at least that; this one is %d"
            % (_MIN_BLOCK_N, kv_block_size)
        )
    if kv_block_size % block_n != 0:
        raise ValueError(
            "blocked flash tiles a cache block at %d rows, which must divide the "
            "block; this block size is %d" % (block_n, kv_block_size)
        )
    return block_n


def _use_wide_cta(num_programs, block_d, device_index):
    """Whether a program gets 8 warps instead of 4.

    Not a function of the tile, which is what the generic launcher sizes warps
    from. On this part the tile is not what decides -- the grid is. A 128x64
    tile wants 4 warps at 4096 tokens and 8 at 128, where the whole problem is
    16 programs on 64 compute units: with only one program per unit there is no
    other program to hide latency behind, so the work has to be spread wider
    instead (0.00936 ms against 0.01288). The same holds for a head-128 tile
    over a short KV walk (1024 tokens, GQA 32/8: 0.19016 against 0.20468, 7%),
    while long walks prefer 4 warps for the occupancy they buy (4096 tokens,
    head 128: 1.16948 against 1.19820).

    ``block_d`` is what separates the two head-128 cases from the head-64 ones
    the same grid size would otherwise put on 8 warps: at a 128-wide head the
    accumulator is twice the size, and a narrow tile leaves too few warps to
    hold it. Both thresholds are multiples of the compute-unit count, so the
    rule is about how full the device is, not about absolute shape sizes.
    """
    compute_units = _compute_units(device_index)
    if compute_units is None:
        return False
    if num_programs < _WIDE_CTA_GRID_SMS * compute_units:
        return True
    if block_d > _WIDE_HEAD_D and num_programs < _WIDE_HEAD_GRID_SMS * compute_units:
        return True
    return False


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
            avoids a device sync -- see below. ``None`` reads the maximum out of
            ``attention_atoms``, which is always correct and is the default.
            It is a **bound, not a hint**: an atom wider than it would have its
            extra rows masked away, silently.

    Returns:
        Tensor: ``out``, updated in place.
    """
    logger.debug("TRAIN_THEAD BLOCKED_FLASH")

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
    # head, padded up.
    BLOCK_M = max(triton.next_power_of_2(q_len_max), 16)
    BLOCK_D = triton.next_power_of_2(head_size)
    BLOCK_N = _block_n_for(kv_block_size)

    device_index = q.device.index if q.device.index is not None else 0
    wide = _use_wide_cta(num_atoms * num_heads_q, BLOCK_D, device_index)
    # The two warp counts live under separate symbols; see the module docstring
    # for why sharing one does not work on this backend.
    kernel = _blocked_flash_fwd_kernel_wide if wide else _blocked_flash_fwd_kernel

    kernel[(num_atoms * num_heads_q,)](
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
        num_warps=8 if wide else 4,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    return out
