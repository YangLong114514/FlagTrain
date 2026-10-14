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
"""Ascend-tuned blocked flash attention.

The generic implementation is ``flag_train.deepspeed.blocked_flash`` and is what
every backend without an override runs. What is kept here is the part of that
operator that does not survive this backend's compiler.

The generic kernel does not launch here at all. At the tilings the tests and the
launcher actually ask for -- a 128-row query tile at head 64 against a 64-token
cache block -- BiShengHIR refuses it:

    ub overflow, requires 1593344 bits while 1572864 bits available

1572864 bits is the 192 KB unified buffer of one AIV core, and the request
overshoots it by 2.5 KB. Measured over the tilings this operator can be launched
with, 30 of the 50 cases in the test suite fail this way. The failure is at
*compile* time, not launch time: past the limit the kernel does not exist, so
there is nothing to catch and retry at the call site.

Three things differ from the generic kernel and launcher.

* The two score masks are one. The generic body writes the causal test and the
  extent test as two ``tl.where`` calls over the same fp32 BLOCK_M x BLOCK_N
  score tile; the compiler keeps a live temporary for each, on top of the scores
  themselves, and that tile is the largest thing inside the KV loop. The two
  tests are independent, so ``where(a & b, x, -inf)`` is the same value as the
  nested form -- and it is one masked write over the tile instead of two. This
  is the fix rather than a tuning knob: it is what brings the common tiling from
  194.5 KB (does not compile) to one that does.

* The query tile is split across programs. ``BLOCK_M`` is capped, and an atom
  wider than the cap is walked by ``ceil(q_len / BLOCK_M)`` programs instead of
  one. The accumulator is BLOCK_M x BLOCK_D of fp32 and lives for the whole KV
  walk, and unlike the KV tile it cannot be narrowed: shrinking it would drop
  query rows silently. Splitting it across programs is the only way to bound it.

* The KV tile is *not* narrowed. Neither existing override's approach transfers.
  Hygon and MetaX size their KV tile against the device's shared-memory budget,
  read from ``get_device_properties()["max_shared_mem"]``; this backend reports
  **1** there -- a placeholder, not a budget -- so a loop that halves until the
  tiling fits would walk to its floor and then raise on every call. MetaX's other
  route, compiling the tiling with ``warmup`` and reading the allocator's own
  figure, is not available either: the entry point is a ``LibEntry``, which has
  no ``warmup``.

  A closed form is what is left, and a monotone one is only writable because of
  the one-mask change above. Before it, the measured table was *not* monotone --
  at head 256 with a 64-row tile, ``BLOCK_N`` 16 and 32 overflowed while 64 fit,
  so a policy of narrowing the KV tile would have walked into a cell that does
  not compile on its way to one that does. After it, every remaining failure is
  explained by ``BLOCK_M`` and ``BLOCK_N`` alone, and ``BLOCK_M <= 64`` compiles
  for every head width and every KV tile this launcher can pick.

What is *not* carried over from the older overrides, and why:

* Hygon's pre-transposed K load. On this backend it is UB-neutral: swept over the
  same tilings it reproduces the generic body's figures exactly, cell for cell, at
  every head width. It costs readability and buys nothing here, so the body keeps
  ``tl.trans``.
* Hygon's and MetaX's flat 1-D grid with the atoms numbered widest-first. That
  ordering is worth 20-30% on those parts because the programs resident together
  read the same cache blocks. It is order-only -- permuting independent programs
  cannot change a result -- so carrying it over could not be *wrong*, but it is
  unmeasured here, and this kernel needs a flat grid for a different reason: the
  query-tile index is a third axis, and the generic 2-D ``(atoms, heads)`` grid
  has no room for it. The ordering this file uses puts the query tile fastest,
  then the head, then the atom, which keeps programs resident together on one
  atom's cache blocks; the atoms themselves run in build order.

Everything shared is imported from the generic module so the two cannot drift:
the atom-field constants (``_ATOM_*``, which are the kernel's view of the
layout), ``_LOG2E``, and ``_check_launch_arguments`` -- the input checks are one
implementation used by both, so a backend cannot accept something another
rejects. Importing ``_check_launch_arguments`` rather than restating it is also
what keeps the reference-launcher tests working: they monkeypatch the capability
reader in the module it is defined in, and that has to be the generic module
whichever backend dispatched here.

Registration is by name. ``_ascend/ops/deepspeed/__init__.py`` re-exports this
module's ``blocked_flash``, and ``SpecOpRegistrar`` writes it over the same-named
generic function in ``flag_train.deepspeed`` when that package is imported -- the
namespace the tests and the benchmark read the operator from.
"""

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

# The widest query tile that compiles, as a function of the two tile widths that
# decide the unified-buffer footprint. See ``_query_tile_cap``.
_NARROW_QUERY_TILE = 64
_WIDE_QUERY_TILE = 128

# The head width past which the narrow query tile is the only one that fits,
# whatever the KV tile is.
_WIDE_HEAD_D = 256

# The KV tile width past which the wide query tile stops fitting. It is a
# ``tl.dot`` operand, so 16 is its floor and 128 is the generic launcher's
# ceiling.
_WIDE_KV_TILE = 64


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
    n_q_tiles,
    IS_CAUSAL: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # A flat 1-D grid, because there are three axes of work and the generic 2-D
    # grid only has room for two: the query tile is the third. Flattening also
    # keeps the fast-varying axis under our control -- the programs dispatched
    # together are the ones nearest in linear-id order, and numbering the query
    # tile fastest puts the programs that read the *same* cache blocks next to
    # each other. Two programs of one atom differ only in which query rows they
    # own; their KV walk is identical, so this is the strongest locality the
    # ordering can offer. The head follows, so the run of programs sharing an
    # atom's cache is ``n_q_tiles * n_heads_q`` long, and the atom varies
    # slowest.
    #
    # Ordering is order-only: the programs are independent and none reads
    # another's result, so any permutation gives the same answer -- which is why
    # this one is safe to choose on locality grounds alone.
    pid = tl.program_id(0)
    q_tile = pid % n_q_tiles
    hq = (pid // n_q_tiles) % n_heads_q
    atom = pid // (n_q_tiles * n_heads_q)

    # --- atom metadata -----------------------------------------------------
    atom_base = atoms_ptr + atom * _ATOM_STRIDE
    block_offset = tl.load(atom_base + _ATOM_BLOCK_OFFSET)
    q_start = tl.load(atom_base + _ATOM_Q_START)
    q_len = tl.load(atom_base + _ATOM_Q_LEN)
    kv_blocks = tl.load(atom_base + _ATOM_KV_BLOCKS)
    total_extent = tl.load(atom_base + _ATOM_TOTAL_EXTENT)
    global_q_idx = tl.load(atom_base + _ATOM_GLOBAL_Q_IDX)

    # The query tile is a slice of the atom, so it starts ``q_tile * BLOCK_M``
    # rows in. An atom narrower than ``n_q_tiles * BLOCK_M`` -- which is every
    # atom but the widest, since the count is sized to the bound -- leaves whole
    # tiles with no rows at all. They return rather than run: the load masks
    # would make them compute on zeros and store nothing, but a tile past the
    # atom's end also has no causal window (``global_q_pos`` is past the
    # sequence, so every key is masked), which drives the online softmax through
    # ``exp2(-inf - -inf)`` to a NaN it then discards. Exiting is cheaper and
    # leaves nothing to reason about.
    q_tile_start = q_tile * BLOCK_M
    if q_tile_start >= q_len:
        return

    row = q_tile_start + tl.arange(0, BLOCK_M)
    m_mask = row < q_len

    # Grouped-query attention: query heads are grouped in contiguous runs over one
    # KV head, so dividing maps 32 query heads and 8 KV heads to runs of four
    # (heads 0-3 read KV head 0, and so on). At h_h_k_ratio == 1 the division is
    # the identity and this is plain multi-head attention.
    hk = hq // h_h_k_ratio

    # --- queries -----------------------------------------------------------
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < HEAD_SIZE

    q_rows = q_start + row
    q_ptrs = q_ptr + q_rows[:, None] * q_row_stride + hq * HEAD_SIZE + offs_d[None, :]
    # Kept in the input dtype so it matches ``k`` in the dot; the scale is applied
    # to the fp32 scores instead, which is where it belongs numerically.
    q = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)

    # A query token's position in the sequence, not in the tensor: a continuation
    # starts at global_q_idx, which is what the causal mask compares against. The
    # query tile offsets it by the rows this program owns, which is what keeps the
    # split from moving the mask.
    global_q_pos = global_q_idx + row

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

            # One masked write over the score tile, not two. The generic body
            # applies the causal test and then the extent test as separate
            # ``tl.where`` calls; each is its own elementwise op over this fp32
            # tile and the compiler keeps a temporary for it on top of the
            # scores themselves. Folding them into a single predicate is the
            # same value -- the tests are independent, so ``where(a & b, x,
            # -inf)`` and the nested form agree on every element -- and it is
            # what puts the tilings this operator is launched with inside the
            # 192 KB unified buffer. On the common one (a 128-row query tile at
            # head 64 against a 64-token block) it is the difference between
            # 194.5 KB and compiling at all.
            #
            # The order of the ``&`` matters for cost, not for value: ``n_mask``
            # is one row of BLOCK_N and the causal test is the full tile, so
            # testing the row first lets the broadcast happen after.
            if IS_CAUSAL:
                # A query attends to every KV position up to and including its
                # own.
                keep = (kv_pos[None, :] <= global_q_pos[:, None]) & n_mask[None, :]
            else:
                keep = tl.broadcast_to(n_mask[None, :], (BLOCK_M, BLOCK_N))
            qk = tl.where(keep, qk, float("-inf"))

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


def _query_tile_cap(block_d, block_n):
    """The widest query tile that compiles, for a given head and KV tile width.

    Read off a sweep of every tiling this launcher can ask for (``BLOCK_D`` in
    {64, 128, 256} x ``BLOCK_N`` in {16, 32, 64, 128, 256}, fp16, one compile
    per cell, the compiler's own ``ub overflow`` figure as the boundary). The
    whole of what it says:

    * ``BLOCK_M = 64`` compiles in **every** cell -- every head width, every KV
      tile. That is the floor this function never goes below.
    * ``BLOCK_M = 128`` compiles while the KV tile is 64 or narrower *and* the
      head is 128 or narrower. Each half is load-bearing: at head 64 a 128-row
      tile against a 128-wide KV tile overflows (306.5 KB), and at a 64-wide KV
      tile a head-256 tile overflows at every KV width (197.3 KB at its best).
    * ``BLOCK_M = 256`` compiles only at head 64 with a KV tile of 128 or less,
      which is a cell ``BLOCK_M = 128`` already covers at a quarter of the
      programs. Not worth a third branch.

    The cap has to be a function of the *KV* tile as well as the head, which is
    why it is not simply read off ``BLOCK_D``: at head 64 a 128-row query tile is
    fine against a 64-wide KV tile and does not compile against a 128-wide one.

    Two notes on why this is a cap and not a search. It is monotone, which is
    what lets a cap be written at all: the boundary was swept before the
    one-mask change too, and it was *not* monotone -- at head 256 with a 64-row
    query tile the 16- and 32-wide KV tiles overflowed while the 64-wide one
    compiled -- so a policy that narrowed a fitting tiling until it fit would
    have passed through a cell that does not compile. With the masks folded the
    monotonicity holds in both arguments, and every failure in the sweep is
    explained by this pair. And it is a cap rather than a compile-and-retry
    because a tiling past the limit is not a runtime failure that can be caught:
    the kernel never compiles, so there is nothing to fall back from -- the
    choice has to be made before the launch, from numbers, which is what these
    are.
    """
    if block_d < _WIDE_HEAD_D and block_n <= _WIDE_KV_TILE:
        return _WIDE_QUERY_TILE
    return _NARROW_QUERY_TILE


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
    logger.debug("TRAIN_ASCEND BLOCKED_FLASH")

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

    # BLOCK_D is the whole head, padded up; it cannot give, because shrinking it
    # would drop head dimensions silently. BLOCK_N is the KV tile walked inside a
    # cache block, capped at 128 -- past the block size a wider tile only masks
    # off more.
    #
    # Unlike the other backends' overrides this one does *not* narrow BLOCK_N to
    # fit a budget, and does not read `max_shared_mem` to decide: this backend
    # reports 1 for that field, which is a placeholder rather than a limit, and a
    # loop that halved against it would reach its floor and raise on every call.
    # The budget is real but it is the compiler's, and it is only visible as a
    # compile failure -- see ``_query_tile_cap`` for the sweep that makes it a
    # number again.
    BLOCK_D = triton.next_power_of_2(head_size)
    BLOCK_N = min(triton.next_power_of_2(kv_block_size), 128)

    # BLOCK_M is capped, and an atom wider than the cap is walked by several
    # programs -- one per BLOCK_M rows. The 16 is the generic launcher's floor
    # and ``tl.dot``'s: a narrower query tile is not a smaller kernel, it is one
    # that does not compile.
    cap = _query_tile_cap(BLOCK_D, BLOCK_N)
    BLOCK_M = max(min(triton.next_power_of_2(q_len_max), cap), 16)
    n_q_tiles = (q_len_max + BLOCK_M - 1) // BLOCK_M

    # Size the CTA to the accumulator. ``acc`` is BLOCK_M x BLOCK_D of fp32 and
    # lives for the whole KV walk, so the per-thread share of it is what decides
    # whether the kernel spills. Four warps is the floor; above it, one warp per
    # 2048 accumulator elements holds the per-thread share near 64.
    #
    # This is the generic rule, kept unchanged, and unlike Hygon's and MetaX's it
    # needs no backend constant. Both of those scale the divisor by their
    # device's warp width, because their warps are 64 threads and the rule is
    # really a count of *threads*; this backend does not report ``warpSize`` at
    # all. Measured, it does not matter here: at the tilings this launcher picks,
    # 4 warps and 8 are within 0.5% of each other (65.4 against 65.1 ms at a
    # 128x64 tile, 73.8 against 73.7 at 128x128-ish, 130.8 against 131.0 at
    # 64x128), and the UB figure is identical either way. So the rule stays as
    # the A100 measurement wrote it.
    num_warps = max(4, (BLOCK_M * BLOCK_D) // 2048)

    _blocked_flash_fwd_kernel[(num_atoms * num_heads_q * n_q_tiles,)](
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
        # The flat grid's axes: the kernel takes the query-tile index as the
        # fastest, and needs the head count and the tile count to unwrap the rest.
        num_heads_q,
        n_q_tiles,
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
