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
"""Flash attention forward over a blocked KV-cache, driven by attention atoms.

Port of DeepSpeed's ``flash_attn_by_atoms`` -- its name upstream, kept here only
in this line, since the entry point below is ``blocked_flash``
(``deepspeed/inference/v2/kernels/ragged_ops/blocked_flash``). The forward pass
is the whole operator: the DeepSpeed kernel has no backward.

An *attention atom* describes the work of one query tile against one sequence's
paged KV-cache. It carries the query rows to run, the KV blocks to read, and the
query's position in the sequence -- which is what makes the causal mask work for
a continuation, where the first query token starts partway into the KV.

Atom layout, ``atoms`` of shape ``[num_atoms, 8]`` int32, matching DeepSpeed's
``AttentionAtom`` field-for-field except for the block list:

    [0] offset of this atom's block indices into ``kv_block_idx``
    [1] unused (DeepSpeed packs a 64-bit host pointer across slots 0-1)
    [2] q_start_idx   first query row of the atom, indexing ``q``/``out``
    [3] q_len         number of query rows in the atom
    [4] kv_blocks     number of KV blocks the atom reads
    [5] total_extent  number of valid KV tokens (the last block may be partial)
    [6] global_q_idx  sequence position of the atom's first query token
    [7] unused

The deviation from DeepSpeed is slot [0]: a device kernel cannot dereference the
host pointer DeepSpeed stores there, so this port takes the block indices as an
explicit ``kv_block_idx`` tensor and keeps an offset instead. Fields [2]-[6]
keep DeepSpeed's byte positions, so only the block-list access changes.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry

logger = logging.getLogger(__name__)

# Atom field indices, in int32 slots. See the module docstring for meanings.
#
# ``tl.constexpr`` rather than plain ints so the kernel can read them too: Triton
# resolves a global inside ``@triton.jit`` only when it was instantiated as
# constexpr, and rejects bare ints at compile time. Host code that needs a real
# index wraps them again (``int(...)``), since ``torch`` indexing will not accept
# a constexpr. Keeping one definition is what stops the slot numbers in the
# kernel from drifting away from the layout documented above.
_ATOM_BLOCK_OFFSET = tl.constexpr(0)
_ATOM_Q_START = tl.constexpr(2)
_ATOM_Q_LEN = tl.constexpr(3)
_ATOM_KV_BLOCKS = tl.constexpr(4)
_ATOM_TOTAL_EXTENT = tl.constexpr(5)
_ATOM_GLOBAL_Q_IDX = tl.constexpr(6)
_ATOM_STRIDE = tl.constexpr(8)

# Slots no field uses. They are written anyway, so an atom never carries whatever
# happened to be in the buffer: upstream's slot [1] is the high half of a 64-bit
# host pointer, which reads as 0 only when that pointer is null, and slot [7] is
# never written at all.
_ATOM_PTR_HIGH = tl.constexpr(1)
_ATOM_UNUSED = tl.constexpr(7)

# log2(e): the kernel's softmax runs on exp2, so this rides on the qk scale.
_LOG2E = 1.4426950408889634


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
    IS_CAUSAL: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    atom = tl.program_id(0)
    hq = tl.program_id(1)

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


def _device_capability():
    """``(major, minor)`` of the current device, or ``None`` if it will not say.

    ``None`` means "cannot tell", and the caller then leaves the device alone
    rather than guessing a limit it did not state.
    """
    try:
        return torch_device_fn.get_device_capability(torch_device_fn.current_device())
    except Exception:
        return None


def _check_launch_arguments(q, k, v, head_size, num_heads_q, num_heads_kv):
    """Reject inputs the kernel cannot serve.

    Mirrors the reference launcher's ``TORCH_CHECK``s (``blocked_flash.cpp:33-59``)
    so a caller gets the same diagnosis the reference gives, instead of a failure
    deep inside the Triton compiler or the driver.

    Two of them cannot be literal on a port that runs on several vendors:

    * upstream asserts ``q.is_cuda()``, which is only how a CUDA-only launcher
      spells "these must be device tensors"; here the test is that they are not
      on the CPU.
    * upstream asserts Ampere-or-newer, which is how it spells "this needs
      tensor-core hardware". That still holds, but it is only checked when the
      device actually reports a capability -- a backend that will not say keeps
      working, because inventing a limit it never stated is how a port rejects a
      device that was fine.

    The rest are the launcher's rules unchanged.
    """
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.device.type == "cpu":
            raise ValueError(f"{name} must be a device tensor, not on the CPU")

    if head_size > 256:
        raise ValueError(f"head_size must be <= 256, got {head_size}")
    if head_size % 8 != 0:
        raise ValueError(f"head_size must be divisible by 8, got {head_size}")

    capability = _device_capability()
    if capability is not None and capability[0] < 8:
        raise ValueError(
            "blocked flash needs a compute capability of 8.0 or newer; this "
            "device reports %d.%d" % capability
        )

    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("blocked flash attention only supports fp16 and bf16")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("query, key and value must share a dtype")
    if num_heads_q % num_heads_kv != 0:
        raise ValueError("n_heads_q must be divisible by n_heads_kv")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError(
            "query, key and value must be contiguous in the last dimension"
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

    This is the generic implementation, and the one every backend without an
    override runs. A backend with a specialised kernel replaces this *name*
    through ``runtime.backend.SpecOpRegistrar`` when ``flag_train.deepspeed`` is
    imported, so that is where to import the operator from -- importing it from
    this module gets the generic kernel on every backend.
    """
    logger.debug("TRAIN BLOCKED_FLASH")

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
    # walked inside a single cache block, capped at 128: past the block size a
    # wider tile only masks off more. BLOCK_D is the whole head, padded up.
    #
    # These three decide the kernel's shared-memory footprint, which is the one
    # limit that actually bites. A program holds BLOCK_M*BLOCK_D queries,
    # BLOCK_N*BLOCK_D keys and values, and BLOCK_M*BLOCK_N of fp32 scores at
    # once; at head_size 256 that is 208 KB for a 128-row tile against the A100's
    # 163 KB, so the launch fails with OutOfResources and the case needs smaller
    # atoms. None of the three is a caller-facing knob -- the atom builder is what
    # moves BLOCK_M, via q_block_size.
    BLOCK_M = max(triton.next_power_of_2(q_len_max), 16)
    BLOCK_N = min(triton.next_power_of_2(kv_block_size), 128)
    BLOCK_D = triton.next_power_of_2(head_size)

    # Size the CTA to the accumulator. ``acc`` is BLOCK_M x BLOCK_D of fp32 and
    # lives in registers for the whole KV walk, so the per-thread share of it is
    # what decides whether the kernel spills -- and capping registers with
    # ``maxnreg`` to buy occupancy is catastrophic here (128x128 goes from 706 us
    # to 3510 us at maxnreg=128), which says the kernel needs the registers it
    # asks for. Four warps is the floor; above it, one warp per 2048 accumulator
    # elements holds the per-thread share near 64. Measured on an A100: 128x64
    # wants 4 warps, 128x128 wants 8.
    num_warps = max(4, (BLOCK_M * BLOCK_D) // 2048)

    _blocked_flash_fwd_kernel[(num_atoms, num_heads_q)](
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
