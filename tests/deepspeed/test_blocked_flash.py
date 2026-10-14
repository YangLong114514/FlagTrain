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
"""Correctness tests for the blocked (paged KV-cache) flash attention forward.

Which implementation this port is *answerable to* is picked by the same
three-tier rule ``benchmark/deepspeed/test_blocked_flash.py`` picks its baseline
by, branch for branch, and both files name the tier ``_ORACLE`` so the two
suites describe one rule instead of two. See the comment above ``_ORACLE``.

Three implementations are compared against, because they fail differently:

* ``blocked_flash_ref`` -- a plain-torch composition that reads the *same*
  atoms and the *same* paged cache. It is checked on every tier and runs on any
  device, so the operator is testable off NVIDIA.
* DeepSpeed's ``BlockedFlashAttn`` -- the kernel this port is a port *of*, run on
  the same paged cache and atoms. It cannot express ``is_causal=False``, which is
  this port's own extension.
* ``flash_attn_varlen_func`` -- the oracle DeepSpeed's own
  ``test_blocked_flash.py`` uses, over the densely gathered KV -- an independent
  implementation of the contract. Its version is recorded in
  ``_FLASH_ATTN_VERSION``, as tests/deepspeed/README.md asks for.

The tier decides which oracle the port is *measured against*. It does not
suppress the others, and that is not a contradiction: none of the three implies
another. The two paged oracles could be wrong the same way about the paging,
which the dense one cannot see by construction, and all three could share a
misreading of the causal offset that only ``blocked_flash_ref`` states outright.
So every oracle that can run does run, and the tier's job is narrower -- it says
which *absence* is a fault, so that a tier losing its oracle gets reported rather
than getting quietly thinner.

Cases are seeded, so a failure reproduces rather than depending on the draw.
"""

import ctypes
import importlib

import pytest
import torch

import flag_train
from flag_train.deepspeed import blocked_flash
from flag_train.deepspeed.blocked_flash import (
    _ATOM_BLOCK_OFFSET,
    _ATOM_GLOBAL_Q_IDX,
    _ATOM_KV_BLOCKS,
    _ATOM_PTR_HIGH,
    _ATOM_Q_LEN,
    _ATOM_Q_START,
    _ATOM_STRIDE,
    _ATOM_TOTAL_EXTENT,
    _ATOM_UNUSED,
)

from .. import accuracy_utils as utils
from ..conftest import TO_CPU

# Which oracle this platform is answerable to, by availability rather than by
# preference:
#
#   1. a backend in ``_DEEPSPEED_BASELINE_VENDORS`` runs DeepSpeed's own kernel.
#      It is the kernel this port is a port *of*, and it is *required*: the
#      loader below raises rather than letting the tier disappear quietly.
#   2. any other backend with ``flash_attn`` installed runs
#      ``flash_attn_varlen_func``. Requiring it costs nothing to state, because a
#      backend reaches this tier only *because* the import succeeded.
#   3. a backend with neither falls back to ``blocked_flash_ref``, the plain-torch
#      composition -- which asks nothing of the platform, and is also checked on
#      the two tiers above.
#
# Tiers 1 and 2 are therefore the ones a backend can be *held* to, and tier 3 is
# the floor. ``_ORACLE`` below is computed from this same three-branch shape,
# in this same order, as the benchmark's baseline choice -- the rule lives in two
# files because the suites are separate, and the shared vocabulary is what makes
# a divergence between them legible.
#
# The reference kernel is not compiled from source here: ``RaggedOpsBuilder``
# links ``-lblockedflash`` against a prebuilt library shipped in the ``dskernels``
# package (``op_builder/ragged_ops.py``).
_DEEPSPEED_BASELINE_VENDORS = {"nvidia"}


def _load_deepspeed_blocked_flash():
    """``(BlockedFlashAttn, DtypeEnum)``, or ``None`` on a backend that does not use it."""
    if flag_train.vendor_name not in _DEEPSPEED_BASELINE_VENDORS:
        return None

    try:
        from deepspeed.inference.v2.inference_utils import DtypeEnum
        from deepspeed.inference.v2.kernels.ragged_ops import BlockedFlashAttn

        return BlockedFlashAttn, DtypeEnum
    except Exception as exc:
        raise RuntimeError(
            f"{flag_train.vendor_name!r} must use DeepSpeed's BlockedFlashAttn as "
            f"its oracle, but it could not be loaded: {exc!r}. Its kernel ships "
            f"prebuilt in the `dskernels` package (`op_builder/ragged_ops.py` links "
            f"`-lblockedflash`), so install that for this platform, or drop the "
            f"backend from _DEEPSPEED_BASELINE_VENDORS."
        ) from exc


# Resolved once, at module import time.
_deepspeed_blocked_flash = _load_deepspeed_blocked_flash()

try:
    import flash_attn
    from flash_attn.flash_attn_interface import flash_attn_varlen_func

    _HAS_FLASH_ATTN = True
    _FLASH_ATTN_VERSION = flash_attn.__version__
except ImportError:
    _HAS_FLASH_ATTN = False
    _FLASH_ATTN_VERSION = None

needs_flash_attn = pytest.mark.skipif(
    not _HAS_FLASH_ATTN, reason="flash_attn is not installed (CUDA-only oracle)"
)

# The tier, named. Compared by name rather than by identity so the two pinning
# tests below, and the benchmark's printed line, all say the same word.
if _deepspeed_blocked_flash is not None:
    _ORACLE = "deepspeed"
elif _HAS_FLASH_ATTN:
    _ORACLE = "flash_attn"
else:
    _ORACLE = "torch_ref"

# Tolerances come from the reference implementation, not from this repository's
# generic per-dtype table: ``inference_test_utils.get_tolerances`` in DeepSpeed is
# what the operator being ported is actually held to, and it is considerably
# looser than the house defaults (fp16 atol 2e-3 against 1e-4). Matching it is
# the requirement; anything tighter would be measuring rounding that no fp16
# tensor-core attention kernel can avoid -- the softmax probabilities have to be
# rounded into the compute dtype before the PV dot, which costs ~5e-4 of absolute
# accuracy for fp16 and more for bf16's eight mantissa bits.
#
# The same pair covers every oracle. The ``flash_attn`` cross-check is a
# comparison between two independent fp16 kernels and is bounded by the same
# argument, so it needs no separate tolerance.
_TOLERANCES = {
    torch.float16: (3e-2, 2e-3),  # (rtol, atol), as DeepSpeed orders them
    torch.bfloat16: (4.8e-1, 3.2e-2),
}


def _assert_close(res, ref, dtype, rtol, atol):
    """Compare against ``ref`` at the reference implementation's tolerances.

    Deliberately not ``flag_train.testing.assert_close``: that helper applies
    this repository's own per-dtype ``RESOLUTION``, and the whole point here is
    to hold the operator to DeepSpeed's looser number instead -- so the number
    belongs at the call site, where it is visible next to ``_TOLERANCES`` and
    the argument for it, rather than threaded through the house helper and
    invisible to everyone else who reads that helper's contract.
    """
    res = utils.to_reference(res)
    ref = utils.to_reference(ref)
    assert res.dtype == dtype
    ref = ref.to(dtype)
    torch.testing.assert_close(res, ref, atol=atol, rtol=rtol)


# The reference has to be a true fp32 reference. NGC images ship
# ``TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1``, which silently runs every fp32 matmul
# in tf32 -- ten mantissa bits, ~5e-4 relative error. The cost lands on the
# *reference*, not the kernel: measured against an fp64 reference the kernel is
# off by 2.7e-7 on average at every output magnitude, while the tf32 reference is
# off by up to 1.9e-3. Left enabled, the suite measures the reference's error and
# reports it against the kernel. ``benchmark/base.py`` already switches it off;
# the tests did not.
torch.backends.cuda.matmul.allow_tf32 = False

_HEAD_SIZE = 64
_HEAD_SIZE_MAX = 256  # the launcher's validated ceiling (blocked_flash.cpp)
_Q_BLOCK_SIZE = 128
_KV_BLOCK_SIZE = 64


# ---------------------------------------------------------------------------
# Reference implementation
#
# A plain-torch composition of the same contract, reading the *same* atoms and
# the *same* paged cache as the kernel -- so it checks the paging and the causal
# offset rather than restating the kernel's tiling. It is checked on every tier,
# and it runs on any device, which is what makes the operator testable off
# NVIDIA: the dense oracle below is CUDA-only. It is also the only one of the
# three that can be pushed onto the CPU under ``--ref cpu``, which is what makes
# it the oracle on a backend that has neither of the other two.
#
# !! KEEP IN SYNC with benchmark/deepspeed/test_blocked_flash.py !!
#
# That file carries a second, byte-identical copy, used as the baseline when
# `flash_attn` is absent. The duplication is deliberate -- a reference belongs to
# the tests that check against it, not to the operator package, so that a checked
# implementation is never code the operator ships. The cost is that the two
# copies can drift: change one and you must change the other, or the benchmark
# will quietly measure against a different contract than the tests accept.
# ---------------------------------------------------------------------------


def blocked_flash_ref(
    out,
    q,
    k,
    v,
    attention_atoms,
    kv_block_idx,
    softmax_scale,
    is_causal=True,
):
    """Reference for the ``blocked_flash`` operator, from plain torch ops.

    Walks the atoms one at a time, gathers each atom's KV out of the paged cache
    into a dense window, and runs the attention in torch.
    """
    head_size = k.size(-1)
    n_heads_q = q.size(-1) // head_size
    n_heads_kv = k.size(-2)
    group = n_heads_q // n_heads_kv
    device = q.device

    for atom in attention_atoms.tolist():
        block_offset, _, q_start, q_len, kv_blocks, total_extent, global_q_idx, _ = atom
        if q_len == 0 or kv_blocks == 0:
            continue

        blocks = kv_block_idx[block_offset : block_offset + kv_blocks].long()
        # Gather the atom's blocks, drop the padding past total_extent, and repeat
        # each KV head for the query heads that share it.
        k_atom = k[blocks].reshape(-1, n_heads_kv, head_size)[:total_extent]
        v_atom = v[blocks].reshape(-1, n_heads_kv, head_size)[:total_extent]
        k_atom = k_atom.repeat_interleave(group, dim=1).transpose(0, 1).float()
        v_atom = v_atom.repeat_interleave(group, dim=1).transpose(0, 1).float()

        q_atom = q[q_start : q_start + q_len].reshape(q_len, n_heads_q, head_size)
        q_atom = q_atom.transpose(0, 1).float()

        scores = torch.matmul(q_atom, k_atom.transpose(-1, -2)) * softmax_scale
        if is_causal:
            # The mask compares sequence positions, so a continuation's first query
            # is offset by the history it was given rather than starting at zero.
            q_pos = torch.arange(q_len, device=device) + global_q_idx
            k_pos = torch.arange(total_extent, device=device)
            scores = scores.masked_fill(
                k_pos[None, None, :] > q_pos[None, :, None], float("-inf")
            )

        probs = torch.softmax(scores, dim=-1)
        atom_out = torch.matmul(probs, v_atom).transpose(0, 1)
        out[q_start : q_start + q_len] = atom_out.reshape(
            q_len, n_heads_q * head_size
        ).to(out.dtype)

    return out


# ---------------------------------------------------------------------------
# Atom-building convenience
#
# !! KEEP IN SYNC with benchmark/deepspeed/test_blocked_flash.py !!
#
# That file carries a second copy of this function. Like the reference above it
# is scaffolding rather than operator API, so it lives with the tests that use it
# rather than in the package -- ``flag_train.deepspeed``'s surface stays to the
# names the reference implementation exposes. The cost is the usual one: change
# this copy and you must change that one, or the benchmark will build their atoms a
# different way than the tests accept.
# ---------------------------------------------------------------------------

# The atom layout's stride, in int32 slots. See ``blocked_flash.py``'s module
# docstring for the field meanings; the reference above already unpacks the
# eight slots positionally, so this is the same knowledge given a name.
_ATOM_SLOTS = 8


def build_blocked_flash_atoms(seq_params, q_block_size, kv_block_size, device):
    """Atoms for a batch of ``(q_len, history_len)`` sequences, and their blocks.

    ``AtomBuilder`` wants a batch and a *host* buffer -- the atoms live in host
    memory upstream because slot [0] holds a pointer there -- and its output then
    has to be moved to the device. This is those three steps in one call.

    Returns:
        tuple: ``(atoms, kv_block_idx)``, the ``[n_atoms, 8]`` int32 atom tensor
        and the flat int32 list of physical block indices, both on ``device``.
    """

    batch = RaggedBatchWrapper(seq_params, kv_block_size)

    max_atoms = sum(
        (q_len + q_block_size - 1) // q_block_size for q_len, _ in seq_params
    )
    # A host tensor: the builder writes atoms where the layout says to, and slot
    # [0] is a pointer upstream, which is why the buffer is not on the device.
    atoms = torch.zeros(
        (max_atoms, _ATOM_SLOTS), dtype=torch.int32, device=torch.device("cpu")
    )
    atoms, kv_block_idx, n_atoms = AtomBuilder()(
        atoms, batch, q_block_size, kv_block_size
    )

    return (
        atoms[:n_atoms].to(device=device).contiguous(),
        kv_block_idx.to(device=device),
    )


def _deepspeed_atoms(seq_params, q_block_size, kv_block_size, device):
    """Atoms in the reference layout, for the DeepSpeed oracle.

    Its kernel dereferences slot [0] as a *host pointer* to the block list, over
    UVA, so the list has to live in pinned memory and each atom has to carry the
    address of its own sequence's run. ``build_blocked_flash_atoms`` writes an
    offset there instead -- a Triton kernel cannot dereference a host pointer
    (blocked_flash.md §5) -- so this turns the offset back into the pointer.

    Returns:
        tuple: ``(atoms, pinned)``. ``pinned`` comes back because the atoms point
        into it: letting it go would leave the kernel reading freed memory.
    """
    atoms, kv_block_idx = build_blocked_flash_atoms(
        seq_params, q_block_size, kv_block_size, device
    )
    pinned = kv_block_idx.cpu().pin_memory()
    base = pinned.data_ptr()
    addresses = torch.tensor(
        [base + 4 * offset for offset in atoms[:, 0].tolist()],
        dtype=torch.int64,
        device=device,
    )
    atoms[:, 0:2] = addresses.view(torch.int32).reshape(-1, 2)
    return atoms, pinned


_deepspeed_kernels = {}


def _deepspeed_kernel(head_size, dtype):
    """One kernel per (head_size, dtype); the constructor is what loads the op."""
    key = (head_size, dtype)
    if key not in _deepspeed_kernels:
        BlockedFlashAttn, DtypeEnum = _deepspeed_blocked_flash
        _deepspeed_kernels[key] = BlockedFlashAttn(head_size, DtypeEnum(dtype))
    return _deepspeed_kernels[key]


def _deepspeed_geometry(head_size):
    """``(q_block_size, kv_block_size)`` DeepSpeed's kernel is *built* for.

    Neither is a parameter of its launcher. ``blocked_flash.cpp`` sets
    ``k_row_stride`` from ``k.stride(1)``, so the row pitch is read off the tensor
    -- but the number of tokens in a cache block never is, because it is a
    template argument of the kernel. Upstream fixes it with these two helpers,
    which is why its own test calls them rather than choosing either value.

    This matters here because the cache this suite pages is sized by
    ``kv_block_size``, which the test chooses freely. Handed a cache whose block
    size is not the one the kernel was built for, the kernel walks it at the
    wrong pitch: it reads the wrong tokens, and since the block list it follows
    then runs off the end of the sequence's blocks, it can touch memory that is
    not the sequence's at all. So the geometry is a precondition of asking this
    oracle anything, not a preference.

    Imported lazily: the module is only reachable when the oracle loaded, which
    is the only situation that calls this.
    """
    from deepspeed.inference.v2.kernels.ragged_ops.blocked_flash.blocked_flash import (
        get_kv_block_size,
        get_q_block_size,
    )

    return get_q_block_size(head_size), get_kv_block_size(head_size)


def _dense_reference(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_kv,
    seq_params,
    head_size,
    n_heads_q,
    n_heads_kv,
    softmax_scale,
    causal=True,
):
    """Varlen flash attention over the densely gathered KV."""
    return flash_attn_varlen_func(
        q.reshape(-1, n_heads_q, head_size),
        k.reshape(-1, n_heads_kv, head_size),
        v.reshape(-1, n_heads_kv, head_size),
        cu_seqlens_q,
        cu_seqlens_kv,
        max(s[0] for s in seq_params),
        max(s[0] + s[1] for s in seq_params),
        softmax_scale=softmax_scale,
        causal=causal,
    ).reshape(-1, n_heads_q * head_size)


def _run_case(
    seq_params,
    head_size=_HEAD_SIZE,
    n_heads_q=16,
    n_heads_kv=16,
    kv_block_size=_KV_BLOCK_SIZE,
    q_block_size=_Q_BLOCK_SIZE,
    softmax_scale=1.0,
    dtype=torch.float16,
    is_causal=True,
    permute_blocks=False,
    out_pad_columns=0,
    require_deepspeed=False,
):
    """Page the KV for ``seq_params``, run the operator, and check every oracle.

    ``permute_blocks`` and ``out_pad_columns`` exist to reach states the rest of
    the suite never produces -- see the tests that set them.

    ``require_deepspeed`` asserts that the DeepSpeed oracle was actually
    compared. It is off by default because most cases cannot be: the kernel is
    templated on a fixed cache block size, so it only answers for a cache built
    the way it expects (see ``_deepspeed_geometry``), and the suite deliberately
    varies that size. A caller that means to prove the tier-1 comparison still
    happens has to say so, or a case that quietly landed outside the geometry
    would report success for a comparison that never ran.
    """
    device = flag_train.device
    # Seeded per case, so a failure reproduces instead of depending on the draw.
    torch.manual_seed(0)

    total_q = sum(q_len for q_len, _ in seq_params)
    n_blocks = sum(
        (history + q_len + kv_block_size - 1) // kv_block_size
        for q_len, history in seq_params
    )

    qkv = torch.randn(
        (total_q, (n_heads_q + 2 * n_heads_kv) * head_size), dtype=dtype, device=device
    )
    q = qkv[:, : n_heads_q * head_size]
    inflight_kv = qkv[:, n_heads_q * head_size :]

    # The cache already holds the history; the current tokens are appended to it,
    # so a continuation's query row i sits at KV position history + i.
    history = [
        (
            torch.randn((h, 2 * n_heads_kv * head_size), dtype=dtype, device=device)
            if h > 0
            else None
        )
        for _, h in seq_params
    ]

    k_cache = torch.zeros(
        (n_blocks, kv_block_size, n_heads_kv, head_size), dtype=dtype, device=device
    )
    v_cache = torch.zeros_like(k_cache)

    cu_seqlens_q = [0]
    cu_seqlens_kv = [0]
    full_kvs = []
    first_block = 0
    for i, (q_len, history_len) in enumerate(seq_params):
        cur = inflight_kv[cu_seqlens_q[i] : cu_seqlens_q[i] + q_len]
        full = cur if history[i] is None else torch.cat([history[i], cur], dim=0)
        full_kvs.append(full)

        n_seq_blocks = (full.shape[0] + kv_block_size - 1) // kv_block_size
        padded = torch.zeros(
            (n_seq_blocks * kv_block_size, full.shape[1]), dtype=dtype, device=device
        )
        padded[: full.shape[0]] = full
        paged = padded.reshape(n_seq_blocks, kv_block_size, 2 * n_heads_kv * head_size)
        k_cache[first_block : first_block + n_seq_blocks] = paged[
            :, :, : n_heads_kv * head_size
        ].reshape(n_seq_blocks, kv_block_size, n_heads_kv, head_size)
        v_cache[first_block : first_block + n_seq_blocks] = paged[
            :, :, n_heads_kv * head_size :
        ].reshape(n_seq_blocks, kv_block_size, n_heads_kv, head_size)
        first_block += n_seq_blocks

        cu_seqlens_q.append(cu_seqlens_q[-1] + q_len)
        cu_seqlens_kv.append(cu_seqlens_kv[-1] + full.shape[0])

    atoms, kv_block_idx = build_blocked_flash_atoms(
        seq_params, q_block_size, kv_block_size, device
    )

    rtol, atol = _TOLERANCES[dtype]

    if permute_blocks:
        # A cache manager hands out whatever physical blocks are free, so a
        # sequence's blocks are not contiguous in general, and the indirection
        # through ``kv_block_idx`` is there for exactly that. Every other case
        # reaches the kernel through ``build_blocked_flash_atoms``, which always
        # numbers blocks in order -- so without this the indirection is only ever
        # exercised where it is the identity.
        permutation = torch.randperm(n_blocks, device=device)
        k_cache = k_cache[permutation].contiguous()
        v_cache = v_cache[permutation].contiguous()
        # ``cache[permutation]`` reads as ``new[i] = old[permutation[i]]``, so the
        # block that used to sit at physical ``p`` is now at ``argsort(permutation)[p]``
        # -- the inverse, not ``permutation[p]``.
        inverse = torch.argsort(permutation)
        kv_block_idx = inverse[kv_block_idx.long()].to(kv_block_idx.dtype)

    if out_pad_columns:
        # The operator strides to each output row rather than assuming the rows
        # are packed, and upstream's own caller hands it a slice of a wider
        # buffer. ``q`` above is already such a view; ``out`` was not.
        out_buffer = torch.zeros(
            (total_q, n_heads_q * head_size + out_pad_columns),
            dtype=dtype,
            device=device,
        )
        out = out_buffer[:, : n_heads_q * head_size]
    else:
        out = torch.zeros((total_q, n_heads_q * head_size), dtype=dtype, device=device)
    blocked_flash(
        out,
        q,
        k_cache,
        v_cache,
        atoms,
        kv_block_idx,
        softmax_scale,
        is_causal=is_causal,
    )

    # Under ``--ref cpu`` the reference is genuinely computed on the CPU, not
    # computed on the GPU and moved: ``blocked_flash_ref`` is plain torch, so
    # feeding it CPU operands runs it without touching a single GPU kernel --
    # which makes it an independent oracle rather than a second reading of the
    # same device arithmetic. ``to_reference`` then only has to move the
    # operator's own output across for the comparison.
    if TO_CPU:
        ref = torch.zeros_like(out).cpu()
        blocked_flash_ref(
            ref,
            q.cpu(),
            k_cache.cpu(),
            v_cache.cpu(),
            atoms.cpu(),
            kv_block_idx.cpu(),
            softmax_scale,
            is_causal=is_causal,
        )
    else:
        ref = torch.zeros_like(out)
        blocked_flash_ref(
            ref,
            q,
            k_cache,
            v_cache,
            atoms,
            kv_block_idx,
            softmax_scale,
            is_causal=is_causal,
        )

    _assert_close(out, ref, dtype, rtol, atol)

    # ``flash_attn_varlen_func`` over the densely gathered KV, checked whenever
    # the package imported -- ``--ref cpu`` included. ``--ref cpu`` moves the
    # operands of the *reference* to the CPU; this is not the reference. It is a
    # device oracle comparing two device tensors, and ``q``/``k_cache``/
    # ``v_cache`` stay on the device under either mode, because the CPU path
    # below copies them rather than rebinding them. Gating this on ``TO_CPU``
    # would let the oracle a ``flash_attn`` backend is *answerable to* disappear
    # under a flag that says nothing about that backend.
    if _HAS_FLASH_ATTN:
        run_kvs = torch.cat(full_kvs, dim=0)
        dense = _dense_reference(
            q,
            run_kvs[:, : n_heads_kv * head_size],
            run_kvs[:, n_heads_kv * head_size :],
            torch.tensor(cu_seqlens_q, dtype=torch.int32, device=device),
            torch.tensor(cu_seqlens_kv, dtype=torch.int32, device=device),
            seq_params,
            head_size,
            n_heads_q,
            n_heads_kv,
            softmax_scale,
            causal=is_causal,
        )
        _assert_close(out, dense, dtype, rtol, atol)

    # The reference implementation's own kernel, on the same inputs -- the oracle
    # this port is answerable to. Checked whenever it loaded, which on the
    # backends that require it is always.
    #
    # Two cases it cannot express, and they are different in kind:
    #
    # * ``is_causal=False``, which the reference class hardcodes to causal -- it
    #   has no way to say what this port's ``is_causal`` extension says. That is
    #   this port having *more* than the kernel.
    # * a cache whose ``q_block_size``/``kv_block_size`` are not the ones the
    #   kernel was built for (``_deepspeed_geometry``). That is this port having
    #   *different* geometry: the kernel would answer about a cache that is not
    #   this one, and read past the sequence's blocks doing it, so the comparison
    #   is refused rather than attempted.
    #
    # The second is also why ``require_deepspeed`` exists: for the ordinary
    # parametrized cases a refusal is correct, but a case whose whole purpose is
    # to prove tier 1 is still reached must not be able to pass by being refused.
    deepspeed_geometry_ok = _deepspeed_blocked_flash is not None and (
        _deepspeed_geometry(head_size) == (q_block_size, kv_block_size)
    )
    if require_deepspeed and not (deepspeed_geometry_ok and is_causal):
        raise AssertionError(
            f"case (head_size={head_size}, q_block_size={q_block_size}, "
            f"kv_block_size={kv_block_size}, is_causal={is_causal}) asks for the "
            f"DeepSpeed oracle to be compared, but it is one the oracle cannot "
            f"express. Pick the geometry from _deepspeed_geometry, and keep "
            f"is_causal True."
        )
    if deepspeed_geometry_ok and is_causal:
        deepspeed_atoms, deepspeed_pinned = _deepspeed_atoms(
            seq_params, q_block_size, kv_block_size, device
        )
        reference_out = torch.zeros_like(out)
        _deepspeed_kernel(head_size, dtype)(
            reference_out, q, k_cache, v_cache, deepspeed_atoms, softmax_scale
        )
        _assert_close(reference_out, ref, dtype, rtol, atol)


@pytest.mark.blocked_flash
def test_matches_deepspeed_oracle():
    """Pin tier 1 -- DeepSpeed's own kernel -- explicitly.

    A tier that quietly stopped being reached would thin the suite without
    saying so. On a backend in ``_DEEPSPEED_BASELINE_VENDORS`` the load itself
    cannot fail quietly, because the module raises at import; what this catches
    is the *comparison* ceasing to happen. That is not hypothetical -- it is what
    happened while this test shared its name with the tier-2 test below -- and
    ``require_deepspeed`` is what keeps it from coming back, since every other
    case in the suite declines this oracle on geometry for good reason.

    The skip serves the backends where tier 1 does not apply at all.
    """
    if _deepspeed_blocked_flash is None:
        pytest.skip(f"tier 1 is not this backend's oracle; this one is {_ORACLE!r}")

    # Ask the oracle which geometry it can answer for, rather than assuming one:
    # the case has to be inside it or ``require_deepspeed`` refuses.
    q_block_size, kv_block_size = _deepspeed_geometry(_HEAD_SIZE)
    _run_case(
        [(332, 628)],
        n_heads_q=32,
        n_heads_kv=32,
        q_block_size=q_block_size,
        kv_block_size=kv_block_size,
        require_deepspeed=True,
    )


@pytest.mark.blocked_flash
@pytest.mark.parametrize("n_tokens", [1, 2, 33, 65, 128, 256, 2037])
def test_single_prompt(n_tokens):
    """One prompt, no history: the KV is exactly the query's own tokens."""
    _run_case([(n_tokens, 0)])


@pytest.mark.blocked_flash
@pytest.mark.parametrize(
    "prompt_lengths", [(128, 128), (192, 38), (514, 713), (83, 312, 610)]
)
def test_multiple_prompts(prompt_lengths):
    """Several prompts in one batch, each with its own block run."""
    _run_case([(length, 0) for length in prompt_lengths])


@pytest.mark.blocked_flash
@pytest.mark.parametrize(
    "seq_params", [(1, 34), (43, 40), (1, 144), (64, 128), (332, 628)]
)
def test_continuation(seq_params):
    """A prompt continuing an existing KV: the causal mask has to offset the
    query's position by the history, not by its row index.

    Head counts are DeepSpeed's own for this case (32/32), kept so the two
    suites exercise the same configuration.
    """
    _run_case([seq_params], n_heads_q=32, n_heads_kv=32)


@pytest.mark.blocked_flash
@pytest.mark.parametrize(
    "head_size, q_block_size",
    [(_HEAD_SIZE, 128), (128, 128), (_HEAD_SIZE_MAX, 32)],
    ids=["64", "128", "256"],
)
def test_head_size(head_size, q_block_size):
    """64 and 128 are the sizes models use; 256 is the launcher's validated
    ceiling, exercised at a narrow query tile.

    256 *used* to need one: the kernel holds BLOCK_M * BLOCK_D of queries and
    BLOCK_N * BLOCK_D of keys at once, and with the KV tile capped at 128 a
    128-row tile asked for 224 KB against the A100's 163 KB -- measured, and it
    did fail with OutOfResources. Capping the KV tile at 64 brings the same case
    to 144 KB, which launches, so the narrow tile here is no longer a
    requirement. It is kept because a second query tile is worth covering, not
    because the wide one is unavailable; see ``_BLOCK_N_MAX`` in
    ``flag_train.deepspeed.blocked_flash``.
    """
    _run_case(
        [(128, 128), (192, 38), (1, 814)],
        head_size=head_size,
        q_block_size=q_block_size,
    )


@pytest.mark.blocked_flash
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_dtype(dtype):
    """Both dtypes the launcher accepts must clear every oracle."""
    _run_case([(128, 128), (192, 38), (1, 814)], dtype=dtype)


@pytest.mark.blocked_flash
@pytest.mark.parametrize("seq_params", [(128, 0), (1, 34)], ids=["prompt", "one_token"])
def test_non_causal(seq_params):
    """``is_causal=False`` is a separate constexpr branch in the kernel, so it
    needs a case of its own: the mask has to be absent rather than narrower.

    Only single-atom sequences are covered, and that is the contract rather than
    a gap in the test. ``build_blocked_flash_atoms`` sizes an atom's KV window to
    ``total_extent = end_toks``, which is sufficient *because* a causal mask
    hides every position past it. Drop the mask and an atom that is not the
    sequence's last would need KV the builder never handed it: at
    ``(332, 628)`` with ``q_block_size=128`` the first atom sees 768 of the
    sequence's 960 tokens, and the dense oracle correctly disagrees. A
    multi-atom ``is_causal=False`` run is therefore not a supported pairing
    until the builder's window sizing takes the mask into account.
    """
    _run_case([seq_params], is_causal=False)


@pytest.mark.blocked_flash
@pytest.mark.parametrize("head_config", [(32, 8), (64, 16), (40, 8)])
def test_gqa(head_config):
    """Grouped-query attention: several query heads share one KV head.

    head_size 128 is DeepSpeed's choice for this case, kept so the two suites
    exercise the same configuration.
    """
    n_heads_q, n_heads_kv = head_config
    _run_case(
        [(128, 128), (192, 38), (1, 814)],
        head_size=128,
        n_heads_q=n_heads_q,
        n_heads_kv=n_heads_kv,
    )


@pytest.mark.blocked_flash
@pytest.mark.parametrize("kv_block_size", [32, 64, 256])
def test_kv_block_size(kv_block_size):
    """The cache's block size must not change the answer; 256 also makes the
    kernel tile a single block in more than one pass."""
    _run_case([(128, 128), (192, 38), (1, 814)], kv_block_size=kv_block_size)


@pytest.mark.blocked_flash
def test_fully_composed():
    """A batch mixing long continuations, short prompts and an empty history."""
    _run_case([(332, 628), (1, 718), (1, 323), (180, 5), (224, 0)])


@pytest.mark.blocked_flash
@needs_flash_attn
@pytest.mark.parametrize("seq_params", [(332, 628), (64, 128)])
def test_matches_flash_attn_oracle(seq_params):
    """Pin tier 2 -- ``flash_attn_varlen_func`` -- explicitly, so a run that
    quietly stopped reaching it is visible rather than silently thinner.

    The name has to be its own. This test used to be spelled
    ``test_matches_deepspeed_oracle``, which is the tier-1 test's name, and a
    second ``def`` of a module-level name *replaces* the first: pytest collects
    what the module holds, not what it defined, so the tier-1 pin was not merely
    unreachable, it was never collected. One pin per tier, one name per pin.
    """
    _run_case([seq_params])


# ---------------------------------------------------------------------------
# Atom layout: checking what cannot be read, and converting what can
#
# Slot [0] is an offset into ``kv_block_idx`` in this port and a host pointer
# upstream, so the two layouts are not interchangeable even though the rest of
# the record is byte-identical. The kernel indexes with slot [0] unguarded, so
# handing it the wrong layout is an illegal memory access rather than a wrong
# number -- and an illegal access does not fail one test, it takes the CUDA
# context down, so every later test in the same process fails for a reason that
# has nothing to do with it. That is how a single wrong cache block size turned
# into 37 failures across this file.
#
# Both tools below live here rather than in the operator module, on the rule
# COMPATIBILITY.md §0 states: a name kept in the package for the tests' benefit is
# the shape that let the previous ``BlockedFlashAttn`` rot unnoticed. They check
# and translate an operator's *arguments*, which makes them the caller's tools.
# The operator does not guard itself -- a check on every call costs a device sync,
# measured at ~45 us on an A100, more than the kernel at the shapes it serves --
# so validation is a caller's step, once per atom tensor.
# ---------------------------------------------------------------------------


def _host_atoms(attention_atoms):
    """``attention_atoms`` as a list of host rows -- one transfer, one sync.

    Whole rows rather than the columns a caller happens to want: the record is
    eight int32, so moving all of it costs nothing worth saving, while selecting
    columns first would add a kernel of its own and the round trip is the
    expensive part.

    ``tolist()`` rather than ``numpy()``: the fields are int32 and are meant to be
    read back as such, so a value DeepSpeed stored as the low half of a pointer
    comes back *negative* rather than as a large positive number, which is what
    makes the range check below able to see it at all.
    """
    return attention_atoms.detach().to("cpu").tolist()


def _check_atom_bounds(rows, numel):
    """Raise unless every atom's block window lies inside ``kv_block_idx``.

    Slot ``[0]`` is an *offset into ``kv_block_idx``* in this port and a 64-bit
    *host pointer* upstream (see the operator module's docstring). Handed
    upstream's atoms, the kernel would read ``kv_block_idx[-1354760192]`` -- an
    illegal memory access that takes the CUDA context down with it, so every later
    test in the same process fails for a reason unrelated to itself.

    The check is host-side because the alternative is worse. Clamping in the
    kernel would avoid the fault but answer *silently*, which for a wrong layout
    is the failure mode hardest to notice; and a device-side flag would need a
    sync to read back, which is the cost the caller is trying to pay only once.
    """
    for index, row in enumerate(rows):
        offset = row[int(_ATOM_BLOCK_OFFSET)]
        count = row[int(_ATOM_KV_BLOCKS)]
        if offset < 0 or offset + count > numel:
            raise ValueError(
                f"attention_atoms[{index}] addresses kv_block_idx out of range: "
                f"slot {int(_ATOM_BLOCK_OFFSET)} is {offset} and slot "
                f"{int(_ATOM_KV_BLOCKS)} is {count}, against a kv_block_idx of "
                f"{numel} element(s). Slot {int(_ATOM_BLOCK_OFFSET)} is an offset "
                f"into kv_block_idx here; DeepSpeed's atoms put the low and high "
                f"halves of a host pointer in slots "
                f"[{int(_ATOM_BLOCK_OFFSET)}, {int(_ATOM_PTR_HIGH)}] instead. Pass "
                f"those through from_deepspeed_atoms() first."
            )


def validate_attention_atoms(attention_atoms, kv_block_idx):
    """Check that ``attention_atoms`` are laid out the way ``blocked_flash`` reads them.

    Costs one device sync, so call it once per atom tensor rather than per step.
    The operator does not call it: it cannot afford to, and a caller that built
    the atoms knows which layout it built them in.

    Raises:
        ValueError: if the atoms' block windows do not fit ``kv_block_idx``,
            which is what DeepSpeed-layout atoms look like from here.
    """
    if attention_atoms.dim() != 2 or attention_atoms.size(1) != int(_ATOM_STRIDE):
        raise ValueError(
            f"attention_atoms must be [num_atoms, {int(_ATOM_STRIDE)}] int32, got "
            f"{tuple(attention_atoms.shape)}"
        )
    _check_atom_bounds(_host_atoms(attention_atoms), kv_block_idx.numel())


def from_deepspeed_atoms(attention_atoms, device=None):
    """Convert DeepSpeed-layout atoms into the layout ``blocked_flash`` reads.

    DeepSpeed's atoms carry a 64-bit *host pointer* to their block list across
    slots ``[0]`` and ``[1]``, in pinned memory it dereferences over UVA. A Triton
    kernel cannot follow that pointer, so this port addresses a device tensor
    instead (see the operator module's docstring). Everything else about the
    layout is identical, which is why this is a translation of one field rather
    than of the record.

    The atoms of one sequence share a pointer, so the lists are deduplicated by
    address: the result is one run per sequence, the shape
    ``build_blocked_flash_atoms`` produces, and for atoms built from the same
    cache the two agree exactly. Deduplicating is not just tidiness -- it is what
    makes the two agree, since the port's own builder emits one run per sequence
    too, with each sequence's atoms sharing an offset.

    Args:
        attention_atoms (Tensor): ``[num_atoms, 8]`` int32 atoms in DeepSpeed's
            layout, on the device or in the pinned host buffer they came in.
        device (optional): where to put the result. Defaults to this tensor's
            device, falling back to the active backend when it is a host buffer.

    Returns:
        tuple: ``(atoms, kv_block_idx)``, ready to pass to ``blocked_flash``.

    The pointer must still be live: this reads through it, so the buffer the
    atoms point into has to outlive the call. That is the same obligation
    DeepSpeed's own kernel carries, which is why its atom builder hands the
    buffer back to the caller to hold.
    """
    if device is None:
        device = (
            attention_atoms.device
            if attention_atoms.device.type != "cpu"
            else flag_train.device
        )

    host = attention_atoms.detach().to("cpu", torch.int32)
    rows = host.tolist()

    def address_of(row):
        return ((row[int(_ATOM_PTR_HIGH)] & 0xFFFFFFFF) << 32) | (
            row[int(_ATOM_BLOCK_OFFSET)] & 0xFFFFFFFF
        )

    addresses = [address_of(row) for row in rows]

    # A sequence's atoms share one pointer but *not* one ``kv_blocks``: an atom's
    # KV window is sized to the history its own query rows can see, so the last
    # atom of a sequence reads more blocks than the first (measured on a
    # three-atom sequence: 2, then 3). Sizing the run by whichever atom claims the
    # address first would cut it short and silently drop the blocks only the later
    # atoms asked for -- so each run is sized to the longest window read from it.
    lengths = {}
    order = []
    for address, row in zip(addresses, rows):
        if address not in lengths:
            lengths[address] = 0
            order.append(address)
        lengths[address] = max(lengths[address], row[int(_ATOM_KV_BLOCKS)])

    blocks = []
    offset_of = {}
    for address in order:
        offset_of[address] = len(blocks)
        if lengths[address]:
            blocks.extend(
                ctypes.cast(
                    address, ctypes.POINTER(ctypes.c_int32 * lengths[address])
                ).contents
            )

    converted = host.clone()
    converted[:, int(_ATOM_BLOCK_OFFSET)] = torch.tensor(
        [offset_of[address] for address in addresses], dtype=torch.int32
    )
    converted[:, int(_ATOM_PTR_HIGH)] = 0

    return (
        converted.contiguous().to(device=device),
        torch.tensor(blocks, dtype=torch.int32, device=device),
    )


def _page_one_case(seq_params, head_size=_HEAD_SIZE, n_heads_q=16, n_heads_kv=16):
    """Query and paged cache for one case, with every sequence starting empty.

    Only what the conversion test below needs: it goes on to build both atom
    layouts from this cache -- ``build_blocked_flash_atoms`` for the one this port
    reads, ``_deepspeed_atoms`` for the one upstream writes -- and compares them
    after a round trip through ``from_deepspeed_atoms``.
    """
    device = flag_train.device
    torch.manual_seed(0)
    total_q = sum(q_len for q_len, _ in seq_params)
    n_blocks = sum(
        (h + q + _KV_BLOCK_SIZE - 1) // _KV_BLOCK_SIZE for q, h in seq_params
    )
    qkv = torch.randn(
        (total_q, (n_heads_q + 2 * n_heads_kv) * head_size),
        dtype=torch.float16,
        device=device,
    )
    q = qkv[:, : n_heads_q * head_size]
    inflight = qkv[:, n_heads_q * head_size :]
    k_cache = torch.zeros(
        (n_blocks, _KV_BLOCK_SIZE, n_heads_kv, head_size),
        dtype=torch.float16,
        device=device,
    )
    v_cache = torch.zeros_like(k_cache)
    first, cursor = 0, 0
    for q_len, h_len in seq_params:
        # One sequence, no history, so the cache is exactly its own KV and the
        # case stays about the atom layout rather than about the paging.
        assert h_len == 0
        cur = inflight[cursor : cursor + q_len]
        cursor += q_len
        n_seq_blocks = (q_len + _KV_BLOCK_SIZE - 1) // _KV_BLOCK_SIZE
        padded = torch.zeros(
            (n_seq_blocks * _KV_BLOCK_SIZE, 2 * n_heads_kv * head_size),
            dtype=torch.float16,
            device=device,
        )
        padded[:q_len] = cur
        paged = padded.reshape(n_seq_blocks, _KV_BLOCK_SIZE, 2 * n_heads_kv * head_size)
        k_cache[first : first + n_seq_blocks] = paged[
            :, :, : n_heads_kv * head_size
        ].reshape(n_seq_blocks, _KV_BLOCK_SIZE, n_heads_kv, head_size)
        v_cache[first : first + n_seq_blocks] = paged[
            :, :, n_heads_kv * head_size :
        ].reshape(n_seq_blocks, _KV_BLOCK_SIZE, n_heads_kv, head_size)
        first += n_seq_blocks
    return q, k_cache, v_cache


@pytest.mark.blocked_flash
def test_validate_attention_atoms_refuses_upstreams_layout():
    """The validator must reject upstream's layout, and name the atom it rejected.

    This is what stands between a wrong atom layout and the kernel. Slot [0] is an
    index into ``kv_block_idx`` here and a 64-bit host pointer upstream, and the
    kernel uses it unguarded -- so upstream's atoms are an illegal memory access
    rather than a wrong number: the negative value a pointer's low half reads as
    sent the kernel to ``kv_block_idx[-1354760192]``, which takes the CUDA context
    down and fails every later test in the process for a reason of its own.

    Note what this test does *not* do: hand upstream's atoms to ``blocked_flash``.
    The operator does not check the layout -- a check on every call costs a device
    sync, measured at ~45 us on an A100, more than the kernel at the shapes it
    serves -- so that call would fault, and the fault is the thing being avoided.
    Validation is the caller's step, which is why it is a function here and not a
    guard in the operator.
    """
    device = flag_train.device
    seq = [(128, 0), (192, 0), (1, 0)]
    atoms, kv_block_idx = build_blocked_flash_atoms(
        seq, _Q_BLOCK_SIZE, _KV_BLOCK_SIZE, device
    )
    deepspeed_atoms, _ = _deepspeed_atoms(seq, _Q_BLOCK_SIZE, _KV_BLOCK_SIZE, device)

    with pytest.raises(ValueError, match="out of range"):
        validate_attention_atoms(deepspeed_atoms, kv_block_idx)

    # And it is the layout it objected to, not everything: this port's own atoms
    # pass the same check.
    validate_attention_atoms(atoms, kv_block_idx)


@pytest.mark.blocked_flash
def test_from_deepspeed_atoms_reproduces_this_ports_atoms():
    """The translation must land on exactly what this port builds for itself.

    ``_deepspeed_atoms`` is the same cache expressed upstream's way, so a
    conversion that is right has to return the *same* atoms and the same
    ``kv_block_idx`` -- not merely an equivalent pair. Comparing the two
    constructions is therefore a stronger check than running both, and it holds
    the deduplication honest too: upstream's atoms share one pointer per sequence,
    and collapsing them wrongly would show up here as a different block list.
    """
    device = flag_train.device
    seq = [(128, 0), (192, 0), (1, 0)]
    ours, our_blocks = build_blocked_flash_atoms(
        seq, _Q_BLOCK_SIZE, _KV_BLOCK_SIZE, device
    )
    deepspeed_atoms, _ = _deepspeed_atoms(seq, _Q_BLOCK_SIZE, _KV_BLOCK_SIZE, device)

    converted, converted_blocks = from_deepspeed_atoms(deepspeed_atoms)

    assert torch.equal(converted_blocks, our_blocks)
    assert torch.equal(converted, ours)
    # Slot [1] held the pointer's high half and means nothing in this layout.
    assert converted[:, int(_ATOM_PTR_HIGH)].eq(0).all()
    validate_attention_atoms(converted, converted_blocks)


@pytest.mark.blocked_flash
def test_converted_atoms_drive_the_operator():
    """End to end: upstream's atoms, translated, must produce the right answer."""
    device = flag_train.device
    seq = [(128, 0), (192, 0), (1, 0)]
    q, k_cache, v_cache = _page_one_case(seq)
    deepspeed_atoms, _ = _deepspeed_atoms(seq, _Q_BLOCK_SIZE, _KV_BLOCK_SIZE, device)
    ref_atoms, ref_blocks = build_blocked_flash_atoms(
        seq, _Q_BLOCK_SIZE, _KV_BLOCK_SIZE, device
    )

    total_q = sum(q_len for q_len, _ in seq)
    ref = torch.zeros((total_q, 16 * _HEAD_SIZE), dtype=torch.float16, device=device)
    blocked_flash_ref(ref, q, k_cache, v_cache, ref_atoms, ref_blocks, 1.0)

    converted, converted_blocks = from_deepspeed_atoms(deepspeed_atoms)
    out = torch.zeros_like(ref)
    blocked_flash(
        out,
        q,
        k_cache,
        v_cache,
        converted,
        converted_blocks,
        1.0,
        q_block_size=_Q_BLOCK_SIZE,
    )

    _assert_close(out, ref, torch.float16, *_TOLERANCES[torch.float16])


# ---------------------------------------------------------------------------
# AtomBuilder
#
# Upstream keeps this in its own file (``test_atom_builder.py``) because it is
# its own operator there. The port lives in ``blocked_flash.py``, so the test
# does too -- and the assertions below are upstream's, kept verbatim, so a
# divergence surfaces here rather than at some later call site.
# ---------------------------------------------------------------------------

_ATOM_Q_BLOCK_SIZE = 128
_ATOM_KV_BLOCK_SIZE = 128


@pytest.mark.blocked_flash
@pytest.mark.parametrize(
    "case, message",
    [
        ("cpu_tensors", "must be a device tensor"),
        ("head_size_over_limit", "<= 256"),
        ("head_size_not_multiple_of_8", "divisible by 8"),
        ("capability_too_old", "compute capability"),
    ],
)
def test_rejects_what_the_reference_launcher_rejects(case, message, monkeypatch):
    """The reference launcher's input checks, reproduced.

    ``blocked_flash.cpp`` rejects these before it assembles a single parameter,
    so a caller gets the same diagnosis the reference gives rather than a failure
    somewhere inside Triton. Each case is one of its ``TORCH_CHECK``s.

    The capability case needs a monkeypatch because this machine is the one the
    limit was written for -- see ``test_launch_is_not_blocked_by_a_silent_device``
    for the other half of that rule.
    """
    device = flag_train.device
    head_size, n_heads_q, n_heads_kv = 64, 16, 16
    if case == "head_size_over_limit":
        head_size = 264  # a multiple of 8, past the launcher's ceiling
    elif case == "head_size_not_multiple_of_8":
        head_size = 12  # under the ceiling, but not a multiple of 8

    n_tokens, kv_block_size, q_block_size = 128, 64, 128
    n_blocks = (n_tokens + kv_block_size - 1) // kv_block_size

    q = torch.randn(
        (n_tokens, n_heads_q * head_size), dtype=torch.float16, device=device
    )
    k = torch.randn(
        (n_blocks, kv_block_size, n_heads_kv, head_size),
        dtype=torch.float16,
        device=device,
    )
    v = torch.randn_like(k)
    out = torch.zeros_like(q)
    atoms, kv_block_idx = build_blocked_flash_atoms(
        [(n_tokens, 0)], q_block_size, kv_block_size, device
    )

    if case == "cpu_tensors":
        q, k, v, out = q.cpu(), k.cpu(), v.cpu(), out.cpu()
    elif case == "capability_too_old":
        # ``_check_launch_arguments`` reads the capability from the module it is
        # defined in, whichever backend dispatched to it, so patching that one
        # module covers the generic kernel and the Hygon one alike.
        blocked_flash_module = importlib.import_module(
            "flag_train.deepspeed.blocked_flash"
        )
        monkeypatch.setattr(blocked_flash_module, "_device_capability", lambda: (7, 5))

    with pytest.raises(ValueError, match=message):
        blocked_flash(out, q, k, v, atoms, kv_block_idx, 1.0, q_block_size=q_block_size)


@pytest.mark.blocked_flash
def test_launch_is_not_blocked_by_a_silent_device(monkeypatch):
    """A device that will not report a capability is left alone.

    The reference asserts Ampere-or-newer, which is how its CUDA-only launcher
    spells "tensor cores required". This port runs on several vendors, and one
    that cannot answer the question is not evidence that it is too old --
    inventing a limit it never stated is how a port rejects hardware that was
    working. So an unreadable capability must not raise.
    """
    device = flag_train.device
    blocked_flash_module = importlib.import_module("flag_train.deepspeed.blocked_flash")
    monkeypatch.setattr(blocked_flash_module, "_device_capability", lambda: None)

    atoms, kv_block_idx = build_blocked_flash_atoms([(128, 0)], 128, 64, device)
    q = torch.randn((128, 16 * 64), dtype=torch.float16, device=device)
    k = torch.randn((2, 64, 16, 64), dtype=torch.float16, device=device)
    v = torch.randn_like(k)
    out = torch.zeros_like(q)

    blocked_flash(out, q, k, v, atoms, kv_block_idx, 1.0, q_block_size=128)
    assert torch.isfinite(out).all()


@pytest.mark.blocked_flash
@pytest.mark.parametrize(
    "seq_params", [(1, 0, 0), (1, 228, 0), (383, 0, 0), (1, 494, 0)]
)
def test_atom_builder_matches_upstream(seq_params):
    """Upstream's ``test_atom_builder.py::test_single_sequence``, assertions intact.

    The third element of each upstream tuple is the KV block pointer, which the
    upstream test passes as 0. This port has no pointers and needs no stand-in,
    so it is unused rather than replaced.
    """
    seq_len, n_seen_tokens, _ = seq_params

    batch = RaggedBatchWrapper([(seq_len, n_seen_tokens)], _ATOM_KV_BLOCK_SIZE)
    atoms = torch.empty((8, 8), dtype=torch.int32, device=torch.device("cpu"))
    atoms, kv_block_idx, n_atoms = AtomBuilder()(
        atoms, batch, _ATOM_Q_BLOCK_SIZE, _ATOM_KV_BLOCK_SIZE
    )

    assert n_atoms == (seq_len + 127) // 128

    for i, atom in enumerate(atoms[:n_atoms]):
        # Upstream asserts 0 because its pointer was 0. Here it is an offset into
        # kv_block_idx, which is 0 for the first -- and in this case only --
        # sequence, so the same assertion holds for a different reason.
        assert atom[0] == 0
        assert atom[1] == 0

        assert atom[2] == i * _ATOM_Q_BLOCK_SIZE
        assert atom[3] == min(_ATOM_Q_BLOCK_SIZE, seq_len - i * _ATOM_Q_BLOCK_SIZE)

        total_toks = i * _ATOM_Q_BLOCK_SIZE + min(
            _ATOM_Q_BLOCK_SIZE, seq_len - i * _ATOM_Q_BLOCK_SIZE
        )
        assert (
            atom[4]
            == (total_toks + n_seen_tokens + _ATOM_KV_BLOCK_SIZE - 1)
            // _ATOM_KV_BLOCK_SIZE
        )
        assert atom[5] == total_toks + n_seen_tokens
        assert atom[6] == n_seen_tokens + i * _ATOM_Q_BLOCK_SIZE

    # The run of blocks the atoms address must be exactly as long as the last
    # atom needs -- history plus this sequence's own tokens.
    assert (
        kv_block_idx.numel()
        == (seq_len + n_seen_tokens + _ATOM_KV_BLOCK_SIZE - 1) // _ATOM_KV_BLOCK_SIZE
    )


@pytest.mark.blocked_flash
def test_q_block_size_bound_matches_the_synced_path():
    """Handing over the atom bound must not change the answer.

    ``blocked_flash`` sizes its query tile two ways: read the maximum atom
    ``q_len`` off the device -- a sync, and the default -- or be told the bound
    the atoms were built with. The second exists only to avoid the sync, so the
    two have to agree bit for bit. If they ever diverge the fast path is
    silently wrong, and the benchmark would be measuring the wrong kernel.

    Nothing else in this file covers the bound path: ``_run_case`` deliberately
    leaves it out so the default stays exercised.
    """
    device = flag_train.device
    torch.manual_seed(0)

    seq_params = [(300, 0), (1, 128), (128, 256)]
    q_block_size, kv_block_size = 128, 64
    atoms, kv_block_idx = build_blocked_flash_atoms(
        seq_params, q_block_size, kv_block_size, device
    )

    total_q = sum(q_len for q_len, _ in seq_params)
    head_size, n_heads_q, n_heads_kv = 64, 16, 16
    n_blocks = int(kv_block_idx.numel())

    q = torch.randn(
        (total_q, n_heads_q * head_size), dtype=torch.float16, device=device
    )
    k = torch.randn(
        (n_blocks, kv_block_size, n_heads_kv, head_size),
        dtype=torch.float16,
        device=device,
    )
    v = torch.randn_like(k)

    synced = torch.zeros_like(q)
    blocked_flash(synced, q, k, v, atoms, kv_block_idx, 1.0)

    bounded = torch.zeros_like(q)
    blocked_flash(bounded, q, k, v, atoms, kv_block_idx, 1.0, q_block_size=q_block_size)

    assert torch.equal(synced, bounded)


@pytest.mark.blocked_flash
def test_atom_builder_multiple_sequences():
    """Where upstream's own builder test is thin: several sequences, one batch.

    Each sequence addresses its own run of KV blocks, so slot [0] stops being 0
    and becomes that sequence's offset -- the deviation made visible, and the
    property the multi-sequence kernel cases depend on.
    """
    seq_params = [(300, 0), (1, 128), (128, 256)]

    batch = RaggedBatchWrapper(seq_params, _ATOM_KV_BLOCK_SIZE)
    atoms = torch.empty((16, 8), dtype=torch.int32, device=torch.device("cpu"))
    atoms, kv_block_idx, n_atoms = AtomBuilder()(
        atoms, batch, _ATOM_Q_BLOCK_SIZE, _ATOM_KV_BLOCK_SIZE
    )

    # One run per sequence, laid down in order, each as long as that sequence's
    # history plus its own tokens.
    expected_offsets = []
    total_blocks = 0
    for q_len, history_len in seq_params:
        expected_offsets.append(total_blocks)
        total_blocks += (history_len + q_len + _ATOM_KV_BLOCK_SIZE - 1) // (
            _ATOM_KV_BLOCK_SIZE
        )
    assert kv_block_idx.numel() == total_blocks

    # Atom rows stay grouped by sequence and in sequence order.
    spans = [0]
    for q_len, _ in seq_params:
        spans.append(spans[-1] + (q_len + _ATOM_Q_BLOCK_SIZE - 1) // _ATOM_Q_BLOCK_SIZE)
    assert n_atoms == spans[-1]

    for seq_i in range(len(seq_params)):
        for row in range(spans[seq_i], spans[seq_i + 1]):
            assert atoms[row, 0] == expected_offsets[seq_i]


# ---------------------------------------------------------------------------
# Attention atoms: the batch metadata, and the builder that fills them
#
# !! KEEP IN SYNC with benchmark/deepspeed/test_blocked_flash.py !!
#
# Port of DeepSpeed's ``AtomBuilder`` (``ragged_ops/atom_builder``). Upstream it
# is a *separate* operator -- its own directory, its own ``DSKernelBase``
# subclass, its own ``test_atom_builder.py`` -- and it is kept out of the
# operator module here for the same reason the reference above is: nothing but
# the tests calls it. ``blocked_flash`` takes atoms as an argument and needs no
# producer; the layer that would supply one (``DSDenseBlockedAttention``) is not
# ported yet, so shipping one would be exporting a name no caller exercises.
#
# The header above carries the sync note; the body below must stay identical.
#
# Both sides are host code: ``atom_builder.cpp`` contains no ``__global__``, and
# upstream reads the batch metadata by reinterpreting host tensors as structs.
# The same three tensors, in the same layouts, are read here. The atom-field
# constants are the operator module's, imported so the layout has one definition.
# ---------------------------------------------------------------------------

# InflightSeqDescriptor field offsets, in int32 slots. Mirrors ragged_dtypes.h:
# start_idx, n_tokens, seen_tokens, then an explicit padding slot DeepSpeed keeps
# so the struct matches its Python code pattern.
_SEQ_START_IDX = 0
_SEQ_N_TOKENS = 1
_SEQ_SEEN_TOKENS = 2
_SEQ_STRIDE = 4

# RaggedBatchDescriptor field offsets: n_tokens, n_sequences.
_BATCH_N_SEQUENCES = 1


class RaggedBatchWrapper:
    """The slice of DeepSpeed's ``RaggedBatchWrapper`` that ``AtomBuilder`` reads.

    Upstream that class is far larger -- it also owns the token stream, the
    token-to-sequence map and masks. ``AtomBuilder`` consumes only three of its
    accessors, and this port provides exactly those, backed by the same tensor
    layouts, so the builder reads them the way the C++ does rather than reaching
    into Python attributes.

    Constructed from ``seq_params``, a list of ``(q_len, history_len)`` pairs --
    the same shorthand the tests and the benchmark already use, and the same
    meaning as ``build_complex_batch``'s ``(seq_len, n_seen_tokens)``.

    Deviation: upstream's ``kv_ptrs`` accessor is replaced by ``kv_block_idx``.
    Upstream hands the kernel an array of per-sequence *host pointers* into the
    block-id lists; a Triton kernel cannot dereference those, so the lists are
    concatenated into one int32 tensor and each sequence's start is an offset.
    """

    def __init__(self, seq_params, kv_block_size):
        self._seq_params = list(seq_params)
        self._kv_block_size = kv_block_size
        self.current_sequences = len(self._seq_params)
        self.current_tokens = sum(q_len for q_len, _ in self._seq_params)

        batch_metadata = torch.zeros(
            (2,), dtype=torch.int32, device=torch.device("cpu")
        )
        batch_metadata[int(_BATCH_N_SEQUENCES)] = self.current_sequences
        self._batch_metadata_shadow = batch_metadata

        descriptors = torch.zeros(
            (max(self.current_sequences, 1), int(_SEQ_STRIDE)),
            dtype=torch.int32,
            device=torch.device("cpu"),
        )
        block_idx = []
        block_offsets = []
        start_idx = 0
        for i, (q_len, history_len) in enumerate(self._seq_params):
            descriptors[i, int(_SEQ_START_IDX)] = start_idx
            descriptors[i, int(_SEQ_N_TOKENS)] = q_len
            descriptors[i, int(_SEQ_SEEN_TOKENS)] = history_len

            # Contiguous physical blocks per sequence, which is what the cache
            # manager hands out in the un-fragmented case. ``AtomBuilder``
            # records only where each sequence's run starts, so a real allocator
            # can substitute any block list here without touching the builder.
            n_blocks = (history_len + q_len + kv_block_size - 1) // kv_block_size
            block_offsets.append(len(block_idx))
            block_idx.extend(range(len(block_idx), len(block_idx) + n_blocks))

            start_idx += q_len

        self._inflight_seq_descriptors_shadow = descriptors
        self._kv_block_offsets = block_offsets
        self._kv_block_idx = torch.tensor(block_idx, dtype=torch.int32)

    def batch_metadata_buffer(self, on_device=False):
        """``RaggedBatchDescriptor``: ``[n_tokens, n_sequences]`` int32.

        Host-only here; upstream also keeps a device copy for its kernels, which
        ``AtomBuilder`` never reads.
        """
        return self._batch_metadata_shadow

    def inflight_seq_descriptors(self, on_device=False):
        """``InflightSeqDescriptor`` per sequence, ``[n_seq, 4]`` int32."""
        return self._inflight_seq_descriptors_shadow[: self.current_sequences]

    def kv_block_offsets(self):
        """Each sequence's start offset into :meth:`kv_block_idx`."""
        return self._kv_block_offsets

    def kv_block_idx(self, device=None):
        """All sequences' physical block ids, concatenated. Replaces ``kv_ptrs``."""
        if device is None:
            return self._kv_block_idx
        return self._kv_block_idx.to(device=device)


class AtomBuilder:
    """Port of DeepSpeed's ``AtomBuilder`` (``ragged_ops/atom_builder``).

    Walks the batch's sequences, splits each into ``ceil(q_len / q_block_size)``
    atoms, and fills in each atom's query range and KV extent. Mirrors
    ``build_atoms`` in ``atom_builder.cpp`` field for field.
    """

    def __call__(self, atoms, ragged_batch, q_block_size, kv_block_size):
        """Populate ``atoms`` in place from ``ragged_batch``.

        Args:
            atoms (Tensor): pre-allocated ``[max_atoms, 8]`` int32 tensor **on the
                CPU**. Upstream requires host memory because its slot [0] is a host
                pointer; this port keeps the requirement so the calling convention
                is unchanged, and so the caller's copy-to-device stays explicit.
            ragged_batch (RaggedBatchWrapper): the batch to describe.
            q_block_size (int): query rows per atom.
            kv_block_size (int): tokens per KV block.

        Returns:
            tuple: ``(atoms, kv_block_idx, n_atoms)``. The middle element is the
            deviation -- see the class docstring.
        """
        if atoms.device != torch.device("cpu"):
            raise RuntimeError("AtomBuilder must be called on tensors")

        batch_meta = ragged_batch.batch_metadata_buffer(on_device=False)
        descriptors = ragged_batch.inflight_seq_descriptors(on_device=False)
        block_offsets = ragged_batch.kv_block_offsets()

        # Host-side ints: the atom slot constants are ``tl.constexpr`` for the
        # kernel's benefit and torch will not index a tensor with one.
        (
            slot_block_offset,
            slot_q_start,
            slot_q_len,
            slot_kv_blocks,
            slot_total_extent,
            slot_global_q_idx,
            slot_ptr_high,
            slot_unused,
        ) = (
            int(_ATOM_BLOCK_OFFSET),
            int(_ATOM_Q_START),
            int(_ATOM_Q_LEN),
            int(_ATOM_KV_BLOCKS),
            int(_ATOM_TOTAL_EXTENT),
            int(_ATOM_GLOBAL_Q_IDX),
            int(_ATOM_PTR_HIGH),
            int(_ATOM_UNUSED),
        )

        n_atoms = 0
        for i in range(int(batch_meta[int(_BATCH_N_SEQUENCES)])):
            n_tokens = int(descriptors[i, int(_SEQ_N_TOKENS)])
            cur_start_idx = int(descriptors[i, int(_SEQ_START_IDX)])
            global_start_idx = int(descriptors[i, int(_SEQ_SEEN_TOKENS)])
            remaining_toks = n_tokens

            for _ in range((n_tokens + q_block_size - 1) // q_block_size):
                atom_q_len = min(remaining_toks, q_block_size)
                end_toks = global_start_idx + atom_q_len

                atoms[n_atoms, slot_block_offset] = block_offsets[i]
                # Upstream packs a 64-bit pointer across slots 0-1; this port's
                # slot [0] is a 32-bit offset, so the high half is an explicit 0
                # rather than a hole.
                atoms[n_atoms, slot_ptr_high] = 0
                atoms[n_atoms, slot_q_start] = cur_start_idx
                atoms[n_atoms, slot_q_len] = atom_q_len
                # DeepSpeed's own comment on this line: the extent assumes a dense
                # cache, and would have to change for a sparse one.
                atoms[n_atoms, slot_kv_blocks] = (
                    end_toks + kv_block_size - 1
                ) // kv_block_size
                atoms[n_atoms, slot_total_extent] = end_toks
                atoms[n_atoms, slot_global_q_idx] = global_start_idx
                # Never read by the kernel; written so the atom is a function of
                # its inputs alone, not of the buffer it was built in.
                atoms[n_atoms, slot_unused] = 0

                cur_start_idx += atom_q_len
                global_start_idx += atom_q_len
                remaining_toks -= atom_q_len
                n_atoms += 1

        return atoms, ragged_batch.kv_block_idx(), n_atoms


# ---------------------------------------------------------------------------
# Scenarios the shape-parameterised cases above do not reach
#
# Those vary one axis at a time -- how long, how many heads, which dtype -- and
# always through the same machinery: contiguous cache blocks, a contiguous
# ``out``, a unit softmax scale, a fresh output buffer, atoms in builder order.
# What follows varies the *state* instead of the size.
# ---------------------------------------------------------------------------


def _operator_inputs(
    seq_params,
    head_size=_HEAD_SIZE,
    n_heads_q=16,
    n_heads_kv=16,
    kv_block_size=_KV_BLOCK_SIZE,
    q_block_size=_Q_BLOCK_SIZE,
    dtype=torch.float16,
):
    """A runnable argument set for the operator, on random cache contents.

    For the property checks below, which are about the *call* rather than the
    arithmetic -- so the values only have to be something the kernel can chew on.
    """
    device = flag_train.device
    torch.manual_seed(0)
    n_tokens = sum(q_len for q_len, _ in seq_params)
    n_blocks = sum(
        (history + q_len + kv_block_size - 1) // kv_block_size
        for q_len, history in seq_params
    )
    q = torch.randn((n_tokens, n_heads_q * head_size), dtype=dtype, device=device)
    k = torch.randn(
        (n_blocks, kv_block_size, n_heads_kv, head_size), dtype=dtype, device=device
    )
    v = torch.randn_like(k)
    atoms, kv_block_idx = build_blocked_flash_atoms(
        seq_params, q_block_size, kv_block_size, device
    )
    out = torch.zeros((n_tokens, n_heads_q * head_size), dtype=dtype, device=device)
    return out, q, k, v, atoms, kv_block_idx, q_block_size


@pytest.mark.blocked_flash
def test_kv_blocks_need_not_be_contiguous():
    """The cache's physical blocks may be in any order.

    This is the indirection the whole operator exists for: a paged cache hands
    out whatever blocks are free, so an atom's blocks are scattered and the only
    thing tying them together is ``kv_block_idx``. Every other case reaches the
    kernel through ``build_blocked_flash_atoms``, which numbers blocks in order,
    so without this one the indirection is only ever exercised where it happens
    to be the identity.
    """
    seq_params = [(300, 0), (1, 128), (128, 256)]
    _run_case(seq_params, permute_blocks=True)


@pytest.mark.blocked_flash
def test_out_may_be_a_view_with_a_wider_row_stride():
    """``out`` is strided to, not assumed packed.

    Upstream's own caller hands the kernel ``q_k_v[:, :n_heads_q * head_size]``
    -- a view whose rows are further apart than the payload. ``q`` here is
    already such a view; ``out`` was not, so the output stride was never
    exercised at anything but its trivial value.
    """
    _run_case([(300, 0), (1, 128)], out_pad_columns=256)


@pytest.mark.blocked_flash
@pytest.mark.parametrize("scale_name, scale", [("unit", 1.0), ("1/sqrt(d)", 0.125)])
def test_softmax_scale_is_applied(scale_name, scale):
    """A scale other than 1.0 reaches the scores.

    The default is 1.0 everywhere else in this file, and a run at 1.0 cannot
    tell "the scale was applied" from "the scale was ignored" -- every other
    case would pass either way.
    """
    _run_case([(128, 128), (192, 38)], softmax_scale=scale)


@pytest.mark.blocked_flash
def test_mqa_single_kv_head():
    """One KV head shared by every query head -- the far end of GQA.

    The ratio here is 16 instead of the 1, 2 and 4 the other cases use, so the
    ``hq // ratio`` mapping is exercised where it collapses to a single head.
    """
    _run_case([(128, 128), (192, 38), (1, 814)], n_heads_q=16, n_heads_kv=1)


@pytest.mark.blocked_flash
def test_many_uneven_sequences():
    """Sixteen sequences of wildly different lengths in one batch.

    ``global_q_idx`` and the block offsets both accumulate across sequences, so
    a batch this uneven is where an off-by-one in either would show: a short
    sequence landing mid-block of a long one's run, or its position restarting.
    """
    lengths = [1, 2, 63, 64, 65, 127, 128, 129, 200, 257, 300, 511, 512, 700, 1000, 3]
    _run_case([(length, 0) for length in lengths])


@pytest.mark.blocked_flash
def test_output_is_overwritten_not_accumulated():
    """A second call assigns ``out``; it does not add to it.

    The contract lets a caller reuse one buffer across steps, and every other
    case passes a fresh zero tensor -- the one thing that hides ``out +=`` .
    """
    args = _operator_inputs([(300, 0), (1, 128)])
    out, q, k, v, atoms, kv_block_idx, q_block_size = args

    blocked_flash(out, q, k, v, atoms, kv_block_idx, 1.0, q_block_size=q_block_size)
    first = out.clone()
    assert first.abs().sum() > 0, "the first call wrote nothing"

    blocked_flash(out, q, k, v, atoms, kv_block_idx, 1.0, q_block_size=q_block_size)
    assert torch.equal(out, first)


@pytest.mark.blocked_flash
def test_atom_order_does_not_matter():
    """Each atom is an independent program; renumbering them changes nothing.

    ``program_id(0)`` selects an atom and no atom reads another's result, so a
    permuted atom tensor must produce the same output. It also pins the other
    direction: the block offsets are self-contained, so permuting the rows
    cannot leave an atom pointing at a neighbour's block list.
    """
    out, q, k, v, atoms, kv_block_idx, q_block_size = _operator_inputs(
        [(300, 0), (1, 128)]
    )
    blocked_flash(out, q, k, v, atoms, kv_block_idx, 1.0, q_block_size=q_block_size)
    reference = out.clone()

    order = torch.randperm(atoms.size(0), device=atoms.device)
    shuffled = torch.zeros_like(out)
    blocked_flash(
        shuffled, q, k, v, atoms[order], kv_block_idx, 1.0, q_block_size=q_block_size
    )
    assert torch.equal(shuffled, reference)


@pytest.mark.blocked_flash
def test_large_logits_stay_finite():
    """Big scores saturate the softmax instead of overflowing it.

    The running max is subtracted before the exponential, so the kernel should
    stay finite where a naive ``exp(scores)`` would not. Scaling ``q`` and ``k``
    up is the cheapest way to push the logits past what fp16 could hold
    unnormalized.
    """
    out, q, k, v, atoms, kv_block_idx, q_block_size = _operator_inputs([(256, 0)])
    q = q * 30.0
    k = k * 30.0

    blocked_flash(out, q, k, v, atoms, kv_block_idx, 1.0, q_block_size=q_block_size)
    assert torch.isfinite(out).all(), "logits overflowed the online softmax"

    # Saturated attention is a convex combination of ``v``, so the output cannot
    # leave ``v``'s range -- a cheap independent bound on a case no oracle here
    # covers at this magnitude.
    assert out.abs().max() <= v.abs().max() + 1e-3
