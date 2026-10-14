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
"""Performance benchmark for the blocked (paged KV-cache) flash attention forward.

The baseline is picked by what the platform can run -- the same three tiers, in
the same order and on the same predicates, that
``tests/deepspeed/test_blocked_flash.py`` holds the operator to. A benchmark
measuring against something the tests do not accept would be answering a
different question, so the two files state one rule:

* on a backend that runs DeepSpeed's own blocked-flash kernel, that kernel is the
  baseline. It is the kernel this port is a port *of*, which makes it the only
  baseline that answers "what did the port cost", and it is *required*: if it
  cannot be loaded the module raises rather than timing something else and
  reporting it under that name;
* on any other backend with ``flash_attn`` installed, ``flash_attn_varlen_func``
  over the densely gathered KV -- a real competitor, and the honest question is
  how the kernel compares to it;
* failing both, ``blocked_flash_ref``, the plain-torch composition over the same
  paged cache. It is not a competitor, and the speedup there says how far the
  kernel is from *a* correct implementation rather than from the best one.

Every baseline takes the same argument tuple, so the dense KV the second one
needs is gathered in ``input_fn`` -- outside the timed region. Gathering it
inside the baseline would charge flash-attn for a copy the blocked kernel exists
to avoid.
"""

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
    _ATOM_TOTAL_EXTENT,
    _ATOM_UNUSED,
)

from .. import base

# ---------------------------------------------------------------------------
# Reference implementation
#
# The baseline when `flash_attn` is absent: a plain-torch composition of the same
# contract, reading the *same* atoms and the *same* paged cache as the kernel.
#
# !! KEEP IN SYNC with tests/deepspeed/test_blocked_flash.py !!
#
# This is a second, byte-identical copy of the reference that file uses as its
# primary oracle. The duplication is deliberate -- a reference belongs to the
# tests that check against it, not to the operator package, so that a checked
# implementation is never code the operator ships. The cost is that the two
# copies can drift: change one and you must change the other, or this benchmark
# will measure against a different contract than the tests accept.
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
# !! KEEP IN SYNC with tests/deepspeed/test_blocked_flash.py !!
#
# That file carries a second copy of this function. Like the reference above it
# is scaffolding rather than operator API, so it lives with the tests that use it
# rather than in the package -- ``flag_train.deepspeed``'s surface stays to the
# names the reference implementation exposes. The cost is the usual one: change
# this copy and you must change that one, or the tests will build their atoms a
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


# The baseline, by platform. The same three-branch rule, in the same order and on
# the same predicates, that tests/deepspeed/test_blocked_flash.py picks the oracle
# it holds the operator to -- both files name the tier ``_ORACLE``, so a reader
# comparing the two sees one rule rather than two that happen to resemble each
# other. This file adds what a benchmark needs and the tests do not: the tier
# decides what is *timed*, so the branch is exclusive here, where the tests run
# every oracle they can.
#
# * on a backend that has DeepSpeed's own blocked-flash kernel, that kernel **is**
#   the baseline -- it is the operator being ported, and measuring anything else
#   answers a different question. If it cannot be loaded there the module raises
#   rather than falling back;
# * elsewhere `flash_attn_varlen_func` over the densely gathered KV, where the
#   package is available -- a real competitor;
# * and failing that the plain-torch composition over the same paged cache, which
#   is not a competitor: the speedup says how far the kernel is from *a* correct
#   implementation, not from the best one.
_DEEPSPEED_BASELINE_VENDORS = {"nvidia"}


def _load_deepspeed_blocked_flash():
    """``(BlockedFlashAttn, DtypeEnum)``, or ``None`` on a backend that does not use it.

    The reference kernel is not compiled from source here: ``RaggedOpsBuilder``
    links ``-lblockedflash`` against a prebuilt library that ships in the
    ``dskernels`` package (``op_builder/ragged_ops.py``). So this needs that
    package installed for the platform; a checkout without it cannot run the
    baseline at all, and says so rather than measuring something else.
    """
    if flag_train.vendor_name not in _DEEPSPEED_BASELINE_VENDORS:
        return None

    try:
        from deepspeed.inference.v2.inference_utils import DtypeEnum
        from deepspeed.inference.v2.kernels.ragged_ops import BlockedFlashAttn

        return BlockedFlashAttn, DtypeEnum
    except Exception as exc:
        raise RuntimeError(
            f"{flag_train.vendor_name!r} must use DeepSpeed's BlockedFlashAttn as "
            f"its baseline, but it could not be loaded: {exc!r}. Its kernel ships "
            f"prebuilt in the `dskernels` package (`op_builder/ragged_ops.py` links "
            f"`-lblockedflash`), so install that for this platform, or drop the "
            f"backend from _DEEPSPEED_BASELINE_VENDORS."
        ) from exc


# Resolved once, like the lamb benchmark's baseline.
_deepspeed_blocked_flash = _load_deepspeed_blocked_flash()

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func

    _HAS_FLASH_ATTN = True
except ImportError:
    _HAS_FLASH_ATTN = False

if _deepspeed_blocked_flash is not None:
    _ORACLE = "deepspeed"
    _BASELINE = "deepspeed BlockedFlashAttn"
elif _HAS_FLASH_ATTN:
    _ORACLE = "flash_attn"
    _BASELINE = "flash_attn_varlen_func (dense KV)"
else:
    _ORACLE = "torch_ref"
    _BASELINE = "blocked_flash_ref (torch, paged KV)"

# (context_length, head_size, n_heads_q, n_heads_kv). Blocked attention needs
# head_size <= 256 and a compute capability of 8.0 or better; the cache's block
# size is independent. The first eight rows keep the 16/16 head layout the
# original benchmark measured; the rest add grouped-query attention, where
# several query heads share one KV head. A 4x ratio is Llama-3-8B's and 8x
# Llama-3-70B's, and GQA is the axis a paged-cache kernel is most likely to
# behave differently on -- a shared KV head is read once per query head, so the
# cache traffic a blocked kernel exists to reduce scales with the ratio.
_BLOCKED_FLASH_SHAPES = [
    (128, 64, 16, 16),
    (512, 64, 16, 16),
    (1024, 64, 16, 16),
    (2048, 64, 16, 16),
    (4096, 64, 16, 16),
    (128, 128, 16, 16),
    (1024, 128, 16, 16),
    (4096, 128, 16, 16),
    (1024, 128, 32, 8),
    (4096, 128, 32, 8),
    (4096, 128, 64, 8),
]

# The geometry for the tiers that do not constrain it. On the DeepSpeed tier the
# KV block size is fixed by the baseline's kernel, so `_cache_geometry` returns
# that instead of `_KV_BLOCK_SIZE` -- see there for why it is not negotiable.
_Q_BLOCK_SIZE = 128
_KV_BLOCK_SIZE = 64
_SOFTMAX_SCALE = 1.0


class BlockedFlashBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _BLOCKED_FLASH_SHAPES
    DEFAULT_SHAPE_DESC = "context_length, head_size, n_heads_q, n_heads_kv"

    def set_shapes(self, shape_file=None):
        self.shapes = list(_BLOCKED_FLASH_SHAPES)

    def set_more_shapes(self):
        return []


def _deepspeed_atoms(seq_params, q_block_size, kv_block_size, device):
    """Atoms in the reference layout, for the DeepSpeed baseline.

    Its kernel dereferences slot [0] as a *host pointer* to the block list, over
    UVA, so that list has to live in pinned memory and every atom has to carry
    the address of its own sequence's run. Our builder writes an offset there
    instead -- a Triton kernel cannot dereference a host pointer
    (blocked_flash.md §5) -- so this turns the offset back into the pointer the
    baseline wants.

    Returns:
        tuple: ``(atoms, pinned)``. ``pinned`` is handed back because the atoms
        point into it: letting it go would leave the kernel reading freed memory.
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
    # A pointer is two int32 slots, low half first -- slot [0] then [1].
    atoms[:, 0:2] = addresses.view(torch.int32).reshape(-1, 2)
    return atoms, pinned


# One kernel per (head_size, dtype), built on first use: the constructor is what
# triggers ``RaggedOpsBuilder().load()``, and that must not land in the timed
# region.
_deepspeed_kernels = {}


def _deepspeed_kernel(head_size, dtype):
    key = (head_size, dtype)
    if key not in _deepspeed_kernels:
        BlockedFlashAttn, DtypeEnum = _deepspeed_blocked_flash
        _deepspeed_kernels[key] = BlockedFlashAttn(head_size, DtypeEnum(dtype))
    return _deepspeed_kernels[key]


def _deepspeed_geometry(head_size):
    """``(q_block_size, kv_block_size)`` DeepSpeed's kernel is *built* for.

    Neither is a parameter of its launcher. ``blocked_flash.cpp`` takes the row
    pitch from ``k.stride(1)``, but the number of tokens in a cache block never
    comes off the tensor at all: it is a template argument of the kernel, fixed
    upstream by these two helpers -- which is why upstream's own test calls them
    rather than choosing either value.

    Handed a cache built at another block size, the kernel walks it at the wrong
    pitch and reads the wrong tokens. Measured on an A100 at head_size 64, a
    64-token block moves this baseline's own output by 5.7 where the operator
    under test differs from the reference by 2e-3 -- so the two are not computing
    the same thing, and a ratio between them would be a ratio between two
    different questions.
    """
    from deepspeed.inference.v2.kernels.ragged_ops.blocked_flash.blocked_flash import (
        get_kv_block_size,
        get_q_block_size,
    )

    return get_q_block_size(head_size), get_kv_block_size(head_size)


def _cache_geometry(head_size):
    """``(q_block_size, kv_block_size)`` to build this case's cache and atoms with.

    On the DeepSpeed tier the geometry is not free, so that tier picks it; on the
    others nothing constrains it and the module defaults stand. The shape list
    mixes head sizes, and the two tiers want different block sizes for the same
    head size -- which is exactly why this is per-shape rather than a constant.
    """
    if _deepspeed_blocked_flash is not None:
        return _deepspeed_geometry(head_size)
    return _Q_BLOCK_SIZE, _KV_BLOCK_SIZE


def blocked_flash_input_fn(shape, dtype, device):
    """One prompt of ``context_length`` tokens against its own paged KV-cache.

    Also yields the same KV gathered densely, for the flash_attn baseline. The
    blocked operator ignores those.
    """
    n_tokens, head_size, n_heads_q, n_heads_kv = shape
    q_block_size, kv_block_size = _cache_geometry(head_size)

    q = torch.randn((n_tokens, n_heads_q * head_size), dtype=dtype, device=device)
    kv = torch.randn((n_tokens, 2 * n_heads_kv * head_size), dtype=dtype, device=device)
    out = torch.empty_like(q)

    # A single sequence, so the cache holds exactly the prompt's own KV.
    n_blocks = (n_tokens + kv_block_size - 1) // kv_block_size
    padded = torch.zeros(
        (n_blocks * kv_block_size, 2 * n_heads_kv * head_size),
        dtype=dtype,
        device=device,
    )
    padded[:n_tokens] = kv
    paged = padded.reshape(n_blocks, kv_block_size, 2 * n_heads_kv * head_size)
    k_cache = (
        paged[:, :, : n_heads_kv * head_size]
        .reshape(n_blocks, kv_block_size, n_heads_kv, head_size)
        .contiguous()
    )
    v_cache = (
        paged[:, :, n_heads_kv * head_size :]
        .reshape(n_blocks, kv_block_size, n_heads_kv, head_size)
        .contiguous()
    )

    atoms, kv_block_idx = build_blocked_flash_atoms(
        [(n_tokens, 0)], q_block_size, kv_block_size, device
    )

    # The DeepSpeed baseline reads atoms carrying host pointers instead; built
    # here, outside the timed region, like the dense KV above.
    if _deepspeed_blocked_flash is not None:
        deepspeed_atoms, deepspeed_pinned = _deepspeed_atoms(
            [(n_tokens, 0)], q_block_size, kv_block_size, device
        )
    else:
        deepspeed_atoms, deepspeed_pinned = None, None

    cu_seqlens = torch.tensor([0, n_tokens], dtype=torch.int32, device=device)
    yield (
        out,
        q,
        k_cache,
        v_cache,
        atoms,
        kv_block_idx,
        kv[:, : n_heads_kv * head_size].contiguous(),
        kv[:, n_heads_kv * head_size :].contiguous(),
        cu_seqlens,
        cu_seqlens,
        n_tokens,
        n_tokens,
        deepspeed_atoms,
        deepspeed_pinned,
        q_block_size,
    )


def torch_op(
    out,
    q,
    k,
    v,
    atoms,
    kv_block_idx,
    dense_k,
    dense_v,
    cu_q,
    cu_kv,
    max_q,
    max_kv,
    deepspeed_atoms,
    deepspeed_pinned,
    q_block_size,
):
    """Baseline, chosen by platform. See the module docstring.

    ``q_block_size`` is the operator under test's to use; a baseline works its own
    geometry out, and the two tiers want different ones for the same head size.
    It is taken and ignored rather than left out so that both ops are handed the
    same tuple.
    """
    if _deepspeed_blocked_flash is not None:
        # Same paged tensors as the operator under test -- this baseline needs no
        # dense gather, it reads the cache the way ours does -- plus the atoms
        # that carry host pointers.
        return _deepspeed_kernel(k.size(-1), out.dtype)(
            out, q, k, v, deepspeed_atoms, _SOFTMAX_SCALE
        )
    if _HAS_FLASH_ATTN:
        head_size = k.size(-1)
        # Read off the tensors rather than module constants, so the GQA rows
        # reshape to their own head counts.
        n_heads_q = q.size(-1) // head_size
        n_heads_kv = k.size(-2)
        return flash_attn_varlen_func(
            q.reshape(-1, n_heads_q, head_size),
            dense_k.reshape(-1, n_heads_kv, head_size),
            dense_v.reshape(-1, n_heads_kv, head_size),
            cu_q,
            cu_kv,
            max_q,
            max_kv,
            softmax_scale=_SOFTMAX_SCALE,
            causal=True,
        ).reshape(q.shape)
    return blocked_flash_ref(out, q, k, v, atoms, kv_block_idx, _SOFTMAX_SCALE)


def train_op(
    out,
    q,
    k,
    v,
    atoms,
    kv_block_idx,
    dense_k,
    dense_v,
    cu_q,
    cu_kv,
    max_q,
    max_kv,
    deepspeed_atoms,
    deepspeed_pinned,
    q_block_size,
):
    """The operator under test; the dense KV is the baseline's concern, not ours.

    ``q_block_size`` comes from ``input_fn``, which built these atoms with it:
    knowing the bound lets the operator skip a per-call device sync (~50 us on an
    A100) that would otherwise be charged against the kernel, which at the small
    shapes here is several times the kernel's own time. It is read from the tuple
    rather than taken from the module constant because on the DeepSpeed tier the
    geometry is the baseline's to choose, and the atoms agree with whichever one
    ``input_fn`` used.
    """
    return blocked_flash(
        out, q, k, v, atoms, kv_block_idx, _SOFTMAX_SCALE, q_block_size=q_block_size
    )


@pytest.mark.blocked_flash
def test_blocked_flash_perf():
    # The tier first, so a run whose numbers moved says whether the baseline it
    # moved against is the one the platform was supposed to pick.
    print(f"\nOracle: {_ORACLE}\nBaseline: {_BASELINE}")

    bench = BlockedFlashBenchmark(
        input_fn=blocked_flash_input_fn,
        op_name="blocked_flash",
        torch_op=torch_op,
        # The blocked kernel supports fp16 and bf16; both are measured, since the
        # kernel's inner loop is dtype-parametric and the two can rank
        # differently against a baseline that has its own dtype-specific paths.
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.set_train(train_op)
    bench.run()


# ---------------------------------------------------------------------------
# Attention atoms: the batch metadata, and the builder that fills them
#
# !! KEEP IN SYNC with tests/deepspeed/test_blocked_flash.py !!
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
