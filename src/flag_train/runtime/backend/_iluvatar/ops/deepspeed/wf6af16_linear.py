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
"""Iluvatar wf6af16_linear: the generic kernel with a verified config space.

The generic implementation is ``flag_train.deepspeed.wf6af16_linear``. Two of
its traits miscompile on Iluvatar's CoreX stack:

* the WIDE_LOAD path (uint32 pointer casts indexed by the K-loop counter)
  returns corrupted unpacked weights whenever the K loop iterates more than
  once, so it is forced off here;
* three of the twelve generic autotune configs -- (16, 16, 64) with 2 warps,
  and the two BLOCK_K=32 tiles (128, 128, 32) / (128, 256, 32) -- produce
  entirely wrong results on otherwise-passing shapes (per-config sweep with
  autotune bypassed; autotune itself cannot be trusted to avoid them because
  it scores speed, not correctness). This module wraps the bare generic GEMM
  kernel with the nine configs that measured correct instead.

Everything else -- the kernel body, the split-K heuristic, the reduction
kernel -- is the generic code's.

Registration is by name. This module's ``__init__`` re-exports
``wf6af16_linear``, and ``SpecOpRegistrar`` writes it over the same-named
generic function in ``flag_train.deepspeed`` when that package is imported --
the namespace the tests and the benchmark read the operator from.
"""

import logging

import torch
import triton

import flag_train
from flag_train.deepspeed.wf6af16_linear import (
    _pick_split_k,
    _wf6af16_linear_gemm,
    wf6af16_split_k_reduce_kernel,
)
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry

logger = logging.getLogger(__name__)


def _gemm_configs():
    """The generic autotune space minus the configs CoreX miscompiles.

    See the module docstring: (16, 16, 64)x2w and the two BLOCK_K=32 tiles
    are dropped. The (64, 16) tiles serve the decode shapes, the (32, 128)
    and larger tiles prefill.
    """
    configs = []
    for block_m, block_n, block_k, warps, stages in [
        (64, 16, 64, 4, 2),
        (64, 16, 64, 4, 4),
        (64, 32, 64, 4, 4),
        (64, 64, 64, 4, 4),
        (128, 64, 64, 4, 3),
        (32, 128, 64, 4, 3),
        (32, 128, 64, 8, 3),
        (64, 128, 64, 4, 4),
        (128, 128, 64, 8, 3),
    ]:
        configs.append(
            triton.Config(
                {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k},
                num_warps=warps,
                num_stages=stages,
            )
        )
    return configs


wf6af16_linear_kernel = libentry()(
    triton.autotune(
        configs=_gemm_configs(),
        key=["M", "N", "K", "split_k"],
    )(_wf6af16_linear_gemm)
)


def wf6af16_linear(
    output,
    hidden_states,
    weights_2bit,
    weights_4bit,
    scale,
    out_channels=None,
    tokens=None,
    in_channels=None,
    split_k=None,
):
    """FP6-weight FP16-activation linear: ``output = hidden @ (W * scale)^T``.

    Same contract as the generic ``flag_train.deepspeed.wf6af16_linear``;
    only the tuning space and the (forced-off) wide loads differ -- see the
    module docstring.
    """
    logger.debug("TRAIN_ILUVATAR WF6AF16_LINEAR")

    assert hidden_states.dtype == torch.float16, "activations must be fp16"
    assert output.dtype == torch.float16, "output must be fp16"
    assert scale.dtype == torch.float16, "scales must be fp16"
    assert weights_2bit.dtype == torch.uint8 and weights_4bit.dtype == torch.uint8
    assert (
        hidden_states.device.type == flag_train.device
    ), f"wf6af16_linear only supports {flag_train.device} tensors"

    tokens = tokens if tokens is not None else hidden_states.shape[0]
    in_channels = in_channels if in_channels is not None else hidden_states.shape[1]
    out_channels = out_channels if out_channels is not None else scale.shape[0]

    assert hidden_states.shape == (tokens, in_channels)
    assert output.shape == (tokens, out_channels)
    assert in_channels % 4 == 0, "in_channels must be a multiple of 4"
    assert weights_2bit.shape == (out_channels, in_channels // 4)
    assert weights_4bit.shape == (out_channels, in_channels // 2)
    # The kernel indexes raw pointers with row-major strides.
    for t in (output, hidden_states, weights_2bit, weights_4bit, scale):
        assert t.is_contiguous(), "wf6af16_linear requires contiguous tensors"

    if split_k is None:
        split_k = _pick_split_k(tokens, out_channels, in_channels)
    elif in_channels % (split_k * 4) != 0:
        # Each slice must start on a byte boundary (4 codes per byte).
        logger.debug(
            "wf6af16_linear: split_k=%d misaligns K=%d slices, using split_k=1",
            split_k,
            in_channels,
        )
        split_k = 1

    def grid(meta):
        return (
            triton.cdiv(out_channels, meta["BLOCK_M"])
            * triton.cdiv(tokens, meta["BLOCK_N"])
            * split_k,
        )

    with torch_device_fn.device(hidden_states.device):
        if split_k == 1:
            wf6af16_linear_kernel[grid](
                output,
                hidden_states,
                weights_2bit,
                weights_4bit,
                scale,
                out_channels,
                tokens,
                in_channels,
                split_k,
                SPLIT_K_MODE=False,
                WIDE_LOAD=False,
            )
        else:
            workspace = torch.empty(
                (split_k, tokens, out_channels),
                dtype=torch.float32,
                device=hidden_states.device,
            )
            wf6af16_linear_kernel[grid](
                workspace,
                hidden_states,
                weights_2bit,
                weights_4bit,
                scale,
                out_channels,
                tokens,
                in_channels,
                split_k,
                SPLIT_K_MODE=True,
                WIDE_LOAD=False,
            )
            total = tokens * out_channels
            wf6af16_split_k_reduce_kernel[(triton.cdiv(total, 1024),)](
                workspace,
                output,
                total,
                split_k,
                BLOCK_SIZE=1024,
            )

    return output
