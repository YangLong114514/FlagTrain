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
"""MetaX-tuned WF6-AF16 linear.

The generic implementation is ``flag_train.deepspeed.wf6af16_linear`` and is
what every backend without an override runs. The kernel *body* is imported
from there unchanged -- the arithmetic is identical to the implementation the
numeric tests cover -- and this module only replaces the tuning around it, for
two MetaX-specific reasons:

* ``num_stages=1`` for every candidate. The MACA backend's software pipeliner
  miscompiles this kernel with deeper pipelining: the activation load's
  predicate depends on the loop bound ``k_end``, and with ``num_stages > 1``
  the same compiled binary flips between right and wrong tiles across
  launches (measured on a C550: 3-7 of 10 identical launches wrong, errors of
  ~1e3 on fp16 outputs). Single-stage compilation never touches the buggy
  path.
* a ``do_bench`` wrapper that reports a launch-time ``RuntimeError`` as
  infinitely slow. MetaX's driver caps per-thread private memory (4 KB by
  default) and rejects over-limit kernels at launch time with a plain
  ``RuntimeError``, which the stock autotuner does not treat as "skip this
  config", so one over-limit config would abort the whole tuning run.

The launcher mirrors the generic one; the pieces that must not drift -- the
split-K heuristic and the reduction kernel -- are imported from the generic
module.

Registration is by name. ``_metax/ops/deepspeed/__init__.py`` re-exports this
module's ``wf6af16_linear``, and ``SpecOpRegistrar`` writes it over the
same-named generic function in ``flag_train.deepspeed`` when that package is
imported -- the namespace the tests and the benchmark read the operator from.
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
    """Autotune candidates for the GEMM kernel on MetaX.

    Same tiles as the generic space, but every entry runs with
    ``num_stages=1`` -- see the module docstring for the pipeliner
    miscompilation this sidesteps.
    """
    configs = []
    for block_m, block_n, block_k, warps in [
        (64, 16, 64, 4),
        (64, 32, 64, 4),
        (64, 64, 64, 4),
        (128, 64, 64, 4),
        (64, 128, 64, 4),
        (128, 128, 32, 8),
        (128, 128, 64, 8),
        (128, 256, 32, 8),
    ]:
        configs.append(
            triton.Config(
                {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k},
                num_warps=warps,
                num_stages=1,
            )
        )
    return configs


def _gemm_do_bench():
    """Autotune bench function that skips configs the driver rejects.

    See the module docstring: over-limit kernels fail at launch with a plain
    ``RuntimeError``, which is reported as infinitely slow so tuning continues
    with the remaining candidates.
    """
    bench = triton.runtime.driver.active.get_benchmarker()

    def do_bench(kernel_call, quantiles):
        try:
            return bench(kernel_call, quantiles=quantiles)
        except RuntimeError as e:
            logger.debug("wf6af16_linear autotune: skipping config: %s", e)
            return [float("inf"), float("inf"), float("inf")]

    return do_bench


wf6af16_linear_kernel = libentry()(
    triton.autotune(
        configs=_gemm_configs(),
        key=["M", "N", "K", "split_k"],
        do_bench=_gemm_do_bench(),
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

    Same contract as the generic ``flag_train.deepspeed.wf6af16_linear``; only
    the kernel tuning differs (see the module docstring).
    """
    logger.debug("TRAIN WF6AF16_LINEAR (MetaX)")

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
