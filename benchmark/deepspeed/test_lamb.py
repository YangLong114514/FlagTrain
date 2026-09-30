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
"""Performance benchmark for the LAMB operator.

The baseline follows ``_DEEPSPEED_BASELINE_VENDORS`` below:

* on a listed backend, DeepSpeed's ``fused_lamb`` **is** the baseline. If it cannot
  be built there the module raises rather than falling back -- measuring the torch
  reference and reporting it under the same name would answer a different question;
* on any other backend the baseline is ``lamb_ref``, the plain-torch composition of
  the same contract, whether or not ``deepspeed`` is installed there. It is not a
  competitor, and the speedup there says how far the kernel is from *a* correct
  implementation rather than from the best one.

Both baselines take the same argument tuple as the operator, so unlike the
blocked-flash benchmark there is nothing to pre-compute outside the timed region.
"""

import math

import pytest
import torch

import flag_train
from flag_train.deepspeed import lamb

from .. import base

# One-dimensional parameter tensors of realistic optimizer sizes. LAMB (like the
# other fused optimizers) operates on a flattened parameter array, so the element
# count is the only shape dimension that matters.
_LAMB_SHAPES = [
    (1024,),
    (4096,),
    (16384,),
    (65536,),
    (262144,),
    (1048576,),
    (11048576,),
]

# Fixed hyper-parameters shared by both implementations so the comparison is
# apples-to-apples. These mirror DeepSpeed's own fused_lamb defaults.
_LR = 1e-3
_BETA1 = 0.9
_BETA2 = 0.999
_MAX_COEFF = 10.0
_MIN_COEFF = 0.01
_EPS = 1e-8
_GRAD_SCALE = 1.0
_STEP = 1
_MODE = 1  # eps outside the sqrt (DeepSpeed default)
_BIAS_CORRECTION = 1
_DECAY = 0.01

# ---------------------------------------------------------------------------
# Reference implementation
#
# Exists so the operator can be checked against a plain-torch composition of the
# same contract on any device. The DeepSpeed oracle (``fused_lamb``) needs the
# ``deepspeed`` package and a CUDA device, so without a torch reference the
# operator would be untestable off NVIDIA.
# ---------------------------------------------------------------------------


def lamb_ref(
    p,
    p_copy,
    m,
    v,
    g,
    lr,
    beta1,
    beta2,
    max_coeff,
    min_coeff,
    eps,
    grad_scale,
    step,
    mode,
    bias_correction,
    decay,
):
    """Reference for the LAMB operator, composed from plain torch ops.

    Steps ``p``/``m``/``v`` in place and returns the trust ratio, matching the
    operator's contract including the order of operations, since the two are
    compared bit-for-bit-ish rather than through a fuzzy formula.
    """
    p_old = p.clone()
    scaled_grad = g / grad_scale
    m_new = beta1 * m + (1 - beta1) * scaled_grad
    v_new = beta2 * v + (1 - beta2) * scaled_grad * scaled_grad

    if mode == 0:
        denom = torch.sqrt(v_new + eps)
    else:
        denom = torch.sqrt(v_new) + eps
    update = m_new / denom + decay * p_old

    # The trust ratio is taken from the *incoming* weights and the update built
    # from them, which is why ``p_old`` is kept rather than reusing ``p``.
    w_norm = torch.sqrt(torch.sum(p_old * p_old))
    u_norm = torch.sqrt(torch.sum(update * update))
    # Either norm being zero leaves the ratio at 1.0 rather than clamped, matching
    # lamb_cuda_kernel_part3.
    if w_norm.item() == 0.0 or u_norm.item() == 0.0:
        coeff = torch.ones((), dtype=p.dtype, device=p.device)
    else:
        coeff = torch.clamp(w_norm / u_norm, min_coeff, max_coeff)

    if bias_correction == 1:
        step_size = lr * math.sqrt(1 - beta2**step) / (1 - beta1**step)
    else:
        step_size = lr

    p_new = p_old - step_size * coeff * update
    p.copy_(p_new)
    m.copy_(m_new)
    v.copy_(v_new)
    if p_copy.numel() > 0:
        p_copy.copy_(p_new)

    return coeff.reshape(1).to(torch.float32)


# Backends whose baseline is DeepSpeed. fused_lamb ships as a CUDA op builder, so
# only a backend that can compile and execute one can host it. On these the
# baseline is not optional -- if it will not load, timing the torch reference and
# reporting it under the DeepSpeed baseline's name would answer another question.
_DEEPSPEED_BASELINE_VENDORS = {"nvidia", "hygon"}


def _load_deepspeed_lamb():
    """DeepSpeed's fused_lamb, or ``None`` on a backend that does not use it.

    ``FusedLambBuilder`` JIT-compiles the CUDA source shipped inside the
    ``deepspeed`` package, then reuses the build cached under ``torch_extensions``.
    """
    if flag_train.vendor_name not in _DEEPSPEED_BASELINE_VENDORS:
        return None

    try:
        from deepspeed.ops.op_builder import FusedLambBuilder

        return FusedLambBuilder().load().lamb
    except Exception as exc:
        raise RuntimeError(
            f"{flag_train.vendor_name!r} must use DeepSpeed's fused_lamb as its "
            f"baseline, but it could not be loaded: {exc!r}. Build deepspeed, or "
            f"drop the backend from _DEEPSPEED_BASELINE_VENDORS."
        ) from exc


# Resolved once, so the first-use JIT compile is not counted in the measurement.
_deepspeed_lamb = _load_deepspeed_lamb()

_BASELINE = (
    "deepspeed fused_lamb" if _deepspeed_lamb is not None else "lamb_ref (torch)"
)


class LambBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _LAMB_SHAPES

    def set_shapes(self, shape_file=None):
        # lamb needs 5 tensors per case (param, p_copy, m, v, g); keep the shapes
        # moderate and explicit to avoid OOM on CI GPUs.
        self.shapes = list(_LAMB_SHAPES)

    def set_more_shapes(self):
        return []


def lamb_input_fn(shape, dtype, device):
    p = torch.randn(shape, dtype=dtype, device=device)
    p_copy = torch.empty((0,), dtype=dtype, device=device)  # skip the copy
    m = torch.zeros(shape, dtype=dtype, device=device)
    v = torch.zeros(shape, dtype=dtype, device=device)
    g = torch.randn(shape, dtype=dtype, device=device)
    yield p, p_copy, m, v, g


def _call(op, p, p_copy, m, v, g):
    return op(
        p,
        p_copy,
        m,
        v,
        g,
        _LR,
        _BETA1,
        _BETA2,
        _MAX_COEFF,
        _MIN_COEFF,
        _EPS,
        _GRAD_SCALE,
        _STEP,
        _MODE,
        _BIAS_CORRECTION,
        _DECAY,
    )


def torch_op(p, p_copy, m, v, g):
    """Baseline, chosen by platform. See the module docstring."""
    baseline = _deepspeed_lamb if _deepspeed_lamb is not None else lamb_ref
    return _call(baseline, p, p_copy, m, v, g)


def train_op(p, p_copy, m, v, g):
    """The operator under test."""
    return _call(lamb, p, p_copy, m, v, g)


@pytest.mark.lamb
def test_lamb_perf():
    print(f"\nBaseline: {_BASELINE}")

    bench = LambBenchmark(
        input_fn=lamb_input_fn,
        op_name="lamb",
        torch_op=torch_op,
        # fused_lamb only supports float32 parameters/state.
        dtypes=[torch.float32],
    )
    bench.set_train(train_op)
    bench.run()
