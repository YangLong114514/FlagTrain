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
"""Correctness tests for the LAMB operator.

Two oracles, because they fail differently:

* ``lamb_ref`` -- a plain-torch composition of the same contract. It is the
  primary check and runs on any device, so the operator is testable off NVIDIA.
  Under ``--ref cpu`` it is fed CPU operands and genuinely executes there, which
  makes it an independent oracle rather than a second reading of the same device
  arithmetic; see ``_reference_copy``.
* DeepSpeed's ``fused_lamb`` -- the operator this implementation ports. It is an
  independent implementation and its version is recorded in
  ``_DEEPSPEED_VERSION``, as tests/deepspeed/README.md asks for, but it needs the
  ``deepspeed`` package, and on a backend in ``_DEEPSPEED_BASELINE_VENDORS`` below
  it is *required* -- if it will not load there the module raises rather than
  losing the check quietly. On other backends it is an additional check that is
  skipped.

Checking both is not redundant: ``lamb_ref`` is the same arithmetic written from
the same reading of the kernel, so a shared misreading of the contract would
survive it. DeepSpeed's operator is the only oracle that can disagree with that
reading.
"""

import math

import pytest
import torch

import flag_train
from flag_train.deepspeed import lamb

from .. import accuracy_utils as utils

# Hyper-parameters shared by both implementations, mirroring the defaults
# DeepSpeed's fused_lamb is exercised with.
_LR = 1e-3
_BETA1 = 0.9
_BETA2 = 0.999
_MAX_COEFF = 10.0
_MIN_COEFF = 0.01
_EPS = 1e-8
_GRAD_SCALE = 1.0
_DECAY = 0.01

# ---------------------------------------------------------------------------
# Reference implementation
#
# Exists so the operator can be checked against a plain-torch composition of the
# same contract on any device. The DeepSpeed oracle (``fused_lamb``) needs the
# ``deepspeed`` package and a CUDA device, so without a torch reference the
# operator would be untestable off NVIDIA.
# ---------------------------------------------------------------------------


def _reference_copy(tensor):
    """An independent copy of ``tensor``, on the device the reference must use.

    ``--ref cpu`` pushes the reference *computation* onto the CPU, not merely its
    result, so every operand ``lamb_ref`` is handed comes from here and no kernel
    of the reference runs on the accelerator. That is what makes it an
    independent oracle -- a reference evaluated by the same device arithmetic it
    is checking can only confirm the arithmetic agrees with itself.
    ``lamb_ref`` checks this rather than trusting it; the comparison helpers move
    the operator's own output across afterwards.

    ``lamb_ref`` steps its parameters in place, so the copy is not optional: on a
    non-CPU reference ``to_reference`` returns its argument unchanged, and the
    reference would then corrupt the operator's operands.
    """
    return utils.to_reference(tensor).clone()


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
    if utils.TO_CPU:
        # ``--ref cpu`` promises the reference *computation* is on the CPU, not
        # only its result. Asserting the precondition here rather than at the
        # call sites means a test that forgets ``_reference_copy`` fails loudly
        # instead of quietly becoming a second reading of the device arithmetic.
        for name, operand in (("p", p), ("g", g), ("m", m), ("v", v)):
            assert operand.device.type == "cpu", (
                f"--ref cpu must run the reference on the CPU; {name} is on "
                f"{operand.device}"
            )

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


_DEEPSPEED_UNAVAILABLE_MSG = (
    "DeepSpeed's fused_lamb reference is unavailable; install the deepspeed "
    "package on a CUDA host to run this check."
)


# Backends whose reference is DeepSpeed. fused_lamb ships as a CUDA op builder, so
# only a backend that can compile and execute one can host it. On these the
# reference is not optional -- a missing one is an environment fault, and skipping
# quietly would thin the suite without saying so.
_DEEPSPEED_BASELINE_VENDORS = {"nvidia", "hygon"}


def _load_deepspeed_lamb():
    """``(op, version)`` for DeepSpeed's fused_lamb, or ``(None, None)``.

    ``FusedLambBuilder`` JIT-compiles the CUDA source shipped inside the
    ``deepspeed`` package, then reuses the build cached under ``torch_extensions``.
    """
    if flag_train.vendor_name not in _DEEPSPEED_BASELINE_VENDORS:
        return None, None

    try:
        import deepspeed
        from deepspeed.ops.op_builder import FusedLambBuilder

        return FusedLambBuilder().load().lamb, deepspeed.__version__
    except Exception as exc:
        raise RuntimeError(
            f"{flag_train.vendor_name!r} must use DeepSpeed's fused_lamb as its "
            f"reference, but it could not be loaded: {exc!r}. Build deepspeed, or "
            f"drop the backend from _DEEPSPEED_BASELINE_VENDORS."
        ) from exc


# Resolved once, at module import time.
_deepspeed_lamb, _DEEPSPEED_VERSION = _load_deepspeed_lamb()

# The torch reference is always available, so only the DeepSpeed checks skip.
requires_deepspeed_reference = pytest.mark.skipif(
    _deepspeed_lamb is None, reason=_DEEPSPEED_UNAVAILABLE_MSG
)


def _no_p_copy(dtype, device):
    """The empty placeholder that tells every implementation to skip the copy.

    It takes the parameter's device rather than the backend's, because under
    ``--ref cpu`` the operator runs on the backend while the reference runs on
    the CPU, and each is handed the placeholder built for its own operands.
    """
    return torch.empty((0,), dtype=dtype, device=device)


def _step(
    op, param, exp_avg, exp_avg_sq, grad, step, mode, bias_correction, p_copy=None
):
    """Run one in-place step through ``op`` with the shared hyper-parameters.

    Every implementation updates its tensors in place and returns the layer's
    trust ratio, so callers pass tensors they own.
    """
    if p_copy is None:
        p_copy = _no_p_copy(param.dtype, param.device)
    return op(
        param,
        p_copy,
        exp_avg,
        exp_avg_sq,
        grad,
        _LR,
        _BETA1,
        _BETA2,
        _MAX_COEFF,
        _MIN_COEFF,
        _EPS,
        _GRAD_SCALE,
        step,
        mode,
        int(bias_correction),
        _DECAY,
    )


def _assert_step_matches(
    train_coeff,
    train_p,
    train_m,
    train_v,
    other_coeff,
    other_p,
    other_m,
    other_v,
    dtype,
):
    """The trust ratio and the three updated tensors must all agree."""
    utils.train_assert_close(
        utils.to_reference(train_coeff), utils.to_reference(other_coeff), dtype
    )
    utils.train_assert_close(
        utils.to_reference(train_p), utils.to_reference(other_p), dtype
    )
    utils.train_assert_close(
        utils.to_reference(train_m), utils.to_reference(other_m), dtype
    )
    utils.train_assert_close(
        utils.to_reference(train_v), utils.to_reference(other_v), dtype
    )


@pytest.mark.lamb
@pytest.mark.parametrize("shape", [(1024,), (4096,), (16384,)])
@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("bias_correction", [0, 1])
def test_lamb(shape, mode, bias_correction):
    """A single LAMB step must match the torch reference, and DeepSpeed's
    fused_lamb when it is available."""
    dtype = torch.float32

    p = torch.randn(shape, dtype=dtype, device=flag_train.device)
    g = torch.randn(shape, dtype=dtype, device=flag_train.device)
    m = torch.zeros(shape, dtype=dtype, device=flag_train.device)
    v = torch.zeros(shape, dtype=dtype, device=flag_train.device)

    # Every implementation steps in place, so each runs on its own copies.
    train = [p.clone(), m.clone(), v.clone()]
    train_coeff = _step(lamb, *train, g, 1, mode, bias_correction)

    torch_ref = [_reference_copy(p), _reference_copy(m), _reference_copy(v)]
    torch_coeff = _step(
        lamb_ref, *torch_ref, _reference_copy(g), 1, mode, bias_correction
    )
    _assert_step_matches(train_coeff, *train, torch_coeff, *torch_ref, dtype)

    if _deepspeed_lamb is not None:
        deepspeed = [p.clone(), m.clone(), v.clone()]
        deepspeed_coeff = _step(
            _deepspeed_lamb, *deepspeed, g, 1, mode, bias_correction
        )
        _assert_step_matches(train_coeff, *train, deepspeed_coeff, *deepspeed, dtype)


@pytest.mark.lamb
@pytest.mark.parametrize("shape", [(1024,), (4096,)])
def test_lamb_p_copy(shape):
    """The optional output copy must mirror the updated parameter.

    DeepSpeed cannot serve as the oracle here: fused_lamb_cuda_kernel.cu passes
    NULL for p_copy on fp32 operands ("don't output p_copy for fp32, it's wasted
    write"), so its copy is never written at any precision these tests run at.
    The torch reference does write it, so it is the oracle.
    """
    dtype = torch.float32

    p = torch.randn(shape, dtype=dtype, device=flag_train.device)
    g = torch.randn(shape, dtype=dtype, device=flag_train.device)
    m = torch.zeros(shape, dtype=dtype, device=flag_train.device)
    v = torch.zeros(shape, dtype=dtype, device=flag_train.device)

    torch_p, torch_m, torch_v = (
        _reference_copy(p),
        _reference_copy(m),
        _reference_copy(v),
    )
    torch_copy = torch.empty_like(torch_p)
    _step(
        lamb_ref,
        torch_p,
        torch_m,
        torch_v,
        _reference_copy(g),
        1,
        1,
        1,
        p_copy=torch_copy,
    )

    p_copy = torch.empty_like(p)
    _step(lamb, p, m, v, g, 1, 1, 1, p_copy=p_copy)

    utils.train_assert_close(
        utils.to_reference(p_copy), utils.to_reference(torch_copy), dtype
    )
    # The copy must also mirror the parameter the same call produced.
    utils.train_assert_close(utils.to_reference(p_copy), utils.to_reference(p), dtype)


@pytest.mark.lamb
@pytest.mark.parametrize("mode", [0, 1], ids=["eps_inside_sqrt", "eps_outside_sqrt"])
@pytest.mark.parametrize(
    "bias_correction", [True, False], ids=["bias_corr", "no_bias_corr"]
)
def test_lamb_matches_reference(mode, bias_correction):
    """Run several LAMB steps and compare against the torch reference.

    Mirrors DeepSpeed's ``test_fused_adam_matches_reference``: multiple parameter
    tensors, fresh gradients each step, accumulating first/second moments, with
    the operator under test compared to the reference at every step.
    """
    dtype = torch.float32
    torch.manual_seed(0)

    train_params = [
        torch.randn(1024, dtype=dtype, device=flag_train.device) for _ in range(3)
    ]
    # Same values as the operator's operands -- ``to_reference`` copies rather
    # than converts -- but on whichever device the reference is meant to run.
    ref_params = [_reference_copy(p) for p in train_params]
    train_m = [torch.zeros_like(p) for p in train_params]
    ref_m = [torch.zeros_like(p) for p in ref_params]
    train_v = [torch.zeros_like(p) for p in train_params]
    ref_v = [torch.zeros_like(p) for p in ref_params]

    for step in range(1, 6):
        for i in range(len(train_params)):
            grad = torch.randn_like(train_params[i])

            _step(
                lamb_ref,
                ref_params[i],
                ref_m[i],
                ref_v[i],
                _reference_copy(grad),
                step,
                mode,
                bias_correction,
            )
            _step(
                lamb,
                train_params[i],
                train_m[i],
                train_v[i],
                grad,
                step,
                mode,
                bias_correction,
            )

    for train_param, ref_param in zip(train_params, ref_params):
        utils.train_assert_close(
            utils.to_reference(train_param), utils.to_reference(ref_param), dtype
        )
    for train_exp_avg, ref_exp_avg in zip(train_m, ref_m):
        utils.train_assert_close(
            utils.to_reference(train_exp_avg), utils.to_reference(ref_exp_avg), dtype
        )
    for train_exp_avg_sq, ref_exp_avg_sq in zip(train_v, ref_v):
        utils.train_assert_close(
            utils.to_reference(train_exp_avg_sq),
            utils.to_reference(ref_exp_avg_sq),
            dtype,
        )


@pytest.mark.lamb
@requires_deepspeed_reference
@pytest.mark.parametrize("mode", [0, 1], ids=["eps_inside_sqrt", "eps_outside_sqrt"])
@pytest.mark.parametrize("shape", [(1024,), (16384,)])
def test_matches_deepspeed_oracle(mode, shape):
    """Pin the DeepSpeed oracle explicitly, so a run that quietly stopped
    reaching it (deepspeed missing) is visible rather than silently thinner."""
    dtype = torch.float32

    p = torch.randn(shape, dtype=dtype, device=flag_train.device)
    g = torch.randn(shape, dtype=dtype, device=flag_train.device)
    m = torch.zeros(shape, dtype=dtype, device=flag_train.device)
    v = torch.zeros(shape, dtype=dtype, device=flag_train.device)

    train = [p.clone(), m.clone(), v.clone()]
    train_coeff = _step(lamb, *train, g, 1, mode, 1)

    deepspeed = [p.clone(), m.clone(), v.clone()]
    deepspeed_coeff = _step(_deepspeed_lamb, *deepspeed, g, 1, mode, 1)
    _assert_step_matches(train_coeff, *train, deepspeed_coeff, *deepspeed, dtype)
