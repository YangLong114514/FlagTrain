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
import logging
import math

import torch
import triton
import triton.language as tl

from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry
from flag_train.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def lamb_part1_kernel(
    p_ptr,
    m_ptr,
    v_ptr,
    g_ptr,
    n,
    grad_scale,
    b1,
    b2,
    eps,
    decay,
    max_coeff,
    min_coeff,
    norms_ptr,
    counter_ptr,
    lamb_coeff_ptr,
    mode: tl.constexpr,
    REDUCE_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Adam moment update, partial norm reduction and the trust ratio.

    Mirrors ``lamb_cuda_kernel_part1`` and ``lamb_cuda_kernel_part2`` in
    DeepSpeed: update the first/second moments, build the Adam update vector
    ``update = m/denom + decay*w``, reduce the squared weight/update norms
    inside this program, and emit one pair of partial sums per program.

    DeepSpeed then launches a second kernel to fold those partials, because no
    program can see another's result without one.  Here the arrival counter
    supplies that ordering inside this kernel instead, which removes a launch
    from the critical path of a step this small.  The program that completes the
    count folds the partials, publishes the trust ratio, and resets the counter
    so the scratch is ready for the next step.

    ``acq_rel`` on the arrival makes each program's partial stores visible to
    the one that completes the count, and that program's trust ratio visible to
    the part 3 launch that follows on the stream.
    """
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    p = tl.load(p_ptr + offsets, mask=mask, other=0.0)
    g = tl.load(g_ptr + offsets, mask=mask, other=0.0)
    m = tl.load(m_ptr + offsets, mask=mask, other=0.0)
    v = tl.load(v_ptr + offsets, mask=mask, other=0.0)

    scaled_grad = g / grad_scale
    m_new = b1 * m + (1 - b1) * scaled_grad
    v_new = b2 * v + (1 - b2) * scaled_grad * scaled_grad

    if mode == 0:
        denom = tl.sqrt(v_new + eps)
    else:
        denom = tl.sqrt(v_new) + eps

    update = m_new / denom + decay * p

    tl.store(m_ptr + offsets, m_new, mask=mask)
    tl.store(v_ptr + offsets, v_new, mask=mask)

    # Masked lanes loaded 0.0, so their squares contribute nothing to the sum.
    reg_w = tl.sum(p * p)
    reg_u = tl.sum(update * update)

    num_blocks = tle.num_programs(0)
    tl.store(norms_ptr + pid, reg_w)
    tl.store(norms_ptr + num_blocks + pid, reg_u)

    if tl.atomic_add(counter_ptr, 1, sem="acq_rel") == num_blocks - 1:
        red_offsets = tl.arange(0, REDUCE_BLOCK)
        w_partial = tl.zeros((REDUCE_BLOCK,), dtype=tl.float32)
        u_partial = tl.zeros((REDUCE_BLOCK,), dtype=tl.float32)

        for start in range(0, num_blocks, REDUCE_BLOCK):
            idx = start + red_offsets
            in_range = idx < num_blocks
            w_partial += tl.load(norms_ptr + idx, mask=in_range, other=0.0)
            u_partial += tl.load(norms_ptr + num_blocks + idx, mask=in_range, other=0.0)

        w_norm = tl.sqrt(tl.sum(w_partial))
        u_norm = tl.sqrt(tl.sum(u_partial))
        # Guard the division so a zero norm yields coeff == 1.0 (DeepSpeed leaves
        # it unclamped in that case); tl.where evaluates both sides, so keep the
        # denominator finite even when u_norm == 0.
        u_denom = tl.where(u_norm == 0.0, 1.0, u_norm)
        trust = w_norm / u_denom
        trust = tl.minimum(tl.maximum(trust, min_coeff), max_coeff)
        tl.store(
            lamb_coeff_ptr,
            tl.where((w_norm == 0.0) | (u_norm == 0.0), 1.0, trust),
        )
        tl.store(counter_ptr, 0)


@libentry()
@triton.jit
def lamb_part3_kernel(
    p_ptr,
    p_copy_ptr,
    m_ptr,
    v_ptr,
    n,
    eps,
    step_size,
    decay,
    lamb_coeff_ptr,
    mode: tl.constexpr,
    HAS_P_COPY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply the parameter update scaled by the layer-wise trust ratio.

    Mirrors ``lamb_cuda_kernel_part3`` in DeepSpeed: ``trust = ||w|| / ||u||``
    (1.0 when either norm is zero), clamped to ``[min_coeff, max_coeff]``, then
    ``p = p - step_size * trust * update`` with ``update`` recomputed from the
    already-updated moments.  The ratio itself is already folded over every
    program by part 1, so this kernel only reads it.
    """
    lamb_coeff = tl.load(lamb_coeff_ptr)

    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    p = tl.load(p_ptr + offsets, mask=mask, other=0.0)
    m = tl.load(m_ptr + offsets, mask=mask, other=0.0)
    v = tl.load(v_ptr + offsets, mask=mask, other=0.0)

    if mode == 0:
        denom = tl.sqrt(v + eps)
    else:
        denom = tl.sqrt(v) + eps

    update = m / denom + decay * p
    p_new = p - step_size * lamb_coeff * update

    tl.store(p_ptr + offsets, p_new, mask=mask)
    if HAS_P_COPY:
        tl.store(p_copy_ptr + offsets, p_new.to(p_copy_ptr.dtype.element_ty), mask=mask)


def lamb(
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
    """Fused LAMB (Layer-wise Adaptive Moments) optimizer step.

    Faithful port of DeepSpeed's ``fused_lamb`` CUDA operator. Performs one
    Adam-style moment update, then scales the step per-layer by the trust ratio
    ``||w|| / ||update||`` (clamped to ``[min_coeff, max_coeff]``), which makes
    training robust to very large batch sizes.

    Args:
        p (Tensor): model parameter (fp32).
        p_copy (Tensor): optional reduced-precision copy of the updated weights;
            pass an empty tensor to skip.
        m (Tensor): first moment estimate (fp32).
        v (Tensor): second moment estimate (fp32).
        g (Tensor): gradient (fp32).
        lr (float): learning rate.
        beta1 (float): coefficient for the running average of the gradient.
        beta2 (float): coefficient for the running average of the squared gradient.
        max_coeff (float): upper bound of the trust ratio.
        min_coeff (float): lower bound of the trust ratio.
        eps (float): term added to the denominator for numerical stability.
        grad_scale (float): factor dividing the gradient before the update.
        step (int): optimizer step, used only for bias correction.
        mode (int): 0 keeps eps under the square root, 1 keeps it outside.
        bias_correction (int): 1 enables Adam bias correction, 0 disables it.
        decay (float): weight decay added to the update vector.

    Returns:
        Tensor: the per-layer trust ratio as a one-element fp32 tensor. It is
            allocated per call, so the caller may keep it.
    """
    logger.debug("TRAIN LAMB")

    assert p.dtype == torch.float32, "lamb only supports float32 parameters"
    # Accept whatever the active backend calls its device -- 'cuda' on
    # nvidia/hygon, 'npu' on ascend -- rather than hard-coding CUDA. A CPU
    # tensor still has to be rejected: the kernels would be launched against
    # memory the accelerator cannot see.
    assert (
        p.device.type == runtime_device.name
    ), f"lamb only supports {runtime_device.name} tensors"

    n = p.numel()
    assert m.numel() == n and v.numel() == n and g.numel() == n
    assert p_copy.numel() == 0 or p_copy.numel() == n

    # Adam bias correction, matching DeepSpeed's step_size computation.
    if bias_correction == 1:
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        step_size = lr * math.sqrt(bias_correction2) / bias_correction1
    else:
        step_size = lr

    # A wide block leaves only a handful of programs for a mid-sized tensor, so
    # the kernels spend their time waiting on memory instead of covering it.
    # Capping the block at 1024 elements keeps enough programs in flight to hide
    # that latency without making the cross-block reduction expensive.
    BLOCK_SIZE = triton.next_power_of_2(n)
    BLOCK_SIZE = max(BLOCK_SIZE, 128)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)
    num_blocks = triton.cdiv(n, BLOCK_SIZE)

    # Reduction workspace, allocated per call.  A module-level pool keyed by
    # (device, num_blocks) has no natural bound, so a caller cycling through
    # element counts would grow it without limit; these three small allocations
    # are the whole cost of not keeping one.  ``counter`` is the only buffer that
    # must start at zero, and being the only zeroed one is why it costs a memset.
    norms = torch.empty((2 * num_blocks,), dtype=torch.float32, device=p.device)
    counter = torch.zeros((1,), dtype=torch.int32, device=p.device)
    lamb_coeff_val = torch.empty((1,), dtype=torch.float32, device=p.device)
    reduce_block = min(max(triton.next_power_of_2(num_blocks), 32), 2048)

    # ``p_copy_ptr`` is only dereferenced when HAS_P_COPY is true, so when the
    # copy is not requested the parameter doubles as a stand-in pointer and no
    # dummy tensor is allocated.
    has_p_copy = p_copy.numel() > 0
    p_copy_in = p_copy if has_p_copy else p

    with torch_device_fn.device(p.device):
        lamb_part1_kernel[(num_blocks,)](
            p,
            m,
            v,
            g,
            n,
            grad_scale,
            beta1,
            beta2,
            eps,
            decay,
            max_coeff,
            min_coeff,
            norms,
            counter,
            lamb_coeff_val,
            mode=mode,
            REDUCE_BLOCK=reduce_block,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        lamb_part3_kernel[(num_blocks,)](
            p,
            p_copy_in,
            m,
            v,
            n,
            eps,
            step_size,
            decay,
            lamb_coeff_val,
            mode=mode,
            HAS_P_COPY=has_p_copy,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return lamb_coeff_val
