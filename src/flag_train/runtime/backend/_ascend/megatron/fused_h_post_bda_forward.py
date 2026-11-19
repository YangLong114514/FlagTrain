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
"""Ascend Triton implementation for fused mHC H_post BDA forward."""

import logging
from contextlib import nullcontext
from typing import Optional

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime.driver import driver

from flag_train.megatron.fused_h_post_bda_forward import (
    _disable_aabs,
    _triton_hpb_fwd_kernel,
    _validate_inputs,
)
from flag_train.runtime import torch_device_fn

logger = logging.getLogger(__name__)


def _device_context(device):
    if torch_device_fn.current_device() == device.index:
        return nullcontext()
    return torch_device_fn.device(device)


_FIXED_CONFIG = {"BLOCK_C": 64, "BLOCK_S": 1, "num_warps": 4}
_MAX_CORE_DIM = 65535
_DOT_BLOCK_C = 2048
_DOT_LAUNCH_TILES = 4096
_FIXED_LAUNCH_CACHE = {}


def _launch_fixed_tiles(kernel, num_tiles, *args, offset_index, num_warps):
    specialization = []
    for param, arg in zip(kernel.params, args):
        if param.is_constexpr:
            specialization.append(arg)
        elif hasattr(arg, "data_ptr"):
            specialization.append((arg.dtype, arg.data_ptr() % 16))
        elif isinstance(arg, int):
            specialization.append((arg == 1, arg % 16 == 0))
        else:
            specialization.append((type(arg), arg))
    specialization = tuple(specialization)
    device = torch_device_fn.current_device()
    stream = driver.active.get_current_stream(device)
    fast_launch = not triton.knobs.runtime.launch_enter_hook.calls and not (
        triton.knobs.runtime.launch_exit_hook.calls
    )
    for tile_offset in range(0, num_tiles, _DOT_LAUNCH_TILES):
        grid = (min(num_tiles - tile_offset, _DOT_LAUNCH_TILES),)
        key = (kernel.cache_key, device, grid, num_warps, specialization)
        entry = _FIXED_LAUNCH_CACHE.get(key)
        batch_args = args[:offset_index] + (tile_offset,) + args[offset_index + 1 :]
        if entry is None:
            compiled = kernel.warmup(*batch_args, grid=grid, num_warps=num_warps)
            entry = (compiled, compiled[(grid[0], 1, 1)])
            _FIXED_LAUNCH_CACHE[key] = entry
        compiled, runner = entry
        if fast_launch:
            compiled.run(
                grid[0],
                1,
                1,
                stream,
                compiled.function,
                compiled.packed_metadata,
                None,
                None,
                None,
                *batch_args,
            )
        else:
            runner(*batch_args, stream=stream)


@triton.jit
def _triton_hpb_fwd_dot_kernel(
    hr_ptr,
    orig_ptr,
    hp_ptr,
    x_ptr,
    bias_ptr,
    out_ptr,
    tile_offset,
    C: tl.constexpr,
    N: tl.constexpr,
    NP: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    C_BLOCKS: tl.constexpr,
):
    tile = tl.program_id(0) + tile_offset
    pid_s = tile // C_BLOCKS
    pid_c = tile % C_BLOCKS
    offs_n = tl.arange(0, NP)
    mask_n = offs_n < N
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    mask = mask_n[:, None] & mask_c[None, :]
    hr = tl.load(
        hr_ptr + pid_s * N * N + offs_n[None, :] * N + offs_n[:, None],
        mask=mask_n[:, None] & mask_n[None, :],
        other=0.0,
    )
    orig = tl.load(
        orig_ptr + pid_s * N * C + offs_n[:, None] * C + offs_c[None, :],
        mask=mask,
        other=0.0,
    )
    hp = tl.load(hp_ptr + pid_s * N + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    x_tile = tl.load(x_ptr + pid_s * C + offs_c, mask=mask_c, other=0.0).to(tl.float32)
    if HAS_BIAS:
        x_tile += tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
    out = tl.dot(hr, orig, input_precision="ieee") + hp[:, None] * x_tile[None, :]
    tl.store(
        out_ptr + pid_s * N * C + offs_n[:, None] * C + offs_c[None, :],
        out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _triton_hpb_fwd_row_dot_kernel(
    hr_ptr,
    orig_ptr,
    hp_ptr,
    x_ptr,
    bias_ptr,
    out_ptr,
    tile_offset,
    sb,
    C: tl.constexpr,
    N: tl.constexpr,
    NP: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    C_BLOCKS: tl.constexpr,
):
    pid_s = tl.program_id(0) + tile_offset
    offs_n = tl.arange(0, NP)
    mask_n = offs_n < N
    hr = tl.load(
        hr_ptr + pid_s * N * N + offs_n[None, :] * N + offs_n[:, None],
        mask=mask_n[:, None] & mask_n[None, :],
        other=0.0,
    )
    hp = tl.load(hp_ptr + pid_s * N + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    for pid_c in range(0, C_BLOCKS):
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        mask = mask_n[:, None] & mask_c[None, :]
        orig = tl.load(
            orig_ptr + pid_s * N * C + offs_n[:, None] * C + offs_c[None, :],
            mask=mask,
            other=0.0,
        )
        x_tile = tl.load(x_ptr + pid_s * C + offs_c, mask=mask_c, other=0.0).to(
            tl.float32
        )
        if HAS_BIAS:
            x_tile += tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
        out = tl.dot(hr, orig, input_precision="ieee") + hp[:, None] * x_tile[None, :]
        tl.store(
            out_ptr + pid_s * N * C + offs_n[:, None] * C + offs_c[None, :],
            out.to(out_ptr.dtype.element_ty),
            mask=mask,
        )


@triton.jit
def _triton_hpb_fwd_grouped_dot_kernel(
    hr_ptr,
    orig_ptr,
    hp_ptr,
    x_ptr,
    bias_ptr,
    out_ptr,
    tile_offset,
    C: tl.constexpr,
    N: tl.constexpr,
    NP: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    C_BLOCKS: tl.constexpr,
    C_GROUP: tl.constexpr,
    C_GROUPS: tl.constexpr,
):
    tile = tl.program_id(0) + tile_offset
    pid_s = tile // C_GROUPS
    pid_group = tile % C_GROUPS
    offs_n = tl.arange(0, NP)
    mask_n = offs_n < N
    hr = tl.load(
        hr_ptr + pid_s * N * N + offs_n[None, :] * N + offs_n[:, None],
        mask=mask_n[:, None] & mask_n[None, :],
        other=0.0,
    )
    hp = tl.load(hp_ptr + pid_s * N + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    for c_offset in range(0, C_GROUP):
        pid_c = pid_group * C_GROUP + c_offset
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        mask = mask_n[:, None] & mask_c[None, :]
        orig = tl.load(
            orig_ptr + pid_s * N * C + offs_n[:, None] * C + offs_c[None, :],
            mask=mask,
            other=0.0,
        )
        x_tile = tl.load(x_ptr + pid_s * C + offs_c, mask=mask_c, other=0.0).to(
            tl.float32
        )
        if HAS_BIAS:
            x_tile += tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
        out = tl.dot(hr, orig, input_precision="ieee") + hp[:, None] * x_tile[None, :]
        tl.store(
            out_ptr + pid_s * N * C + offs_n[:, None] * C + offs_c[None, :],
            out.to(out_ptr.dtype.element_ty),
            mask=mask,
        )


@triton.jit
def _triton_hpb_fwd_vector4_kernel(
    hr_ptr,
    orig_ptr,
    hp_ptr,
    x_ptr,
    bias_ptr,
    out_ptr,
    tile_offset,
    sb,
    C: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_s = tl.program_id(0) + tile_offset
    hr_base = hr_ptr + pid_s * 16
    h00 = tl.load(hr_base + 0)
    h01 = tl.load(hr_base + 1)
    h02 = tl.load(hr_base + 2)
    h03 = tl.load(hr_base + 3)
    h10 = tl.load(hr_base + 4)
    h11 = tl.load(hr_base + 5)
    h12 = tl.load(hr_base + 6)
    h13 = tl.load(hr_base + 7)
    h20 = tl.load(hr_base + 8)
    h21 = tl.load(hr_base + 9)
    h22 = tl.load(hr_base + 10)
    h23 = tl.load(hr_base + 11)
    h30 = tl.load(hr_base + 12)
    h31 = tl.load(hr_base + 13)
    h32 = tl.load(hr_base + 14)
    h33 = tl.load(hr_base + 15)
    hp_base = hp_ptr + pid_s * 4
    hp0 = tl.load(hp_base + 0).to(tl.float32)
    hp1 = tl.load(hp_base + 1).to(tl.float32)
    hp2 = tl.load(hp_base + 2).to(tl.float32)
    hp3 = tl.load(hp_base + 3).to(tl.float32)
    for c_start in range(0, C, BLOCK_C):
        offs_c = c_start + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        xb = tl.load(x_ptr + pid_s * C + offs_c, mask=mask_c, other=0.0).to(tl.float32)
        if HAS_BIAS:
            xb += tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
        orig0 = tl.load(orig_ptr + pid_s * 4 * C + offs_c, mask=mask_c, other=0.0).to(
            tl.float32
        )
        orig1 = tl.load(
            orig_ptr + pid_s * 4 * C + C + offs_c, mask=mask_c, other=0.0
        ).to(tl.float32)
        orig2 = tl.load(
            orig_ptr + pid_s * 4 * C + 2 * C + offs_c, mask=mask_c, other=0.0
        ).to(tl.float32)
        orig3 = tl.load(
            orig_ptr + pid_s * 4 * C + 3 * C + offs_c, mask=mask_c, other=0.0
        ).to(tl.float32)
        out0 = h00 * orig0 + h10 * orig1 + h20 * orig2 + h30 * orig3 + hp0 * xb
        out1 = h01 * orig0 + h11 * orig1 + h21 * orig2 + h31 * orig3 + hp1 * xb
        out2 = h02 * orig0 + h12 * orig1 + h22 * orig2 + h32 * orig3 + hp2 * xb
        out3 = h03 * orig0 + h13 * orig1 + h23 * orig2 + h33 * orig3 + hp3 * xb
        out_base = out_ptr + pid_s * 4 * C + offs_c
        tl.store(out_base + 0 * C, out0.to(out_ptr.dtype.element_ty), mask=mask_c)
        tl.store(out_base + 1 * C, out1.to(out_ptr.dtype.element_ty), mask=mask_c)
        tl.store(out_base + 2 * C, out2.to(out_ptr.dtype.element_ty), mask=mask_c)
        tl.store(out_base + 3 * C, out3.to(out_ptr.dtype.element_ty), mask=mask_c)


@triton.jit
def _triton_hpb_fwd_vector8_kernel(
    hr_ptr,
    orig_ptr,
    hp_ptr,
    x_ptr,
    bias_ptr,
    out_ptr,
    tile_offset,
    sb,
    C: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_s = tl.program_id(0) + tile_offset
    hr_base = hr_ptr + pid_s * 64
    h00 = tl.load(hr_base + 0)
    h01 = tl.load(hr_base + 1)
    h02 = tl.load(hr_base + 2)
    h03 = tl.load(hr_base + 3)
    h04 = tl.load(hr_base + 4)
    h05 = tl.load(hr_base + 5)
    h06 = tl.load(hr_base + 6)
    h07 = tl.load(hr_base + 7)
    h10 = tl.load(hr_base + 8)
    h11 = tl.load(hr_base + 9)
    h12 = tl.load(hr_base + 10)
    h13 = tl.load(hr_base + 11)
    h14 = tl.load(hr_base + 12)
    h15 = tl.load(hr_base + 13)
    h16 = tl.load(hr_base + 14)
    h17 = tl.load(hr_base + 15)
    h20 = tl.load(hr_base + 16)
    h21 = tl.load(hr_base + 17)
    h22 = tl.load(hr_base + 18)
    h23 = tl.load(hr_base + 19)
    h24 = tl.load(hr_base + 20)
    h25 = tl.load(hr_base + 21)
    h26 = tl.load(hr_base + 22)
    h27 = tl.load(hr_base + 23)
    h30 = tl.load(hr_base + 24)
    h31 = tl.load(hr_base + 25)
    h32 = tl.load(hr_base + 26)
    h33 = tl.load(hr_base + 27)
    h34 = tl.load(hr_base + 28)
    h35 = tl.load(hr_base + 29)
    h36 = tl.load(hr_base + 30)
    h37 = tl.load(hr_base + 31)
    h40 = tl.load(hr_base + 32)
    h41 = tl.load(hr_base + 33)
    h42 = tl.load(hr_base + 34)
    h43 = tl.load(hr_base + 35)
    h44 = tl.load(hr_base + 36)
    h45 = tl.load(hr_base + 37)
    h46 = tl.load(hr_base + 38)
    h47 = tl.load(hr_base + 39)
    h50 = tl.load(hr_base + 40)
    h51 = tl.load(hr_base + 41)
    h52 = tl.load(hr_base + 42)
    h53 = tl.load(hr_base + 43)
    h54 = tl.load(hr_base + 44)
    h55 = tl.load(hr_base + 45)
    h56 = tl.load(hr_base + 46)
    h57 = tl.load(hr_base + 47)
    h60 = tl.load(hr_base + 48)
    h61 = tl.load(hr_base + 49)
    h62 = tl.load(hr_base + 50)
    h63 = tl.load(hr_base + 51)
    h64 = tl.load(hr_base + 52)
    h65 = tl.load(hr_base + 53)
    h66 = tl.load(hr_base + 54)
    h67 = tl.load(hr_base + 55)
    h70 = tl.load(hr_base + 56)
    h71 = tl.load(hr_base + 57)
    h72 = tl.load(hr_base + 58)
    h73 = tl.load(hr_base + 59)
    h74 = tl.load(hr_base + 60)
    h75 = tl.load(hr_base + 61)
    h76 = tl.load(hr_base + 62)
    h77 = tl.load(hr_base + 63)
    hp_base = hp_ptr + pid_s * 8
    hp0 = tl.load(hp_base + 0).to(tl.float32)
    hp1 = tl.load(hp_base + 1).to(tl.float32)
    hp2 = tl.load(hp_base + 2).to(tl.float32)
    hp3 = tl.load(hp_base + 3).to(tl.float32)
    hp4 = tl.load(hp_base + 4).to(tl.float32)
    hp5 = tl.load(hp_base + 5).to(tl.float32)
    hp6 = tl.load(hp_base + 6).to(tl.float32)
    hp7 = tl.load(hp_base + 7).to(tl.float32)
    for c_start in range(0, C, BLOCK_C):
        offs_c = c_start + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        xb = tl.load(x_ptr + pid_s * C + offs_c, mask=mask_c, other=0.0).to(tl.float32)
        if HAS_BIAS:
            xb += tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
        orig_base = orig_ptr + pid_s * 8 * C + offs_c
        orig0 = tl.load(orig_base, mask=mask_c, other=0.0).to(tl.float32)
        orig1 = tl.load(orig_base + C, mask=mask_c, other=0.0).to(tl.float32)
        orig2 = tl.load(orig_base + 2 * C, mask=mask_c, other=0.0).to(tl.float32)
        orig3 = tl.load(orig_base + 3 * C, mask=mask_c, other=0.0).to(tl.float32)
        orig4 = tl.load(orig_base + 4 * C, mask=mask_c, other=0.0).to(tl.float32)
        orig5 = tl.load(orig_base + 5 * C, mask=mask_c, other=0.0).to(tl.float32)
        orig6 = tl.load(orig_base + 6 * C, mask=mask_c, other=0.0).to(tl.float32)
        orig7 = tl.load(orig_base + 7 * C, mask=mask_c, other=0.0).to(tl.float32)
        out0 = (
            h00 * orig0
            + h10 * orig1
            + h20 * orig2
            + h30 * orig3
            + h40 * orig4
            + h50 * orig5
            + h60 * orig6
            + h70 * orig7
            + hp0 * xb
        )
        out1 = (
            h01 * orig0
            + h11 * orig1
            + h21 * orig2
            + h31 * orig3
            + h41 * orig4
            + h51 * orig5
            + h61 * orig6
            + h71 * orig7
            + hp1 * xb
        )
        out2 = (
            h02 * orig0
            + h12 * orig1
            + h22 * orig2
            + h32 * orig3
            + h42 * orig4
            + h52 * orig5
            + h62 * orig6
            + h72 * orig7
            + hp2 * xb
        )
        out3 = (
            h03 * orig0
            + h13 * orig1
            + h23 * orig2
            + h33 * orig3
            + h43 * orig4
            + h53 * orig5
            + h63 * orig6
            + h73 * orig7
            + hp3 * xb
        )
        out4 = (
            h04 * orig0
            + h14 * orig1
            + h24 * orig2
            + h34 * orig3
            + h44 * orig4
            + h54 * orig5
            + h64 * orig6
            + h74 * orig7
            + hp4 * xb
        )
        out5 = (
            h05 * orig0
            + h15 * orig1
            + h25 * orig2
            + h35 * orig3
            + h45 * orig4
            + h55 * orig5
            + h65 * orig6
            + h75 * orig7
            + hp5 * xb
        )
        out6 = (
            h06 * orig0
            + h16 * orig1
            + h26 * orig2
            + h36 * orig3
            + h46 * orig4
            + h56 * orig5
            + h66 * orig6
            + h76 * orig7
            + hp6 * xb
        )
        out7 = (
            h07 * orig0
            + h17 * orig1
            + h27 * orig2
            + h37 * orig3
            + h47 * orig4
            + h57 * orig5
            + h67 * orig6
            + h77 * orig7
            + hp7 * xb
        )
        out_base = out_ptr + pid_s * 8 * C + offs_c
        tl.store(out_base, out0.to(out_ptr.dtype.element_ty), mask=mask_c)
        tl.store(out_base + C, out1.to(out_ptr.dtype.element_ty), mask=mask_c)
        tl.store(out_base + 2 * C, out2.to(out_ptr.dtype.element_ty), mask=mask_c)
        tl.store(out_base + 3 * C, out3.to(out_ptr.dtype.element_ty), mask=mask_c)
        tl.store(out_base + 4 * C, out4.to(out_ptr.dtype.element_ty), mask=mask_c)
        tl.store(out_base + 5 * C, out5.to(out_ptr.dtype.element_ty), mask=mask_c)
        tl.store(out_base + 6 * C, out6.to(out_ptr.dtype.element_ty), mask=mask_c)
        tl.store(out_base + 7 * C, out7.to(out_ptr.dtype.element_ty), mask=mask_c)


@triton.jit
def _triton_hpb_fwd_vector4_rows_kernel(
    hr_ptr,
    orig_ptr,
    hp_ptr,
    x_ptr,
    bias_ptr,
    out_ptr,
    tile_offset,
    sb,
    C: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    ROWS: tl.constexpr,
):
    pid = tl.program_id(0) + tile_offset
    for row in range(ROWS):
        pid_s = pid * ROWS + row
        hr_base = hr_ptr + pid_s * 16
        h00 = tl.load(hr_base + 0)
        h01 = tl.load(hr_base + 1)
        h02 = tl.load(hr_base + 2)
        h03 = tl.load(hr_base + 3)
        h10 = tl.load(hr_base + 4)
        h11 = tl.load(hr_base + 5)
        h12 = tl.load(hr_base + 6)
        h13 = tl.load(hr_base + 7)
        h20 = tl.load(hr_base + 8)
        h21 = tl.load(hr_base + 9)
        h22 = tl.load(hr_base + 10)
        h23 = tl.load(hr_base + 11)
        h30 = tl.load(hr_base + 12)
        h31 = tl.load(hr_base + 13)
        h32 = tl.load(hr_base + 14)
        h33 = tl.load(hr_base + 15)
        hp_base = hp_ptr + pid_s * 4
        hp0 = tl.load(hp_base + 0).to(tl.float32)
        hp1 = tl.load(hp_base + 1).to(tl.float32)
        hp2 = tl.load(hp_base + 2).to(tl.float32)
        hp3 = tl.load(hp_base + 3).to(tl.float32)
        for c_start in range(0, C, BLOCK_C):
            offs_c = c_start + tl.arange(0, BLOCK_C)
            mask_c = offs_c < C
            xb = tl.load(x_ptr + pid_s * C + offs_c, mask=mask_c, other=0.0).to(
                tl.float32
            )
            if HAS_BIAS:
                xb += tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
            orig_base = orig_ptr + pid_s * 4 * C + offs_c
            orig0 = tl.load(orig_base, mask=mask_c, other=0.0).to(tl.float32)
            orig1 = tl.load(orig_base + C, mask=mask_c, other=0.0).to(tl.float32)
            orig2 = tl.load(orig_base + 2 * C, mask=mask_c, other=0.0).to(tl.float32)
            orig3 = tl.load(orig_base + 3 * C, mask=mask_c, other=0.0).to(tl.float32)
            out0 = h00 * orig0 + h10 * orig1 + h20 * orig2 + h30 * orig3 + hp0 * xb
            out1 = h01 * orig0 + h11 * orig1 + h21 * orig2 + h31 * orig3 + hp1 * xb
            out2 = h02 * orig0 + h12 * orig1 + h22 * orig2 + h32 * orig3 + hp2 * xb
            out3 = h03 * orig0 + h13 * orig1 + h23 * orig2 + h33 * orig3 + hp3 * xb
            out_base = out_ptr + pid_s * 4 * C + offs_c
            tl.store(out_base, out0.to(out_ptr.dtype.element_ty), mask=mask_c)
            tl.store(out_base + C, out1.to(out_ptr.dtype.element_ty), mask=mask_c)
            tl.store(out_base + 2 * C, out2.to(out_ptr.dtype.element_ty), mask=mask_c)
            tl.store(out_base + 3 * C, out3.to(out_ptr.dtype.element_ty), mask=mask_c)


def _grid_a_config(sb, channels):
    config = _FIXED_CONFIG.copy()
    config["BLOCK_C"] = min(
        config["BLOCK_C"], max(64, triton.next_power_of_2(channels))
    )
    c_blocks = triton.cdiv(channels, config["BLOCK_C"])
    if c_blocks > _MAX_CORE_DIM:
        raise ValueError("channel grid exceeds the Ascend core dimension limit")
    grid = (triton.cdiv(sb, config["BLOCK_S"]), c_blocks)
    while grid[0] * grid[1] > _MAX_CORE_DIM:
        config["BLOCK_S"] *= 2
        grid = (triton.cdiv(sb, config["BLOCK_S"]), c_blocks)
    return grid, config


def fused_h_post_bda_forward(
    h_res: Tensor,
    original_residual: Tensor,
    h_post: Tensor,
    x: Tensor,
    bias: Optional[Tensor] = None,
) -> Tensor:
    """Compute ``H_res.T @ original_residual + H_post * (x + bias)``."""
    logger.debug("TRAIN_ASCEND FUSED_H_POST_BDA_FORWARD")
    s, b, n, channels = _validate_inputs(h_res, original_residual, h_post, x, bias)
    sb = s * b
    out = torch.empty(s, b, n, channels, dtype=h_res.dtype, device=h_res.device)
    if n <= 16:
        hr_flat = h_res.contiguous()
        orig_flat = original_residual.contiguous()
        hp_flat = h_post.contiguous()
        x_flat = x.contiguous()
        bias_flat = bias.contiguous() if bias is not None else x_flat
        max_block_c = 1024 if h_res.dtype == torch.float32 else _DOT_BLOCK_C
        channel_block = max(64, triton.next_power_of_2(channels))
        dot_block_c = min(max_block_c, channel_block)
        vector_block_c = min(
            2048 if h_res.dtype == torch.float32 else 1024, channel_block
        )
        c_blocks = triton.cdiv(channels, dot_block_c)
        num_tiles = sb * c_blocks
        with _device_context(h_res.device):
            if (
                n == 4
                and channels >= 64
                and h_res.dtype in (torch.bfloat16, torch.float16)
                and (
                    (sb >= 4096 and sb % 8 == 0) or (1024 <= sb < 4096 and sb % 4 == 0)
                )
            ):
                rows = 8 if sb >= 4096 else 4
                _launch_fixed_tiles(
                    _triton_hpb_fwd_vector4_rows_kernel,
                    sb // rows,
                    hr_flat,
                    orig_flat,
                    hp_flat,
                    x_flat,
                    bias_flat,
                    out,
                    0,
                    sb,
                    channels,
                    bias is not None,
                    vector_block_c,
                    rows,
                    offset_index=6,
                    num_warps=4,
                )
            elif n == 4 and channels >= 64 and (c_blocks > 1 or sb >= 512):
                _launch_fixed_tiles(
                    _triton_hpb_fwd_vector4_kernel,
                    sb,
                    hr_flat,
                    orig_flat,
                    hp_flat,
                    x_flat,
                    bias_flat,
                    out,
                    0,
                    sb,
                    channels,
                    bias is not None,
                    vector_block_c,
                    offset_index=6,
                    num_warps=4,
                )
            elif n == 8 and 64 <= channels <= 1024 and sb >= 256:
                _launch_fixed_tiles(
                    _triton_hpb_fwd_vector8_kernel,
                    sb,
                    hr_flat,
                    orig_flat,
                    hp_flat,
                    x_flat,
                    bias_flat,
                    out,
                    0,
                    sb,
                    channels,
                    bias is not None,
                    vector_block_c,
                    offset_index=6,
                    num_warps=4,
                )
            elif n <= 8 and sb <= 8192 and c_blocks > 1:
                _launch_fixed_tiles(
                    _triton_hpb_fwd_row_dot_kernel,
                    sb,
                    hr_flat,
                    orig_flat,
                    hp_flat,
                    x_flat,
                    bias_flat,
                    out,
                    0,
                    sb,
                    channels,
                    n,
                    max(4, n),
                    bias is not None,
                    dot_block_c,
                    c_blocks,
                    offset_index=6,
                    num_warps=4,
                )
            elif n <= 8 and c_blocks > 1:
                default_c_group = 8 if h_res.dtype == torch.float32 else 4
                c_groups = triton.cdiv(c_blocks, default_c_group)
                c_group = triton.cdiv(c_blocks, c_groups)
                _launch_fixed_tiles(
                    _triton_hpb_fwd_grouped_dot_kernel,
                    sb * c_groups,
                    hr_flat,
                    orig_flat,
                    hp_flat,
                    x_flat,
                    bias_flat,
                    out,
                    0,
                    channels,
                    n,
                    max(4, n),
                    bias is not None,
                    dot_block_c,
                    c_blocks,
                    c_group,
                    c_groups,
                    offset_index=6,
                    num_warps=4,
                )
            else:
                _launch_fixed_tiles(
                    _triton_hpb_fwd_dot_kernel,
                    num_tiles,
                    hr_flat,
                    orig_flat,
                    hp_flat,
                    x_flat,
                    bias_flat,
                    out,
                    0,
                    channels,
                    n,
                    max(4, n),
                    bias is not None,
                    dot_block_c,
                    c_blocks,
                    offset_index=6,
                    num_warps=4,
                )
        return out

    hr_flat = h_res.contiguous().view(sb, n, n)
    orig_flat = original_residual.contiguous().view(sb, n, channels)
    hp_flat = h_post.contiguous().view(sb, n)
    x_flat = x.contiguous().view(sb, channels)
    bias_flat = bias.contiguous() if bias is not None else x_flat
    out_flat = out.view(sb, n, channels)
    grid, config = _grid_a_config(sb, channels)
    with _disable_aabs(), _device_context(h_res.device):
        _triton_hpb_fwd_kernel.fn.fn[grid](
            hr_flat,
            orig_flat,
            hp_flat,
            x_flat,
            bias_flat,
            out_flat,
            sb,
            channels,
            n,
            hr_flat.stride(0),
            hr_flat.stride(1),
            hr_flat.stride(2),
            orig_flat.stride(0),
            orig_flat.stride(1),
            orig_flat.stride(2),
            out_flat.stride(0),
            out_flat.stride(1),
            out_flat.stride(2),
            HAS_BIAS=(bias is not None),
            REUSE_INPUT=False,
            **config,
        )
    return out
