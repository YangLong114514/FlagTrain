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
"""Ascend C++ fast-launch plans for the fused mHC H_post BDA backward.

Mirrors :mod:`._hpb_native_forward`: registered compiled Triton kernels are
relaunched through a single ``OpCommand`` with ``rtKernelLaunch``, bypassing
the Python-side launch path for the n == 4 vectorized backward and the
small-shape (sb <= 8, C <= 1024, n <= 4) fused backward.
"""

import hashlib
import importlib.util
import logging
import os
import subprocess
import sys
import sysconfig
import tempfile
import threading
from pathlib import Path

import torch
import triton
from triton.runtime.cache import get_cache_manager
from triton.runtime.driver import driver

logger = logging.getLogger(__name__)

_MODULE_NAME = "flag_train_hpb_native_backward"
_NATIVE_MODULE = None
_NATIVE_FAILED = False
_NATIVE_LOCK = threading.Lock()
_PLANS = {}

_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_variable.h>
#include <ATen/EmptyTensor.h>
#include <runtime/runtime/rt.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#include <cstdint>
#include <cstring>
#include <deque>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct PlanKey {
    int64_t shape[4];
    int dtype;
    int device;
    bool has_bias;
    uint64_t stream;

    bool operator==(const PlanKey& other) const {
        return shape[0] == other.shape[0] && shape[1] == other.shape[1] &&
            shape[2] == other.shape[2] && shape[3] == other.shape[3] &&
            dtype == other.dtype && device == other.device &&
            has_bias == other.has_bias && stream == other.stream;
    }
};

struct PlanKeyHash {
    size_t operator()(const PlanKey& key) const {
        size_t hash = key.dtype * 31 + key.device;
        for (int64_t dim : key.shape) {
            hash = hash * 131 + static_cast<size_t>(dim);
        }
        hash = hash * 131 + key.has_bias;
        return hash * 131 + static_cast<size_t>(key.stream);
    }
};

PlanKey make_plan_key(
    const at::Tensor& orig, bool has_bias, uint64_t stream) {
    return PlanKey{
        {orig.size(0), orig.size(1), orig.size(2), orig.size(3)},
        static_cast<int>(orig.scalar_type()),
        orig.get_device(),
        has_bias,
        stream,
    };
}

std::unordered_map<PlanKey, std::pair<int, int>, PlanKeyHash> plan_lookup;
std::unordered_map<int, void*> ffts_addresses;

void* ffts_address(int device) {
    auto found = ffts_addresses.find(device);
    if (found != ffts_addresses.end()) {
        return found->second;
    }
    void* ffts_addr = nullptr;
    uint32_t ffts_len = 0;
    rtError_t ret = rtGetC2cCtrlAddr(
        reinterpret_cast<uint64_t*>(&ffts_addr), &ffts_len);
    TORCH_CHECK(ret == RT_ERROR_NONE, "rtGetC2cCtrlAddr failed: ", ret);
    ffts_addresses[device] = ffts_addr;
    return ffts_addr;
}

at::Tensor empty_output(at::IntArrayRef sizes, at::ScalarType dtype) {
    return at::detail::empty_generic(
        sizes,
        c10_npu::NPUCachingAllocator::get(),
        c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
        dtype,
        c10::MemoryFormat::Contiguous);
}

struct ContiguousTensor {
    const at::Tensor* tensor;
    at::Tensor owned;

    explicit ContiguousTensor(const at::Tensor& input) : tensor(&input) {
        if (!input.is_contiguous()) {
            owned = input.contiguous();
            tensor = &owned;
        }
    }

    const at::Tensor& get() const {
        return *tensor;
    }

    const void* data_ptr() const {
        return tensor->data_ptr();
    }
};

void bind_plan(
    int plan_id,
    int kind,
    const at::Tensor& orig,
    bool has_bias,
    uint64_t stream) {
    plan_lookup[make_plan_key(orig, has_bias, stream)] = {kind, plan_id};
}

std::vector<at::Tensor> make_backward_result(
    at::Tensor&& g_hr,
    at::Tensor&& g_res,
    at::Tensor&& g_hp,
    at::Tensor&& g_x,
    at::Tensor&& g_bias,
    bool has_bias) {
    std::vector<at::Tensor> result;
    result.reserve(has_bias ? 5 : 4);
    result.emplace_back(std::move(g_hr));
    result.emplace_back(std::move(g_res));
    result.emplace_back(std::move(g_hp));
    result.emplace_back(std::move(g_x));
    if (has_bias) {
        result.emplace_back(std::move(g_bias));
    }
    return result;
}

struct BackwardPlan {
    uint64_t func_a;
    int grid_a;
    std::string name_a;
    at::Tensor workspace_a;
    uint64_t func_b;
    int grid_b;
    std::string name_b;
    at::Tensor workspace_b;
    uint64_t func_bias;
    int grid_bias;
    std::string name_bias;
    at::Tensor workspace_bias;
};

std::vector<BackwardPlan> plans;

int register_backward_plan(
    uint64_t func_a,
    int grid_a,
    std::string name_a,
    at::Tensor workspace_a,
    uint64_t func_b,
    int grid_b,
    std::string name_b,
    at::Tensor workspace_b,
    uint64_t func_bias,
    int grid_bias,
    std::string name_bias,
    at::Tensor workspace_bias) {
    plans.push_back(BackwardPlan{
        func_a,
        grid_a,
        std::move(name_a),
        std::move(workspace_a),
        func_b,
        grid_b,
        std::move(name_b),
        std::move(workspace_b),
        func_bias,
        grid_bias,
        std::move(name_bias),
        std::move(workspace_bias),
    });
    return static_cast<int>(plans.size() - 1);
}

std::vector<at::Tensor> backward(
    int plan_id,
    uint64_t stream,
    const at::Tensor& go,
    const at::Tensor& h_res,
    const at::Tensor& orig,
    const at::Tensor& h_post,
    const at::Tensor& x,
    const c10::optional<at::Tensor>& bias) {
    auto require = [](bool condition, const char* message) {
        if (!condition) {
            throw pybind11::value_error(message);
        }
    };
    require(orig.dim() == 4, "original_residual must have shape [s, b, 4, C]");
    const int64_t s = orig.size(0);
    const int64_t b = orig.size(1);
    const int64_t n = orig.size(2);
    const int64_t channels = orig.size(3);
    require(n == 4, "native backward only supports n == 4");
    require(
        go.dim() == 4 && go.size(0) == s && go.size(1) == b &&
            go.size(2) == n && go.size(3) == channels,
        "grad_output must have shape [s, b, 4, C]");
    require(
        h_res.dim() == 4 && h_res.size(0) == s && h_res.size(1) == b &&
            h_res.size(2) == n && h_res.size(3) == n,
        "h_res must have shape [s, b, 4, 4]");
    require(
        h_post.dim() == 3 && h_post.size(0) == s && h_post.size(1) == b &&
            h_post.size(2) == n,
        "h_post must have shape [s, b, 4]");
    require(
        x.dim() == 3 && x.size(0) == s && x.size(1) == b && x.size(2) == channels,
        "x must have shape [s, b, C]");
    require(
        h_res.device() == orig.device() && h_post.device() == orig.device() &&
            x.device() == orig.device() && go.device() == orig.device(),
        "all inputs must be on the same device");
    require(
        h_res.scalar_type() == orig.scalar_type() &&
            h_post.scalar_type() == orig.scalar_type() &&
            x.scalar_type() == orig.scalar_type() &&
            go.scalar_type() == orig.scalar_type(),
        "all inputs must have the same dtype");
    if (bias.has_value()) {
        require(
            bias->dim() == 1 && bias->size(0) == channels,
            "bias must have shape [C]");
        require(
            bias->device() == orig.device() &&
                bias->scalar_type() == orig.scalar_type(),
            "bias must have the same device and dtype as the other inputs");
    }
    require(
        orig.device().type() == at::kPrivateUse1,
        "fused_h_post_bda only supports Ascend tensors");
    require(s * b != 0 && channels != 0, "empty inputs are not supported");

    const BackwardPlan& plan = plans.at(plan_id);
    ContiguousTensor go_c(go);
    ContiguousTensor hr_c(h_res);
    ContiguousTensor orig_c(orig);
    ContiguousTensor hp_c(h_post);
    ContiguousTensor x_c(x);
    ContiguousTensor bias_c(bias.has_value() ? bias.value() : x_c.get());
    auto aligned = [](const void* ptr) {
        return reinterpret_cast<uintptr_t>(ptr) % 16 == 0;
    };
    if (!aligned(go_c.data_ptr()) || !aligned(hr_c.data_ptr()) ||
        !aligned(orig_c.data_ptr()) || !aligned(hp_c.data_ptr()) ||
        !aligned(x_c.data_ptr()) ||
        (bias.has_value() && !aligned(bias_c.data_ptr()))) {
        return {};
    }

    at::Tensor g_hr = empty_output({s, b, n, n}, orig.scalar_type());
    at::Tensor g_res = empty_output({s, b, n, channels}, orig.scalar_type());
    at::Tensor g_hp = empty_output({s, b, n}, orig.scalar_type());
    at::Tensor g_x = empty_output({s, b, channels}, orig.scalar_type());
    at::Tensor g_bias = bias.has_value() ? empty_output({channels}, orig.scalar_type())
                                         : at::Tensor();
    const int64_t sb = s * b;
    const void* bias_ptr = bias.has_value() ? bias_c.data_ptr() : x_c.data_ptr();
    const void* g_bias_ptr = bias.has_value() ? g_bias.data_ptr() : g_x.data_ptr();
    const uint64_t func_a = plan.func_a;
    const int grid_a = plan.grid_a;
    const std::string& name_a = plan.name_a;
    const void* workspace_a = plan.workspace_a.data_ptr();
    const uint64_t func_b = plan.func_b;
    const int grid_b = plan.grid_b;
    const void* workspace_b = plan.workspace_b.data_ptr();
    const uint64_t func_bias = plan.func_bias;
    const int grid_bias = plan.grid_bias;
    const void* workspace_bias = plan.workspace_bias.data_ptr();
    const void* go_ptr = go_c.data_ptr();
    const void* hr_ptr = hr_c.data_ptr();
    const void* orig_ptr = orig_c.data_ptr();
    const void* hp_ptr = hp_c.data_ptr();
    const void* x_ptr = x_c.data_ptr();
    const void* g_hr_ptr = g_hr.data_ptr();
    const void* g_res_ptr = g_res.data_ptr();
    const void* g_hp_ptr = g_hp.data_ptr();
    const void* g_x_ptr = g_x.data_ptr();

    void* ffts_addr = ffts_address(orig.get_device());
    auto launch_call = [=]() -> rtError_t {
        rtError_t ret = RT_ERROR_NONE;
        {
            struct __attribute__((packed)) {
                void* ffts_addr __attribute__((aligned(8)));
                void* sync_block_lock __attribute__((aligned(8)));
                void* workspace __attribute__((aligned(8)));
                void* go __attribute__((aligned(8)));
                void* hr __attribute__((aligned(8)));
                void* hp __attribute__((aligned(8)));
                void* g_res __attribute__((aligned(8)));
                void* g_x __attribute__((aligned(8)));
                void* g_bias __attribute__((aligned(8)));
                int32_t tile_offset __attribute__((aligned(4)));
                int32_t grid_x __attribute__((aligned(4)));
                int32_t grid_y __attribute__((aligned(4)));
                int32_t grid_z __attribute__((aligned(4)));
            } kernel_args = {
                ffts_addr,
                nullptr,
                const_cast<void*>(workspace_a),
                const_cast<void*>(go_ptr),
                const_cast<void*>(hr_ptr),
                const_cast<void*>(hp_ptr),
                const_cast<void*>(g_res_ptr),
                const_cast<void*>(g_x_ptr),
                const_cast<void*>(g_bias_ptr),
                0,
                grid_a,
                1,
                1,
            };
            ret = rtKernelLaunch(
                reinterpret_cast<const void*>(func_a), grid_a,
                static_cast<void*>(&kernel_args), sizeof(kernel_args), nullptr,
                reinterpret_cast<rtStream_t>(stream));
            if (ret != RT_ERROR_NONE) {
                return ret;
            }
        }
        {
            struct __attribute__((packed)) {
                void* ffts_addr __attribute__((aligned(8)));
                void* sync_block_lock __attribute__((aligned(8)));
                void* workspace __attribute__((aligned(8)));
                void* go __attribute__((aligned(8)));
                void* orig __attribute__((aligned(8)));
                void* x __attribute__((aligned(8)));
                void* bias __attribute__((aligned(8)));
                void* g_hr __attribute__((aligned(8)));
                void* g_hp __attribute__((aligned(8)));
                int32_t sb __attribute__((aligned(4)));
                int32_t grid_x __attribute__((aligned(4)));
                int32_t grid_y __attribute__((aligned(4)));
                int32_t grid_z __attribute__((aligned(4)));
            } kernel_args = {
                ffts_addr,
                nullptr,
                const_cast<void*>(workspace_b),
                const_cast<void*>(go_ptr),
                const_cast<void*>(orig_ptr),
                const_cast<void*>(x_ptr),
                const_cast<void*>(bias_ptr),
                const_cast<void*>(g_hr_ptr),
                const_cast<void*>(g_hp_ptr),
                static_cast<int32_t>(sb),
                grid_b,
                1,
                1,
            };
            ret = rtKernelLaunch(
                reinterpret_cast<const void*>(func_b), grid_b,
                static_cast<void*>(&kernel_args), sizeof(kernel_args), nullptr,
                reinterpret_cast<rtStream_t>(stream));
            if (ret != RT_ERROR_NONE) {
                return ret;
            }
        }
        if (func_bias != 0) {
            struct __attribute__((packed)) {
                void* ffts_addr __attribute__((aligned(8)));
                void* sync_block_lock __attribute__((aligned(8)));
                void* workspace __attribute__((aligned(8)));
                void* g_x __attribute__((aligned(8)));
                void* g_bias __attribute__((aligned(8)));
                int32_t sb __attribute__((aligned(4)));
                int32_t grid_x __attribute__((aligned(4)));
                int32_t grid_y __attribute__((aligned(4)));
                int32_t grid_z __attribute__((aligned(4)));
            } kernel_args = {
                ffts_addr,
                nullptr,
                const_cast<void*>(workspace_bias),
                const_cast<void*>(g_x_ptr),
                const_cast<void*>(g_bias_ptr),
                static_cast<int32_t>(sb),
                grid_bias,
                1,
                1,
            };
            ret = rtKernelLaunch(
                reinterpret_cast<const void*>(func_bias), grid_bias,
                static_cast<void*>(&kernel_args), sizeof(kernel_args), nullptr,
                reinterpret_cast<rtStream_t>(stream));
            if (ret != RT_ERROR_NONE) {
                return ret;
            }
        }
        return RT_ERROR_NONE;
    };
    at_npu::native::OpCommand cmd;
    cmd.Name(name_a.c_str()).SetCustomHandler(launch_call).Run();
    return make_backward_result(
        std::move(g_hr),
        std::move(g_res),
        std::move(g_hp),
        std::move(g_x),
        std::move(g_bias),
        bias.has_value());
}

struct SmallPlan {
    uint64_t func_small;
    int grid_small;
    std::string name_small;
    at::Tensor workspace_small;
    bool small_writes_bias;
    std::vector<int64_t> ints_small;
    uint64_t func_bias;
    int grid_bias;
    std::string name_bias;
    at::Tensor workspace_bias;
    std::vector<int64_t> ints_bias;
};

std::deque<SmallPlan> small_plans;

int register_small_plan(
    uint64_t func_small,
    int grid_small,
    std::string name_small,
    at::Tensor workspace_small,
    bool small_writes_bias,
    std::vector<int64_t> ints_small,
    uint64_t func_bias,
    int grid_bias,
    std::string name_bias,
    at::Tensor workspace_bias,
    std::vector<int64_t> ints_bias) {
    small_plans.push_back(SmallPlan{
        func_small,
        grid_small,
        std::move(name_small),
        std::move(workspace_small),
        small_writes_bias,
        std::move(ints_small),
        func_bias,
        grid_bias,
        std::move(name_bias),
        std::move(workspace_bias),
        std::move(ints_bias),
    });
    return static_cast<int>(small_plans.size() - 1);
}

rtError_t launch_packed(
    uint64_t function,
    int grid,
    const void* workspace,
    void* ffts_addr,
    const void* const* ptrs,
    size_t ptr_count,
    const std::vector<int64_t>& ints,
    uint64_t stream) {
    alignas(8) uint8_t args[256];
    size_t args_size = 24 + ptr_count * 8 + ints.size() * 4 + 12;
    if (args_size > sizeof(args)) {
        return -1;
    }
    size_t offset = 0;
    auto put_ptr = [&](const void* value) {
        std::memcpy(args + offset, &value, sizeof(value));
        offset += sizeof(value);
    };
    put_ptr(ffts_addr);
    put_ptr(nullptr);
    put_ptr(workspace);
    for (size_t i = 0; i < ptr_count; ++i) {
        put_ptr(ptrs[i]);
    }
    for (int64_t value : ints) {
        int32_t v = static_cast<int32_t>(value);
        std::memcpy(args + offset, &v, sizeof(v));
        offset += sizeof(v);
    }
    int32_t dims[3] = {grid, 1, 1};
    std::memcpy(args + offset, dims, sizeof(dims));
    return rtKernelLaunch(
        reinterpret_cast<const void*>(function), grid,
        static_cast<void*>(args), args_size, nullptr,
        reinterpret_cast<rtStream_t>(stream));
}

std::vector<at::Tensor> backward_small(
    int plan_id,
    uint64_t stream,
    const at::Tensor& go,
    const at::Tensor& h_res,
    const at::Tensor& orig,
    const at::Tensor& h_post,
    const at::Tensor& x,
    const c10::optional<at::Tensor>& bias) {
    auto require = [](bool condition, const char* message) {
        if (!condition) {
            throw pybind11::value_error(message);
        }
    };
    require(orig.dim() == 4, "original_residual must have shape [s, b, n, C]");
    const int64_t s = orig.size(0);
    const int64_t b = orig.size(1);
    const int64_t n = orig.size(2);
    const int64_t channels = orig.size(3);
    require(
        go.dim() == 4 && go.size(0) == s && go.size(1) == b &&
            go.size(2) == n && go.size(3) == channels,
        "grad_output must have shape [s, b, n, C]");
    require(
        h_res.dim() == 4 && h_res.size(0) == s && h_res.size(1) == b &&
            h_res.size(2) == n && h_res.size(3) == n,
        "h_res must have shape [s, b, n, n]");
    require(
        h_post.dim() == 3 && h_post.size(0) == s && h_post.size(1) == b &&
            h_post.size(2) == n,
        "h_post must have shape [s, b, n]");
    require(
        x.dim() == 3 && x.size(0) == s && x.size(1) == b && x.size(2) == channels,
        "x must have shape [s, b, C]");
    require(
        h_res.device() == orig.device() && h_post.device() == orig.device() &&
            x.device() == orig.device() && go.device() == orig.device(),
        "all inputs must be on the same device");
    require(
        h_res.scalar_type() == orig.scalar_type() &&
            h_post.scalar_type() == orig.scalar_type() &&
            x.scalar_type() == orig.scalar_type() &&
            go.scalar_type() == orig.scalar_type(),
        "all inputs must have the same dtype");
    if (bias.has_value()) {
        require(
            bias->dim() == 1 && bias->size(0) == channels,
            "bias must have shape [C]");
        require(
            bias->device() == orig.device() &&
                bias->scalar_type() == orig.scalar_type(),
            "bias must have the same device and dtype as the other inputs");
    }
    require(
        orig.device().type() == at::kPrivateUse1,
        "fused_h_post_bda only supports Ascend tensors");
    require(s * b != 0 && channels != 0, "empty inputs are not supported");

    const SmallPlan& plan = small_plans.at(plan_id);
    ContiguousTensor go_c(go);
    ContiguousTensor hr_c(h_res);
    ContiguousTensor orig_c(orig);
    ContiguousTensor hp_c(h_post);
    ContiguousTensor x_c(x);
    ContiguousTensor bias_c(bias.has_value() ? bias.value() : x_c.get());
    auto aligned = [](const void* ptr) {
        return reinterpret_cast<uintptr_t>(ptr) % 16 == 0;
    };
    if (!aligned(go_c.data_ptr()) || !aligned(hr_c.data_ptr()) ||
        !aligned(orig_c.data_ptr()) || !aligned(hp_c.data_ptr()) ||
        !aligned(x_c.data_ptr()) ||
        (bias.has_value() && !aligned(bias_c.data_ptr()))) {
        return {};
    }

    at::Tensor g_hr = empty_output({s, b, n, n}, orig.scalar_type());
    at::Tensor g_res = empty_output({s, b, n, channels}, orig.scalar_type());
    at::Tensor g_hp = empty_output({s, b, n}, orig.scalar_type());
    at::Tensor g_x = empty_output({s, b, channels}, orig.scalar_type());
    at::Tensor g_bias = bias.has_value() ? empty_output({channels}, orig.scalar_type())
                                         : at::Tensor();
    const int64_t sb = s * b;
    const void* bias_ptr = bias.has_value() ? bias_c.data_ptr() : x_c.data_ptr();
    const void* g_bias_ptr = bias.has_value() ? g_bias.data_ptr() : g_x.data_ptr();
    const uint64_t func_small = plan.func_small;
    const int grid_small = plan.grid_small;
    const std::string& name_small = plan.name_small;
    const void* workspace_small = plan.workspace_small.data_ptr();
    const bool small_writes_bias = plan.small_writes_bias;
    const std::vector<int64_t>* ints_small = &plan.ints_small;
    const uint64_t func_bias = plan.func_bias;
    const int grid_bias = plan.grid_bias;
    const void* workspace_bias = plan.workspace_bias.data_ptr();
    const std::vector<int64_t>* ints_bias = &plan.ints_bias;
    const void* go_ptr = go_c.data_ptr();
    const void* hr_ptr = hr_c.data_ptr();
    const void* orig_ptr = orig_c.data_ptr();
    const void* hp_ptr = hp_c.data_ptr();
    const void* x_ptr = x_c.data_ptr();
    const void* g_hr_ptr = g_hr.data_ptr();
    const void* g_res_ptr = g_res.data_ptr();
    const void* g_hp_ptr = g_hp.data_ptr();
    const void* g_x_ptr = g_x.data_ptr();

    void* ffts_addr = ffts_address(orig.get_device());
    auto launch_call = [=]() -> rtError_t {
        rtError_t ret = RT_ERROR_NONE;
        {
            const void* ptrs[11] = {
                go_ptr, hr_ptr, orig_ptr,
                hp_ptr, x_ptr, bias_ptr,
                g_hr_ptr, g_res_ptr, g_hp_ptr,
                g_x_ptr, const_cast<void*>(g_bias_ptr),
            };
            size_t ptr_count = small_writes_bias ? 11 : 10;
            ret = launch_packed(
                func_small, grid_small, workspace_small, ffts_addr, ptrs,
                ptr_count, *ints_small, stream);
            if (ret != RT_ERROR_NONE) {
                return ret;
            }
        }
        if (func_bias != 0) {
            const void* ptrs[2] = {
                g_x_ptr, const_cast<void*>(g_bias_ptr),
            };
            ret = launch_packed(
                func_bias, grid_bias, workspace_bias,
                ffts_addr, ptrs, 2, *ints_bias, stream);
            if (ret != RT_ERROR_NONE) {
                return ret;
            }
        }
        return RT_ERROR_NONE;
    };
    at_npu::native::OpCommand cmd;
    cmd.Name(name_small.c_str()).SetCustomHandler(launch_call).Run();
    return make_backward_result(
        std::move(g_hr),
        std::move(g_res),
        std::move(g_hp),
        std::move(g_x),
        std::move(g_bias),
        bias.has_value());
}

std::vector<at::Tensor> backward_cached(
    const at::Tensor& go,
    const at::Tensor& h_res,
    const at::Tensor& orig,
    const at::Tensor& h_post,
    const at::Tensor& x,
    const c10::optional<at::Tensor>& bias) {
    if (orig.dim() != 4 || orig.device().type() != at::kPrivateUse1) {
        return {};
    }
    c10_npu::NPUGuard guard(orig.device());
    uint64_t stream = reinterpret_cast<uint64_t>(
        c10_npu::getCurrentNPUStream(orig.get_device()).stream());
    auto found = plan_lookup.find(make_plan_key(orig, bias.has_value(), stream));
    if (found == plan_lookup.end()) {
        return {};
    }
    int kind = found->second.first;
    int plan_id = found->second.second;
    if (kind == 0) {
        return backward(plan_id, stream, go, h_res, orig, h_post, x, bias);
    }
    return backward_small(plan_id, stream, go, h_res, orig, h_post, x, bias);
}

PyObject* backward_cached_fast(
    PyObject*, PyObject* const* args, Py_ssize_t nargs) {
    HANDLE_TH_ERRORS
    if (nargs != 6) {
        PyErr_SetString(PyExc_TypeError, "backward expects six arguments");
        return nullptr;
    }
    for (int i = 0; i < 5; ++i) {
        if (!THPVariable_Check(args[i])) {
            PyErr_SetString(PyExc_TypeError, "backward inputs must be tensors");
            return nullptr;
        }
    }
    if (args[5] != Py_None && !THPVariable_Check(args[5])) {
        PyErr_SetString(PyExc_TypeError, "bias must be a tensor or None");
        return nullptr;
    }
    c10::optional<at::Tensor> bias;
    if (args[5] != Py_None) {
        bias = THPVariable_Unpack(args[5]);
    }
    std::vector<at::Tensor> out;
    try {
        out = backward_cached(
            THPVariable_Unpack(args[0]),
            THPVariable_Unpack(args[1]),
            THPVariable_Unpack(args[2]),
            THPVariable_Unpack(args[3]),
            THPVariable_Unpack(args[4]),
            bias);
    } catch (const py::value_error& error) {
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
    if (out.empty()) {
        Py_RETURN_NONE;
    }
    PyObject* result = PyTuple_New(5);
    if (result == nullptr) {
        return nullptr;
    }
    for (int i = 0; i < 4; ++i) {
        PyTuple_SET_ITEM(result, i, THPVariable_Wrap(std::move(out[i])));
    }
    if (out.size() == 5) {
        PyTuple_SET_ITEM(result, 4, THPVariable_Wrap(std::move(out[4])));
    } else {
        Py_INCREF(Py_None);
        PyTuple_SET_ITEM(result, 4, Py_None);
    }
    return result;
    END_HANDLE_TH_ERRORS
}

PyMethodDef fast_methods[] = {
    {"backward_cached_fast",
     (PyCFunction)(void (*)(void))backward_cached_fast,
     METH_FASTCALL,
     nullptr},
    {nullptr, nullptr, 0, nullptr},
};

}  // namespace

PYBIND11_MODULE(flag_train_hpb_native_backward, m) {
    m.def("register_backward_plan", &register_backward_plan);
    m.def("backward", &backward);
    m.def("register_small_plan", &register_small_plan);
    m.def("backward_small", &backward_small);
    m.def("backward_cached", &backward_cached);
    m.def("bind_plan", &bind_plan);
    if (PyModule_AddFunctions(m.ptr(), fast_methods) != 0) {
        throw py::error_already_set();
    }
}
"""


def _build_native_module():
    from triton.backends.ascend.utils import _get_ascend_path

    torch_path = Path(torch.__file__).resolve().parent
    torch_npu = __import__("torch_npu")
    torch_npu_path = Path(torch_npu.__file__).resolve().parent
    ascend_path = Path(_get_ascend_path())
    abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
    key = hashlib.sha256(
        "|".join(
            (
                _CPP_SOURCE,
                sys.version,
                torch.__version__,
                triton.__version__,
                str(ascend_path),
                str(abi),
            )
        ).encode()
    ).hexdigest()
    cache = get_cache_manager(key)
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    so_name = f"{_MODULE_NAME}{suffix}"
    cache_path = cache.get_file(so_name)
    if cache_path is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = Path(tmpdir) / f"{_MODULE_NAME}.cpp"
            so_path = Path(tmpdir) / so_name
            src_path.write_text(_CPP_SOURCE)
            command = [
                os.environ.get("CXX", "c++"),
                str(src_path),
                f"-I{sysconfig.get_paths()['include']}",
                f"-I{torch_path / 'include'}",
                f"-I{torch_path / 'include/torch/csrc/api/include'}",
                f"-I{torch_npu_path / 'include'}",
                f"-I{ascend_path / 'include'}",
                f"-I{ascend_path / 'include/experiment'}",
                f"-I{ascend_path / 'pkg_inc'}",
                f"-I{ascend_path / 'pkg_inc/profiling'}",
                f"-L{ascend_path / 'lib64'}",
                f"-L{torch_npu_path / 'lib'}",
                f"-L{torch_path / 'lib'}",
                "-lruntime",
                "-lascendcl",
                "-ltorch_npu",
                "-ltorch",
                "-ltorch_cpu",
                "-lc10",
                "-ltorch_python",
                f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
                "-std=c++17",
                "-shared",
                "-fPIC",
                "-o",
                str(so_path),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(result.stderr[-4000:])
            cache_path = cache.put(so_path.read_bytes(), so_name, binary=True)
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, cache_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native_module():
    global _NATIVE_MODULE, _NATIVE_FAILED
    if _NATIVE_MODULE is not None:
        return _NATIVE_MODULE
    if _NATIVE_FAILED:
        return None
    with _NATIVE_LOCK:
        if _NATIVE_MODULE is None and not _NATIVE_FAILED:
            try:
                module = sys.modules.get(_MODULE_NAME)
                if module is None:
                    module = _build_native_module()
                    sys.modules[_MODULE_NAME] = module
                _NATIVE_MODULE = module
            except Exception as error:
                _NATIVE_FAILED = True
                logger.warning(
                    "Ascend native h_post BDA backward is disabled: %s", error
                )
                return None
    return _NATIVE_MODULE


def _metadata_supported(compiled):
    metadata = compiled.metadata
    if getattr(metadata, "force_simt_only", True):
        return False
    if getattr(metadata, "compile_on_910_95", True):
        return False
    if getattr(metadata, "debug", True) or getattr(
        metadata, "debug_launch_hidden_arg", False
    ):
        return False
    if int(getattr(metadata, "lock_num", -1) or 0) > 0:
        return False
    if getattr(metadata, "enable_auto_blockify", False):
        return False
    return True


def _ffts_supported():
    try:
        from triton.backends.ascend.utils import (
            _is_auto_map_parallel_blocks_enabled,
            force_disable_ffts,
            get_ascend_arch_from_env,
            is_ffts_supported,
        )
    except ImportError:
        return False
    if _is_auto_map_parallel_blocks_enabled():
        return False
    arch = get_ascend_arch_from_env()
    return is_ffts_supported(arch) and not force_disable_ffts()


def check_a_kernel(compiled):
    expected = (
        "go_ptr",
        "hr_ptr",
        "hp_ptr",
        "g_res_ptr",
        "g_x_ptr",
        "g_bias_ptr",
        "tile_offset",
        "C",
        "HAS_BIAS",
        "BLOCK_C",
    )
    signature = compiled.src.signature
    if tuple(signature) != expected:
        return False
    if any(signature[name][0] != "*" for name in expected[:6]):
        return False
    if signature["tile_offset"] != "i32":
        return False
    return _metadata_supported(compiled) and _ffts_supported()


def check_b_kernel(compiled):
    expected = (
        "go_ptr",
        "orig_ptr",
        "x_ptr",
        "bias_ptr",
        "g_hr_ptr",
        "g_hp_ptr",
        "sb",
        "C",
        "HAS_BIAS",
        "BLOCK_C",
    )
    signature = compiled.src.signature
    if tuple(signature) != expected:
        return False
    if any(signature[name][0] != "*" for name in expected[:6]):
        return False
    if signature["sb"] != "i32":
        return False
    return _metadata_supported(compiled) and _ffts_supported()


def check_bias_kernel(compiled):
    expected = ("g_x_ptr", "bias_ptr", "sb", "C", "BLOCK_C", "BLOCK_S")
    signature = compiled.src.signature
    if tuple(signature) != expected:
        return False
    if any(signature[name][0] != "*" for name in expected[:2]):
        return False
    if signature["sb"] != "i32":
        return False
    return _metadata_supported(compiled) and _ffts_supported()


def get_plan(key):
    return _PLANS.get(key)


def register_plan(
    key,
    compiled_a,
    grid_a,
    compiled_b,
    grid_b,
    compiled_bias,
    grid_bias,
    tensors,
    device,
):
    if key in _PLANS:
        return None
    has_bias = key[3]
    if not check_a_kernel(compiled_a) or not check_b_kernel(compiled_b):
        return None
    if has_bias and not check_bias_kernel(compiled_bias):
        return None
    if any(tensor.data_ptr() % 16 for tensor in tensors):
        return None
    native = native_module()
    if native is None:
        return None
    try:
        compiled_a._init_handles()
        compiled_b._init_handles()
        if has_bias:
            compiled_bias._init_handles()
    except Exception:
        return None
    workspace_a = torch.empty(16, dtype=torch.uint8, device=device)
    workspace_b = torch.empty(16, dtype=torch.uint8, device=device)
    if has_bias:
        workspace_bias = torch.empty(16, dtype=torch.uint8, device=device)
        plan_id = native.register_backward_plan(
            compiled_a.function,
            grid_a,
            compiled_a.packed_metadata["kernel_name"],
            workspace_a,
            compiled_b.function,
            grid_b,
            compiled_b.packed_metadata["kernel_name"],
            workspace_b,
            compiled_bias.function,
            grid_bias,
            compiled_bias.packed_metadata["kernel_name"],
            workspace_bias,
        )
        plan = (plan_id, native, (compiled_a, compiled_b, compiled_bias))
    else:
        plan_id = native.register_backward_plan(
            compiled_a.function,
            grid_a,
            compiled_a.packed_metadata["kernel_name"],
            workspace_a,
            compiled_b.function,
            grid_b,
            compiled_b.packed_metadata["kernel_name"],
            workspace_b,
            0,
            0,
            "",
            workspace_a,
        )
        plan = (plan_id, native, (compiled_a, compiled_b))
    native.bind_plan(plan_id, 0, tensors[2], has_bias, key[4])
    _PLANS[key] = plan
    return plan


def launch_plan(plan, stream, go, h_res, orig, h_post, x, bias):
    out = plan[1].backward(plan[0], stream, go, h_res, orig, h_post, x, bias)
    if not out:
        return None
    if len(out) == 5:
        return tuple(out)
    return (*out, None)


def launch_cached(go, h_res, orig, h_post, x, bias):
    native = _NATIVE_MODULE
    if native is None or not _PLANS:
        return None
    return native.backward_cached_fast(go, h_res, orig, h_post, x, bias)


def check_small_kernel(compiled, small_writes_bias):
    expected = [
        "go_ptr",
        "hr_ptr",
        "orig_ptr",
        "hp_ptr",
        "x_ptr",
        "bias_ptr",
        "g_hr_ptr",
        "g_res_ptr",
        "g_hp_ptr",
        "g_x_ptr",
    ]
    if small_writes_bias:
        expected.append("g_bias_ptr")
    expected += [
        "sb",
        "C",
        "N",
        "stride_go_s",
        "stride_go_n",
        "stride_go_c",
        "stride_hr_s",
        "stride_hr_i",
        "stride_hr_j",
        "stride_orig_s",
        "stride_orig_n",
        "stride_orig_c",
    ]
    if small_writes_bias:
        expected += ["BLOCK_C", "BLOCK_S", "SUM_BIAS"]
    else:
        expected += ["HAS_BIAS", "BLOCK_C", "BLOCK_S"]
    expected = tuple(expected)
    signature = compiled.src.signature
    if tuple(signature) != expected:
        return False
    ptr_count = 11 if small_writes_bias else 10
    if any(signature[name][0] != "*" for name in expected[:ptr_count]):
        return False
    if any(
        signature[name] not in ("i32", "constexpr") for name in expected[ptr_count:]
    ):
        return False
    return _metadata_supported(compiled) and _ffts_supported()


def _runtime_int_args(compiled, values):
    args = []
    for name, kind in compiled.src.signature.items():
        if kind == "i32":
            args.append(int(values[name]))
        elif kind == "constexpr" or kind.startswith("*"):
            continue
        else:
            return None
    return args


def register_small_plan(
    key,
    compiled_small,
    grid_small,
    small_writes_bias,
    int_args_small,
    compiled_bias,
    grid_bias,
    int_args_bias,
    tensors,
    device,
):
    if key in _PLANS:
        return None
    if not check_small_kernel(compiled_small, small_writes_bias):
        return None
    if compiled_bias is not None and not check_bias_kernel(compiled_bias):
        return None
    ints_small = _runtime_int_args(compiled_small, int_args_small)
    if ints_small is None:
        return None
    ints_bias = []
    if compiled_bias is not None:
        ints_bias = _runtime_int_args(compiled_bias, int_args_bias)
        if ints_bias is None:
            return None
    if any(tensor.data_ptr() % 16 for tensor in tensors):
        return None
    native = native_module()
    if native is None:
        return None
    try:
        compiled_small._init_handles()
        if compiled_bias is not None:
            compiled_bias._init_handles()
    except Exception:
        return None
    workspace_small = torch.empty(16, dtype=torch.uint8, device=device)
    workspace_bias = torch.empty(16, dtype=torch.uint8, device=device)
    if compiled_bias is not None:
        plan_id = native.register_small_plan(
            compiled_small.function,
            grid_small,
            compiled_small.packed_metadata["kernel_name"],
            workspace_small,
            small_writes_bias,
            ints_small,
            compiled_bias.function,
            grid_bias,
            compiled_bias.packed_metadata["kernel_name"],
            workspace_bias,
            ints_bias,
        )
        plan = (plan_id, native, (compiled_small, compiled_bias), True)
    else:
        plan_id = native.register_small_plan(
            compiled_small.function,
            grid_small,
            compiled_small.packed_metadata["kernel_name"],
            workspace_small,
            small_writes_bias,
            ints_small,
            0,
            0,
            "",
            workspace_bias,
            [],
        )
        plan = (plan_id, native, (compiled_small,), True)
    native.bind_plan(plan_id, 1, tensors[2], key[3], key[4])
    _PLANS[key] = plan
    return plan


def launch_small_plan(plan, stream, go, h_res, orig, h_post, x, bias):
    out = plan[1].backward_small(plan[0], stream, go, h_res, orig, h_post, x, bias)
    if not out:
        return None
    if len(out) == 5:
        return tuple(out)
    return (*out, None)


def current_stream(device_index):
    torch_npu = __import__("torch_npu")
    raw_stream = getattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait", None)
    if raw_stream is not None:
        return raw_stream(device_index)
    return driver.active.get_current_stream(device_index)
