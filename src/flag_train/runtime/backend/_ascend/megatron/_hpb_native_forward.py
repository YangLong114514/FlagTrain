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

_MODULE_NAME = "flag_train_hpb_native_forward"
_NATIVE_MODULE = None
_NATIVE_FAILED = False
_NATIVE_LOCK = threading.Lock()
_PLANS = {}

_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <runtime/runtime/rt.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace {

struct DotPlan {
    const void* function;
    int grid;
    std::string name;
    at::Tensor workspace;
};

std::vector<DotPlan> plans;

int register_dot_plan(
    uint64_t function,
    int grid,
    std::string name,
    at::Tensor workspace) {
    plans.push_back(DotPlan{
        reinterpret_cast<const void*>(function),
        grid,
        std::move(name),
        std::move(workspace),
    });
    return static_cast<int>(plans.size() - 1);
}

c10::optional<at::Tensor> forward_dot(
    int plan_id,
    uint64_t stream,
    const at::Tensor& hr,
    const at::Tensor& orig,
    const at::Tensor& hp,
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
        hr.dim() == 4 && hr.size(0) == s && hr.size(1) == b &&
            hr.size(2) == n && hr.size(3) == n,
        "h_res must have shape [s, b, n, n]");
    require(
        hp.dim() == 3 && hp.size(0) == s && hp.size(1) == b && hp.size(2) == n,
        "h_post must have shape [s, b, n]");
    require(
        x.dim() == 3 && x.size(0) == s && x.size(1) == b && x.size(2) == channels,
        "x must have shape [s, b, C]");
    require(
        hr.device() == orig.device() && hp.device() == orig.device() &&
            x.device() == orig.device(),
        "all inputs must be on the same device");
    require(
        hr.scalar_type() == orig.scalar_type() &&
            hp.scalar_type() == orig.scalar_type() &&
            x.scalar_type() == orig.scalar_type(),
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
    require(n > 0 && (n & (n - 1)) == 0, "n must be a positive power of two");

    const DotPlan& plan = plans.at(plan_id);
    at::Tensor hr_c = hr.contiguous();
    at::Tensor orig_c = orig.contiguous();
    at::Tensor hp_c = hp.contiguous();
    at::Tensor x_c = x.contiguous();
    c10::optional<at::Tensor> bias_c;
    if (bias.has_value()) {
        bias_c = bias->contiguous();
    }
    auto aligned = [](const at::Tensor& tensor) {
        return reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0;
    };
    if (!aligned(hr_c) || !aligned(orig_c) || !aligned(hp_c) || !aligned(x_c) ||
        (bias_c.has_value() && !aligned(*bias_c))) {
        return c10::nullopt;
    }

    at::Tensor out = at::empty(orig.sizes(), orig.options());
    const void* bias_ptr = bias_c.has_value() ? bias_c->data_ptr() : x_c.data_ptr();
    const void* function = plan.function;
    const int grid = plan.grid;
    const std::string name = plan.name;
    const void* hr_ptr = hr_c.data_ptr();
    const void* orig_ptr = orig_c.data_ptr();
    const void* hp_ptr = hp_c.data_ptr();
    const void* x_ptr = x_c.data_ptr();
    const void* out_ptr = out.data_ptr();
    const void* workspace = plan.workspace.data_ptr();
    auto launch_call = [=]() -> rtError_t {
        void* ffts_addr = nullptr;
        uint32_t ffts_len = 0;
        rtError_t ret = rtGetC2cCtrlAddr(
            reinterpret_cast<uint64_t*>(&ffts_addr), &ffts_len);
        if (ret != RT_ERROR_NONE) {
            return ret;
        }
        struct __attribute__((packed)) {
            void* ffts_addr __attribute__((aligned(8)));
            void* sync_block_lock __attribute__((aligned(8)));
            void* workspace __attribute__((aligned(8)));
            void* hr __attribute__((aligned(8)));
            void* orig __attribute__((aligned(8)));
            void* hp __attribute__((aligned(8)));
            void* x __attribute__((aligned(8)));
            void* bias __attribute__((aligned(8)));
            void* out __attribute__((aligned(8)));
            int32_t tile_offset __attribute__((aligned(4)));
            int32_t grid_x __attribute__((aligned(4)));
            int32_t grid_y __attribute__((aligned(4)));
            int32_t grid_z __attribute__((aligned(4)));
        } kernel_args = {
            ffts_addr,
            nullptr,
            const_cast<void*>(workspace),
            const_cast<void*>(hr_ptr),
            const_cast<void*>(orig_ptr),
            const_cast<void*>(hp_ptr),
            const_cast<void*>(x_ptr),
            const_cast<void*>(bias_ptr),
            const_cast<void*>(out_ptr),
            0,
            grid,
            1,
            1,
        };
        return rtKernelLaunch(
            function, grid, static_cast<void*>(&kernel_args),
            sizeof(kernel_args), nullptr,
            reinterpret_cast<rtStream_t>(stream));
    };
    at_npu::native::OpCommand cmd;
    cmd.Name(name.c_str()).SetCustomHandler(launch_call).Run();
    return out;
}

}  // namespace

PYBIND11_MODULE(flag_train_hpb_native_forward, m) {
    m.def("register_dot_plan", &register_dot_plan);
    m.def("forward_dot", &forward_dot);
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


def _native_module():
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
                    "Ascend native h_post BDA forward is disabled: %s", error
                )
                return None
    return _NATIVE_MODULE


def _supported_compiled_kernel(compiled):
    try:
        from triton.backends.ascend.utils import (
            _is_auto_map_parallel_blocks_enabled,
            force_disable_ffts,
            get_ascend_arch_from_env,
            is_ffts_supported,
        )
    except ImportError:
        return False

    expected_names = (
        "hr_ptr",
        "orig_ptr",
        "hp_ptr",
        "x_ptr",
        "bias_ptr",
        "out_ptr",
        "tile_offset",
        "C",
        "N",
        "NP",
        "HAS_BIAS",
        "BLOCK_C",
        "C_BLOCKS",
    )
    signature = compiled.src.signature
    if tuple(signature) != expected_names:
        return False
    if any(signature[name][0] != "*" for name in expected_names[:6]):
        return False
    if signature["tile_offset"] != "i32":
        return False
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
    if int(getattr(metadata, "workspace_size", -1)) < 0:
        return False
    if getattr(metadata, "enable_auto_blockify", False):
        return False
    if _is_auto_map_parallel_blocks_enabled():
        return False
    arch = get_ascend_arch_from_env()
    return is_ffts_supported(arch) and not force_disable_ffts()


def get_plan(key):
    return _PLANS.get(key)


def register_plan(key, compiled, grid, tensors, device):
    if key in _PLANS or not _supported_compiled_kernel(compiled):
        return None
    if any(tensor.data_ptr() % 16 for tensor in tensors):
        return None
    native = _native_module()
    if native is None:
        return None
    workspace_size = int(compiled.metadata.workspace_size)
    workspace = torch.empty(
        max(1, workspace_size * grid), dtype=torch.uint8, device=device
    )
    plan_id = native.register_dot_plan(
        compiled.function,
        grid,
        compiled.packed_metadata["kernel_name"],
        workspace,
    )
    plan = (plan_id, native, compiled, workspace)
    _PLANS[key] = plan
    return plan


def launch_plan(plan, stream, h_res, original_residual, h_post, x, bias):
    return plan[1].forward_dot(
        plan[0], stream, h_res, original_residual, h_post, x, bias
    )


def current_stream(device_index):
    torch_npu = __import__("torch_npu")
    raw_stream = getattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait", None)
    if raw_stream is not None:
        return raw_stream(device_index)
    return driver.active.get_current_stream(device_index)


def plan_count():
    return len(_PLANS)
