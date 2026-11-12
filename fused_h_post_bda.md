# `fused_h_post_bda` 算子说明

## 1. 概述

`fused_h_post_bda` 是 FlagTrain 为 Megatron mHC（Manifold-Constrained Hyper-Connections）路径提供的 Triton 融合算子，包含两个独立接口：

- `fused_h_post_bda_forward`：完成 mHC 的后置残差混合、层输出缩放和 bias 加法。
- `fused_h_post_bda_backward`：计算上述前向计算对全部输入的梯度。

实现参考 NVIDIA Megatron-LM `main` 分支提交
`2052769557dcf7045c4091d1996f0bc59afeb80e`，对应 Megatron Core
`0.21.0+2052769`。当前实现位于：

- `src/flag_train/megatron/fused_h_post_bda.py`
- `src/flag_train/megatron/__init__.py`

当前分支为 `fused_h_post_bda`。本文描述截至代码提交 `807a1ae` 的实现和验证结果。

## 2. 数学功能

### 2.1 前向计算

设：

- `s`：序列长度维度的分段数。
- `b`：batch 大小。
- `n`：mHC 的流数量。
- `C`：隐藏通道数。
- `sb = s * b`：合并后的序列-batch 维度。

输入定义如下：

| 输入 | 形状 | 含义 |
| --- | --- | --- |
| `h_res` | `[s,b,n,n]` | 残差混合矩阵，记为 `H_res` |
| `original_residual` | `[s,b,n,C]` | 原始多流残差，记为 `R` |
| `h_post` | `[s,b,n]` | 层输出缩放系数，记为 `H_post` |
| `x` | `[s,b,C]` | 当前层输出 |
| `bias` | `[C]` 或 `None` | 可选输出 bias |

输出形状为 `[s,b,n,C]`，计算式为：

```text
output = H_res.T @ original_residual + H_post * (x + bias)
```

展开到每个 `sb` 和输出流 `i`：

```text
output[sb,i,c] =
    sum_j(H_res[sb,j,i] * original_residual[sb,j,c])
    + H_post[sb,i] * (x[sb,c] + bias[c])
```

当 `bias is None` 时，不执行 bias 加法。

这里特别需要注意：`H_res` 在矩阵乘中使用的是转置，不能写成
`H_res @ original_residual`。

### 2.2 反向计算

令上游梯度为 `G = grad_output`，形状与 `output` 相同，则：

```text
grad_h_res            = original_residual @ G.T
grad_original_residual = H_res @ G
grad_x                 = sum_n(H_post * G)
grad_h_post            = sum_C(G * (x + bias))
grad_bias              = sum_{s,b}(grad_x)
```

反向返回一个五元组，顺序固定为：

```python
(
    grad_h_res,
    grad_original_residual,
    grad_h_post,
    grad_x,
    grad_bias,
)
```

当 `bias is None` 时，最后一项为 `None`。

## 3. 接口说明

### 3.1 `fused_h_post_bda_forward`

```python
fused_h_post_bda_forward(
    h_res: torch.Tensor,
    original_residual: torch.Tensor,
    h_post: torch.Tensor,
    x: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor
```

返回值：

- 形状：`[s,b,n,C]`。
- dtype：与 `h_res` 相同。
- 设备：与输入相同。

### 3.2 `fused_h_post_bda_backward`

```python
fused_h_post_bda_backward(
    grad_output: torch.Tensor,
    h_res: torch.Tensor,
    original_residual: torch.Tensor,
    h_post: torch.Tensor,
    x: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
]
```

`grad_output` 必须具有 `[s,b,n,C]` 形状，并与其他输入使用相同 dtype 和设备。

返回值形状：

| 返回值 | 形状 |
| --- | --- |
| `grad_h_res` | `[s,b,n,n]` |
| `grad_original_residual` | `[s,b,n,C]` |
| `grad_h_post` | `[s,b,n]` |
| `grad_x` | `[s,b,C]` |
| `grad_bias` | `[C]`；无 bias 时为 `None` |

### 3.3 输入约束

当前 Python 包装层会检查以下条件：

1. `original_residual` 必须为四维 `[s,b,n,C]`。
2. 其他输入必须满足第 2 节列出的形状关系。
3. 所有输入必须位于同一设备，并使用相同 dtype。
4. `bias` 必须是一维 `[C]`，并与输入使用相同 dtype 和设备。
5. `s*b` 和 `C` 不能为 0。
6. `n` 必须是正的 2 的幂，因为 Kernel 使用 `tl.arange(0,N)` 和
   `tl.arange(0,N*N)`。
7. 当前算子只接受 FlagTrain 当前运行时设备上的张量，不接受 CPU 张量。

已验证 dtype：

- `torch.bfloat16`
- `torch.float16`
- `torch.float32`

支持非连续输入。包装层会先对需要按连续地址读取的输入执行 `.contiguous()`，
包括 `bias` 和 `grad_output`。

这两个接口都是**直接算子接口**，不是完整的 `torch.autograd.Function` 公共封装。
调用方需要自行保存前向输入并在反向阶段显式调用 backward。

### 3.4 使用示例

```python
import torch
from flag_train.megatron import (
    fused_h_post_bda_backward,
    fused_h_post_bda_forward,
)

s, b, n, c = 2, 4, 4, 1024
device = "cuda"
dtype = torch.bfloat16

h_res = torch.randn(s, b, n, n, device=device, dtype=dtype)
original_residual = torch.randn(s, b, n, c, device=device, dtype=dtype)
h_post = torch.randn(s, b, n, device=device, dtype=dtype)
x = torch.randn(s, b, c, device=device, dtype=dtype)
bias = torch.randn(c, device=device, dtype=dtype)
grad_output = torch.randn(s, b, n, c, device=device, dtype=dtype)

output = fused_h_post_bda_forward(
    h_res,
    original_residual,
    h_post,
    x,
    bias,
)

grads = fused_h_post_bda_backward(
    grad_output,
    h_res,
    original_residual,
    h_post,
    x,
    bias,
)

grad_h_res, grad_residual, grad_h_post, grad_x, grad_bias = grads
```

无 bias 时：

```python
output = fused_h_post_bda_forward(h_res, original_residual, h_post, x, None)
grads = fused_h_post_bda_backward(grad_output, h_res, original_residual, h_post, x, None)
assert grads[-1] is None
```

## 4. NVIDIA 平台实现方案

### 4.1 Kernel 划分

当前实现按形状选择不同的 Kernel 组合：

| 部分 | 实现 | 主要工作 |
| --- | --- | --- |
| Forward | `_triton_hpb_fwd_kernel` | 计算 `H_res.T @ R + H_post * (x + bias)` |
| Backward A | `_triton_hpb_bwd_g_x_orig_kernel` | 通用路径计算 `grad_x` 和 `grad_original_residual` |
| Backward B | `_triton_hpb_bwd_g_hp_hr_kernel` | 单阶段计算 `grad_h_post` 和 `grad_h_res` |
| 小形状 backward | `_triton_hpb_bwd_small_kernel` | 单 Kernel 计算除 bias 外的四类梯度 |
| Split-C backward B | `_triton_hpb_bwd_g_hp_hr_split_c_kernel` | 沿 C 分片生成 FP32 partial 梯度 |
| Split-C merge | `_triton_hpb_bwd_g_hp_hr_merge_kernel` | 合并 partial 并写出 `grad_h_post`、`grad_h_res` |
| Bias 归约 | `g_x.sum(dim=0).to(bias.dtype)` | 跨全部 `sb` 计算 `grad_bias` |

Grid 组织方式：

- Forward：二维 grid，分别覆盖 `sb` 和 `C`。
- Backward A：二维 grid，覆盖 `sb` 和 `C`。
- 单阶段 Backward B：一维 grid，覆盖 `sb`，在 Kernel 内沿 `C` 分块归约。
- 小形状 backward：一维 grid 覆盖 `sb`，在一个 Kernel 中遍历 C tile 并完成四类梯度。
- Split-C backward B：二维 grid 覆盖 `sb` 和 4 个 C 分片；merge 再沿 `sb` 合并。

普通 backward 使用 A、B 两个 Kernel；小形状融合路径只使用一个计算 Kernel；N=8
split-C 路径使用 A、split-C、merge 三个计算 Kernel。有 bias 时，各路径都额外执行
Torch reduction。计算累加和 split-C partial 都使用 FP32，最终写回各输出对应的输入 dtype。

### 4.2 通用 autotune 配置

主要通用 Kernel 的搜索空间为：

- `BLOCK_C ∈ {64,128,256,512}`
- `BLOCK_S ∈ {1,2,4,8}`
- `num_warps ∈ {2,4,8}`

合计 48 组配置。未启用输入复用时仍使用完整搜索空间。Forward 和 backward A 的
autotune key 为 `C`、`N` 和 `REUSE_INPUT`，Triton 还会把输入 dtype 纳入缓存键；
单阶段 backward B 的 key 仍为 `C` 和 `N`。`REUSE_INPUT` 可以隔离两种计算结构的调优
结果，避免原分支的 block 配置被复用分支错误继承。

昇腾不进入通用 Autotuner，也不在实现中调用 Torch 原生计算算子。普通形状直接启动通用原始
JITFunction：forward/backward A 使用 `BLOCK_S=1、num_warps=4` 和按 C 裁剪的通道块，B
固定 `(64,1,4)`；`sb<=8、C<=1024、N<=4` 的 backward 使用通用单 kernel 融合 A/B。
`N<=16` 的 forward 全部使用 Ascend Triton dot，backward A/B 在 `C>=4096、N<=16` 时使用
Ascend Triton dot。BF16/FP16 的 forward/A 使用最大 `BLOCK_C=2048`，FP32 使用 1024，
B 使用 1024/8 warps；forward 使用 `NP=max(4,N)`
减少 padding，backward A/B 使用 NP=16；矩阵乘均指定 IEEE 精度。forward/A 将二维 tile 映射到一维 program，每次最多 launch 4096 个 tile，包装层
通过 `tile_offset` 覆盖更大 grid，不使用原生回退。`sb<=512` 的 bias 使用单 kernel 归约，
较大 `sb` 使用 16 路 FP32 partial 和 merge 两个 Triton kernel。
通用实现和其他 vendor 的配置不变。

小形状、目标中等形状 forward 和 split-C/merge Kernel 使用固定配置，不进入
Autotuner。首次编译和调优发生在正式性能测量之前。

### 4.3 NVIDIA 输入复用路径

提交 `eeabbc4` 为 forward 和 backward A 增加了 `REUSE_INPUT` 静态分支。满足以下
条件时启用：

```text
vendor == "nvidia"
且不属于小形状
且（n >= 8 或 sb > 512 或 dtype == FP32）
```

Forward 将循环顺序调整为以输入流 `j` 为外层，每个 `original_residual[j]` 只读取
一次，并用 `[N,BLOCK_S,BLOCK_C]` 的 FP32 accumulator 同时维护全部输出流。原实现会在
`N` 个输出流之间重复读取残差，复用分支将这部分读取次数从 `N²` 降为 `N`。

Backward A 使用同样的三维累加方式，在同一轮 `j` 循环中计算 `grad_x` 和全部
`grad_original_residual`，使每个 `grad_output[j]` 只读取一次。未启用小形状融合或
split-C 时，Backward B 和 bias 归约保持原实现。

三维 accumulator 会增加寄存器压力，因此 BF16/FP16 的 `sb=512、N=4` 等收益不稳定
的中等形状仍走原分支。复用分支还将 autotune 配置裁剪为：

| dtype | `BLOCK_C` | `BLOCK_S` | `num_warps` | 配置数 |
| --- | --- | --- | ---: | ---: |
| BF16/FP16 | 512 | 1 或 2 | 2 | 2 |
| FP32 | 256 或 512 | 1 或 2 | 2 | 4 |

### 4.4 NVIDIA 小形状专用路径

小形状对 Python dispatch、Autotuner 和设备上下文非常敏感。Forward 对满足以下条件的
输入使用专用路径：

```text
vendor == "nvidia" 且 sb <= 8 且 C <= 1024
```

Backward 在相同范围内进一步限制 `n <= 4`，启用单 Kernel 融合；此范围内的 `n=8`
仍使用原来的 A、B 两个 Kernel。专用路径直接调用原始 `JITFunction`，完全绕过
`LibEntry` 和 Autotuner，并显式传入固定配置：

| 条件 | `BLOCK_C` | `BLOCK_S` | `num_warps` |
| --- | ---: | ---: | ---: |
| `C <= 256` | 64 | `n <= 2` 时为 2，否则为 1 | 4 |
| `256 < C <= 1024` | 128 | `n <= 2` 时为 2，否则为 1 | 4 |

小形状 backward 融合以 B 的 C 循环为主线，每轮共享 `grad_output` tile，同时计算
`grad_x`、全部 `grad_original_residual`、`grad_h_post` 和 `grad_h_res`。它既减少一次
launch，也避免 A、B 重复读取上游梯度。Bias 梯度仍从已写回输入 dtype 的 `grad_x`
归约，保持原有舍入路径。

NVIDIA 下不进入 `torch_device_fn.device()` 上下文，避免额外固定开销。小形状 forward
和未融合的 backward 仍设置 `REUSE_INPUT=False`。

### 4.5 中等形状 forward 固定配置

BF16/FP16 的 `(64,8,4,4096)` 保留逐输出流原计算结构，但直接调用原始
`JITFunction`，使用独立固定配置并绕过 Autotuner：

| dtype | `BLOCK_C` | `BLOCK_S` | `num_warps` |
| --- | ---: | ---: | ---: |
| BF16 | 256 | 2 | 2 |
| FP16 | 512 | 2 | 4 |

该分支只匹配 `N=4、sb=512、C=4096`，不改变其他中等形状、N=8、大形状或小形状的
分发。目标形状判断放在小形状分支之后的 `elif` 中，小形状不执行新增检查。

全流复用和每次维护两个输出流的候选结构都经过实测，但完整 48 点回归有回退，因此没有
保留。最终收益来自已经测定的配置和更低的提交开销，而不是改变访存计算结构。

### 4.6 N=8 的 Backward B split-C 路径

满足以下受限条件时，Backward B 使用 split-C：

```text
非 native FP32 归约（即非「FP32 且 C >= 4096」，该情形所有 vendor
都走 torch.bmm 的原生归约路径）
且 n == 8
且 sb <= 128
且 C >= 4096
且（vendor == "nvidia" 且不属于小形状 → split-C=4；
    其他 vendor → split-C=16）
```

固定配置为 `BLOCK_C=256`、`BLOCK_S=2`、`num_warps=2`。以 `(128,1,8,7168)` 为例，
单阶段 B 只有 64 个 sb CTA；NVIDIA 的 split-C=4 第一阶段变为 `64 × 4 = 256` 个 CTA，
增加通道方向并行度。第一阶段写出 `[split,sb,n,n]` 和 `[split,sb,n]` 的 FP32 partial，
第二阶段用一个小 Kernel 合并，不使用 atomic，也不改变 bias 的归约来源。

通用后端（如沐曦）同样需要这条路径：该后端 Backward B 的 grid 只沿 sb 展开，
`(128,1,8,7168)` 时每个 CTA 要串行扫完整个 C=7168，g_hp_hr Kernel 占反向总耗时约
82%（371/451 us）。split-C=16 后该 Kernel 降到约 150 us，完整反向（BF16/FP16）从
约 450 us 降到约 220 us，该 case 对 eager 基线的 speedup 从 0.65-0.70x 提升到
1.3-1.4x。沐曦上用更大的 split-C 是因为其单 CTA 沿 C 的串行扫描更长，需要更多并行
度填满设备。

split-C 会改变 FP32 的累加分组，并增加 partial 读写和一次 merge launch，因此没有推广
到其他 N 或大 sb。N=4 的实验虽然降低了 Graph 时间，但完整 API 明显退化，最终仍保留
单阶段 B。FP32 长通道输入使用下一节的原生归约路径，不再进入 split-C。

### 4.7 FP32 长通道原生归约路径

所有平台在 `dtype == FP32` 且 `C >= 4096` 时使用统一的原生 Backward B，不再按 vendor
选择不同数值路径。Backward A 仍由 Triton 计算 `grad_x` 和 `grad_original_residual`，随后用
PyTorch 计算：

```python
g_hr = torch.bmm(orig_flat, go_flat.transpose(1, 2))
x_bias = x_flat.unsqueeze(1)
if bias is not None:
    x_bias = x_bias + bias.contiguous().view(1, 1, C)
g_hp = (go_flat * x_bias).sum(dim=-1)
```

执行 `bmm` 前会临时设置 `torch.backends.cuda.matmul.allow_tf32=False`，并在结束后恢复原值，
避免后端默认 TF32 降低 FP32 精度。这条路径完全不启动 Triton Backward B，也不分配 split-C
partial，避免先由 Triton 计算后再用原生实现覆盖的重复开销。`grad_bias` 仍从已写回输入 dtype
的 `grad_x` 归约。该实现依赖平台的 PyTorch 矩阵乘和归约能力；Hygon 上观察到的 hipBLAS
初始化问题见第 6.14 节。

### 4.8 分发路径和其他 vendor

| 输入情况 | Kernel 调用方式 | 计算结构 | 设备上下文 |
| --- | --- | --- | --- |
| NVIDIA 小形状 forward | 原始 JITFunction + 固定配置 | 原 forward | `nullcontext()` |
| NVIDIA `(64,8,4,4096)` forward，BF16/FP16 | 原始 JITFunction + dtype 固定配置 | 原 forward | `nullcontext()` |
| NVIDIA 小形状 backward，`n<=4` | 原始 JITFunction + 固定配置 | 单 Kernel 融合 | `nullcontext()` |
| NVIDIA 小形状 backward，`n>4` | 原始 JITFunction + 固定配置 | 原 A、B | `nullcontext()` |
| NVIDIA N=8 split-C，BF16/FP16 | A 直接调用 Autotuner；B 固定配置 | A 输入复用 + split-C/merge | `nullcontext()` |
| 任意 vendor，FP32 且 `C>=4096` | A 沿用平台路径；B 使用 PyTorch | 原生 `bmm` 和通道归约 | 沿用当前平台 |
| NVIDIA 其他满足复用条件的形状 | 直接调用 Autotuner + 裁剪配置 | 输入复用 | `nullcontext()` |
| NVIDIA 其他形状 | 直接调用 Autotuner + 完整配置 | 原实现 | `nullcontext()` |
| 昇腾普通形状 | 独立后端，原始 JITFunction + 固定配置 | 原 forward 和 A/B；不启用 split-C | `torch_device_fn.device()` |
| 昇腾 `C>=4096、N<=16` | Ascend Triton dot forward/A/B | 一维 tile 映射 + 4096 tile 分批 launch | `torch_device_fn.device()` |
| 其他 vendor N=8 split-C（非 FP32） | `LibEntry` + 通用 autotune；B 固定配置 | split-C=16/merge | `torch_device_fn.device()` |
| 其他 vendor 的其余形状 | `LibEntry` + 通用 autotune | 原实现 | `torch_device_fn.device()` |

其他 vendor 的通用路径被保留，但目前没有完成与 NVIDIA 相同范围的正确性和性能验证。

## 5. 实现与优化历程

### 5.1 初始移植

初始实现 `8e78e8e` 按 Megatron Triton 结构移植：

- 一个 forward Kernel。
- 两个 backward Kernel。
- 五个梯度的直接返回接口。
- 通用 autotune 和 FlagTrain `LibEntry` 包装。
- PyTorch reference、Megatron Triton baseline 和性能测试框架。

本机最初安装的 Megatron Core `0.19.2` 不包含该文件，因此使用官方 `main` 源码安装
`0.21.0+2052769` 作为参考和 baseline。

### 5.2 测试覆盖扩充

提交 `c0bbf09` 将正确性测试扩充到 51 项，并补齐：

- BF16、FP16、FP32。
- `n=1/2/4/8`。
- bias 有无。
- 非整块 `C` 和非整块 `sb`。
- 大 `sb`。
- 非连续输入。
- 多组核心与综合性能形状。

### 5.3 小形状 forward 优化

提交 `6cb7ffb` 针对小形状 forward 增加原始 JIT 路径，解决 `LibEntry`、Autotuner
和设备上下文共同造成的固定开销。最终 GPU 3 上 48 个 forward 数据点全部达到
0.8x 以上。

### 5.4 小形状 backward 优化

提交 `dc00e6d` 将同样的思路应用于两个 backward Kernel：

- 小形状使用原始 JITFunction 和固定配置。
- 其他 NVIDIA 形状直接调用 Autotuner 并去掉设备上下文。
- bias reduction 保持 Torch 实现。

最终 GPU 3 上 48 个 backward 数据点全部达到 0.8x 以上。

### 5.5 大形状输入复用优化

提交 `eeabbc4` 对 forward 和 backward A 的循环结构进行优化：

- 使用三维 FP32 accumulator 同时维护多个流，减少残差和上游梯度的重复读取。
- 只在实测有稳定收益的 NVIDIA 形状范围启用，保留小形状和中等形状原路径。
- 裁剪复用分支的 autotune 配置，降低寄存器压力和调优波动。
- 将 `REUSE_INPUT` 加入缓存键，隔离两种计算结构的配置。
- 正确性测试从 51 项扩充到 71 项，覆盖三种 dtype 的复用路径和较大 `N=8`。

最终 48 个 forward 数据点的算术平均加速比由 1.167x 提升到 1.240x，48 个
backward 数据点由 1.161x 提升到 1.171x，两个方向均没有低于 0.8x 的数据点。

### 5.6 小形状融合与 N=8 split-C 优化

本轮先用完整 API、CUDA Graph replay 和分段 GPU 计时区分提交开销与 Kernel 瓶颈，再实施
两项受限优化：

1. `sb<=8、C<=1024、n<=4` 的 backward 改为单 Kernel，减少 launch 并共享
   `grad_output` tile。
2. `N==8、sb<=128、C>=4096` 的 Backward B 改为 split-C=4，以 FP32 partial 增加 C
   方向并行，再由 merge Kernel 合并。
3. N=4 split-C 的完整 API 性能退化，因此撤回，不在最终路径中启用。
4. 新增 BF16/FP32 的 `(128,1,8,4096)` split-C 精度测试，正确性测试由 71 项扩充到
   75 项。

最终 48 个 backward 数据点的算术平均加速比由 1.171x 提升到 1.770x，最低加速比为
0.833x，仍没有低于 0.8x 的数据点。Forward 代码和性能口径本轮未改动。

### 5.7 中等形状 forward 固定配置优化

本轮针对 BF16/FP16 的 `(64,8,4,4096)` 比较了 12 组独立配置和三种计算结构：逐输出流、
全部输出流同时累加、每次维护两个输出流。PTX/SASS 检查显示，BF16 原结构与全流复用
分别有 37 和 17 条全局 load 指令，寄存器约为 59 和 56，均无 spill 或 local memory
读写；但指令数量优势没有转化为稳定的完整 API 收益。

全流复用接入目标形状后，BF16 加速比降至 0.881–0.887x，FP16 有 bias 降至 0.818x，
部分 N=8 和大形状也受配置搜索变化影响而回退，因此撤回。双流结构同样没有稳定收益，
实验代码未保留。

最终仅对目标形状保留原计算结构，BF16 使用 `(256,2,2)`、FP16 使用 `(512,2,4)` 的
`(BLOCK_C,BLOCK_S,num_warps)` 固定配置并绕过 Autotuner。新增 7 项 forward 测试后共
82 项，覆盖三种 dtype、有无 bias、非整块 C=4099 和大形状非连续输入。最终 forward
平均加速比为 1.253x，最低为 0.980x。

### 5.8 FP32 长通道归约的跨平台修正

PPU 的 `(128,1,8,4096)` FP32 backward 首先在 `grad_h_res` 上暴露归约误差。初版修复只针对
`thead` 调用原生 `bmm`，随后为避免平台特判，改为所有平台在 FP32 且 `C>=4096` 时统一使用
原生 Backward B。最终版本同时计算 `grad_h_res` 和 `grad_h_post`，完全跳过 Triton B 及
split-C partial 分配，而不是在 Triton 计算后覆盖结果。

本机受影响的 4 项 split-C/长归约测试和模拟 PPU 通用路径的有无 bias 用例均通过。最新通用版本
尚未完成完整 82 项回归，也没有重新测量 FP32 长形状性能，因此第 9 节保留的是修改前的历史
性能结果。

## 6. 历史问题及解决方式

### 6.1 非连续 bias 导致 `grad_h_post` 错误

**现象**：非连续输入测试中 `grad_h_post` 与 PyTorch reference 不一致。

**原因**：Kernel 按连续地址读取 bias，但包装层最初只对四维输入做了连续化，没有处理
一维 bias 的 stride。

**解决**：forward 和 backward 都显式执行 `bias.contiguous()`，并增加包含非连续 bias
和 `grad_output` 的回归测试。

### 6.2 autotune 重复执行导致 atomic bias 累加错误

**现象**：试验 FP32 atomic bias 融合时，bias 梯度在调优后被放大约 9 万倍。

**原因**：autotune 会反复调用同一个 Kernel 测速，atomic 输出在工作区中不断累加。

**处理**：该实验路径被撤回。当前仍使用 `g_x.sum(dim=0)`，不在生产实现中使用 atomic
bias 累加。即使增加 `reset_to_zero`，atomic 版本的性能也明显更差，因此没有保留。

### 6.3 `g_hr` 一次性写出出现转置

**现象**：尝试删除 `N²` 次提取归约、一次性写出 `g_hr_acc` 时，`n>2` 的 `h_res`
梯度出现转置。

**原因**：accumulator 的逻辑列顺序与输出 `(i,j)` 地址映射不一致。

**处理**：虽然修正映射后数值可以通过，但性能没有稳定收益，因此该候选实现也被撤回。
当前仍保留 Megatron 的提取和写出方式。

### 6.4 多种反向结构重写性能退化

以下候选均经过实际测试，但没有保留：

- 单 buffer 分配全部反向梯度：收益约 0.1%，不足以抵消复杂度。
- 按 `i` 独立累加并直接写出：核心反向约 0.116 ms，明显慢于原实现约 0.069 ms。
- 三维 batched `tl.dot`：约 0.118 ms，`n<=8` 时填充到 16 维不划算。
- `sb` 作为额外 autotune key：收益不能稳定复现。
- 单阶段自定义 bias 归约：约 0.102 ms。
- 两阶段 split-sb bias 归约：约 0.128 ms。

结论是：在小 `n` 场景下，Megatron 当前的标量分块结构比这些重写更合适。

### 6.5 性能计数器权限限制

NCU 无法采集寄存器和 occupancy 数据，环境返回 `ERR_NVGPUCTRPERM`。没有通过提权
绕过限制，而是改用 Nsight Systems 的 CUDA 时间线确认反向各部分的 GPU 耗时：

- Backward B 约 28.6 μs。
- Torch bias reduction 约 18.3 μs。
- Backward A 约 12.4 μs。

这也解释了为什么 bias reduction 曾是重点优化对象，尽管自定义方案最终没有超过 Torch。

### 6.6 小形状性能测量波动

早期曾出现 forward 0.4–0.6x，原因有两类：

1. 多个 benchmark 并发执行时，小形状对 CPU 编译、autotune 和其他进程争用非常敏感。
2. 在 GPU 3 上进一步复现后，确认 `LibEntry`、Autotuner 和设备上下文确实会叠加固定
   dispatch 开销。

处理方式：

- 性能测试改为串行执行，避免把进程争用误当成 Kernel 退化。
- 小形状彻底绕过这些通用分发层，而不是仅调整 block 配置。

### 6.7 测试时间增长

扩充 dtype 和 shape 后，每个组合都会触发不同 Kernel 编译和 autotune，完整测试首轮
耗时明显增加。性能 JSON 的生成不改变测量口径，但会增加完整运行时间。当前保留这些
覆盖，因为小形状和 bias 分支确实曾暴露过数值与性能问题。

### 6.8 输入复用分支继承错误的 autotune 配置

**现象**：输入复用 Kernel 在独立微基准中优于原实现，但接入公开 API 后，部分
`(512,4,4,4096)` forward 数据点反而从约 0.05 ms 增加到约 0.07 ms；不同轮次的
加速比出现大幅波动，仅裁剪候选配置也不能解决。

**原因**：原分支和复用分支使用同一个 Autotuner，而缓存键最初只有 `C`、`N` 和
Triton 自动加入的 dtype。同一 `(C,N,dtype)` 下，`sb=512` 的原分支先完成调优后，
`sb=2048` 的复用分支会直接继承其配置，不再执行配置裁剪和独立调优。三维累加器对
block 和 warp 数更敏感，因此继承原分支的高寄存器压力配置会明显退化。

**解决**：

1. 将 `REUSE_INPUT` 加入 forward 和 backward A 的 autotune key，确保两种计算结构
   分别缓存最优配置。
2. 复用分支只保留实测稳定的 `BLOCK_C/BLOCK_S/num_warps` 组合。
3. 连续执行 `(64,8,4,4096)`、`(512,4,4,4096)` 和 `(128,1,8,7168)`，确认缓存中
   原分支与复用分支的配置相互独立，再运行完整 96 点性能回归。

### 6.9 Graph 时间下降但完整 API 退化

本轮先对 BF16 做分段测量。小形状 `(2,4,4,1024)` 无 bias 时，完整 API 约 47.8 μs，
Graph replay 约 32.5 μs，其中 A、B 的 GPU 时间分别约 1.2 μs 和 31.0 μs。这说明
提交和 launch 间隔不可忽略，但 B 本身的 C 循环同样是主要瓶颈。

N=4 的 `(64,8,4,4096)` 使用 split-C=2 后，Graph replay 从约 38.8 μs 降到 31.9 μs，
但完整 48 点 benchmark 中 BF16 无 bias/有 bias 加速比降至 0.744x/0.754x，FP32 更降至
0.603x/0.655x。额外的 partial 分配、merge launch 和提交开销抵消了纯 GPU 收益，因此最终
撤回 N=4 split-C。

N=8 使用 split-C=4 后，`(128,1,8,7168)` 无 bias 的 Graph replay 从约 89.1 μs 降到
45.1 μs，完整 API 也有稳定收益，所以只保留受限的 N=8 路径。这个结果表明 Graph 适合
辅助定位 Kernel 瓶颈，但最终是否保留优化仍必须以完整 API 回归为准。普通 `do_bench`
会清理 L2，而 `do_bench_cudagraph` 不清理 L2，两者差值也不能全部解释为 CPU 开销。

### 6.10 新增融合和 merge 中的索引映射错误

小形状融合初次实现把 `grad_original_residual[j]` 错写为使用 `H_res[i,j]`，导致结果
相当于矩阵方向转置；修正为 `H_res[j,i]` 后通过全部 backward 测试。

Split-C merge 初次实现直接按 accumulator 的 `(i,j)` 直觉提取 partial，造成 `grad_h_res`
转置。当前与原 B 保持一致：累加列使用 `j*N+i`，最终输出 `(i,j)` 时提取 `i*N+j`。
新增的 split-C CPU autograd 回归用例覆盖了该映射。

### 6.11 flagtree AABS 原地污染 autotune 配置导致性能测量失真

**现象**：MetaX 环境 benchmark 中，排在 `(1,1,1,1)` 之后的 case 的 train latency 被放大
100~1000 倍（如 backward `(8192,4,4,7168)` 测出 3378 ms，真实值约 6.5 ms），且同一
shape 的 forward/backward 数字几乎相同。

**原因**：flagtree Triton fork 的 AABS（`FLAGTREE_AABS`，默认开启）在 autotune 时按
"block size > 张量维度则缩小" 的规则调整配置，但 `update_bs` 会**原地修改 autotuner
共享的 config 对象**。遇到 `C=1` 的迷你形状后，全部候选配置的 `BLOCK_C` 被永久缩成 1，
后续所有 key（包括之前已缓存的最优配置，因为缓存持有同一批对象）都退化到
`BLOCK_C=1`，kernel 退化为每个 program 处理 1 个 channel。

**解决**：算子的 kernel 启动点外套 `_disable_aabs()`，调用期间临时关闭
`triton.knobs.autotuning.adjust_block_size`，退出恢复；对无此 knob 的原版 Triton 自动
跳过。根治需要 flagtree 上游修复 `update_bs` 的原地修改问题。

### 6.12 MACA 每线程私有内存上限导致个别配置启动失败

**现象**：MetaX 环境 fp32 backward 测试报 `Triton Error [MACA]: memory size or pointer
value too large to fit in 32 bit`，崩溃点在 autotune 测速
`_triton_hpb_bwd_g_hp_hr_kernel` 的 `BLOCK_C=128/256, BLOCK_S=8` 配置。

**原因**：该 kernel 持有 `(BLOCK_S, N²)` 的累加器，`BLOCK_S=8, N=8` 时编译器产生的
每线程私有内存需求（实测 5~7 KB/thread）超过 MACA 驱动默认上限（4 KB/thread），
`mcModuleLaunchKernel` 直接拒绝启动。stock `triton.autotune` 只捕获
`OutOfResources` 等三类异常，普通 `RuntimeError` 会中断整个调优。

**解决**：本文件的三个 autotuned kernel 改用本地 `_hpb_autotune` 装饰器——在
`triton.autotune` 之外给 `_bench` 包一层 `RuntimeError` 捕获，启动失败的配置按
无穷慢处理并跳过（与 `libentry.LibTuner` 的容错策略一致）。此前全量测试未暴露是因为
dtype14 先在 `sb=1` 下为同一 key 完成了调优，dtype16 直接命中缓存，从未在大 grid 下
启动过该配置。

### 6.13 PPU FP32 `grad_h_res` 长归约误差

**现象**：PPU 上 `(128,1,8,4096)` 的 FP32 backward 在有/无 bias 时均只有 `grad_h_res`
超过 `atol=rtol=1e-5`，其余 80 项通过。实际值与参考值非常接近，不是索引或转置错误。

**原因**：PPU 不进入 NVIDIA split-C 分支，通用 Triton Backward B 对 4096 个通道的 FP32
归约顺序与 Torch reference 不同，累积舍入误差超过当前容差。

**处理**：先验证原生 `torch.bmm(orig, grad_output.T)` 与 Autograd 参考结果一致，再将其推广为
所有平台的 FP32 长通道原生归约路径。为避免重复计算，最终跳过整个 Triton Backward B，
`grad_h_post` 也由原生乘法和归约产生。没有放宽测试容差。本机专项测试和模拟 PPU 路径通过，
PPU 实机仍需复测。

### 6.14 Hygon 测试参考实现触发 hipBLAS 分配失败

**现象**：Hygon 上多个 forward 用例报 `HIPBLAS_STATUS_ALLOC_FAILED when calling
hipblasCreate(handle)`。堆栈显示被测 Triton forward 已完成，失败发生在随后计算参考值的
`torch.bmm`，不是输出数值比对失败。日志中的 TF32 不支持警告和手动中断产生的 Triton
`KeyboardInterrupt` 都不是该错误的根因。

**当前判断**：应优先在 Hygon 新进程和空闲显卡上单独运行最小 `torch.bmm`，结合 `hy-smi`、
内存占用及 HIP/hipBLAS 版本检查运行环境。`HIPBLASLT_ALLOW_TF32=1` 只控制 TF32，不能解决
句柄分配失败。

**对当前实现的影响**：FP32 长通道 backward 也使用原生 `bmm`，如果 Hygon 的最小 `bmm`
同样失败，后续 backward 也会受影响。曾评估把参考矩阵乘改成逐元素乘法求和，但本地 BF16
用例因舍入路径变化超出现有容差，因此没有将该实验作为修复提交。目前此问题仍需 Hygon 实机
最小复现结果，尚未修改测试或算子来绕过 hipBLAS。

### 6.15 昇腾 autotune 触发编译器卡死

**现象**：BF16 `(1,1,8,257)` 无 bias 的 backward 用例长期不返回，调用栈停在
`bishengir-compile`，并非 NPU kernel 同步或 PyTorch reference。只隔离 `N=8` 后，完整回归又在
FP16 `(2,1,4,130)` 的 `N=4` backward 遇到同类问题。

**原因**：通用 Kernel 的部分 autotune 候选会触发该编译器问题；已在 backward A/B
的多组 `BLOCK_C/BLOCK_S/num_warps` 上复现。`_safe_bench` 无法处理这种情况，因为编译子进程
没有抛出异常，而是一直运行。

**解决**：新增昇腾独立后端，直接启动通用 forward、backward A/B 的原始 JITFunction，完全
跳过通用 autotune。forward/A 使用最大 512 的通道块，B 固定 `(64,1,4)` 以避开编译问题；
大 grid 仅调整 forward/A 的 `BLOCK_S`。通用源码和其他 vendor 的搜索空间均不修改。

具体接入步骤如下：

1. 新增 `src/flag_train/runtime/backend/_ascend/megatron/fused_h_post_bda.py`，实现昇腾
   forward/backward 包装层。包装层通过 `_generic()` 延迟导入通用模块，复用其输入校验、原始
   Triton kernel、AABS 保护和 FP32 长通道原生归约；forward/A 的 `_FIXED_CONFIG` 为
   `BLOCK_C=512、BLOCK_S=1、num_warps=4`，并按 C 缩小通道块；B 使用独立 `_B_CONFIG`。
   两条路径均不进入通用 Autotuner，也不启用 split-C。
2. 在 `src/flag_train/runtime/backend/_ascend/megatron/__init__.py` 中导出
   `fused_h_post_bda_forward` 和 `fused_h_post_bda_backward`，并加入 `__all__`，由
   `SpecOpRegistrar` 覆盖同名通用接口。
3. 这两个导出必须放在 `fused_gated_norm` 导入之前，并保留该行的 `# isort: skip`。否则
   `fused_gated_norm` 会先触发通用 Megatron 注册，形成循环导入，导致昇腾覆盖未生效。
   昇腾实现的 `_generic()` 延迟导入同样用于避免注册阶段的循环导入。
4. 通用文件 `src/flag_train/megatron/fused_h_post_bda.py` 不需要修改；其他 vendor 仍走原有
   分发、autotune 和 split-C 路径。

### 6.16 昇腾性能测试误选 Megatron Triton baseline

**现象**：forward 核心性能测试首个正式 shape 长期不返回，中断栈位于 Ascend 的
`triton/backends/ascend/compiler.py`。模块导入时的小 shape baseline 探测却能通过，因此测试
错误地选择了 `megatron _triton_h_post_bda_fwd/_bwd`，而不是组合实现。

**原因**：当前环境虽然能导入 Megatron Triton 实现，但其 48 组 autotune 配置同样会触发昇腾
编译器卡死；小 shape 探测命中已有缓存，不能证明其余正式 shape 可用。

**解决**：仅修改 `benchmark/megatron/test_fused_h_post_bda.py` 的 baseline 选择：昇腾不再
探测 Megatron Triton，直接使用 Megatron-native 组合实现；其他 vendor 的选择逻辑保持不变。
当前环境的 `torch.compile` 还会因 Inductor 与 Ascend Triton 接口不兼容而失败，已有异常处理
会回退到 eager，最终显示 `Megatron-native torch eager (fallback)`。修复后 forward/backward
核心性能集均通过。

### 6.17 昇腾 profiler 空采样导致性能数据全为 0

**现象**：直接运行完整性能命令时反复打印 `No profiling data on the device side`，表格中的
latency 为 0，speedup 因而也为 0；算子本身已经正常执行。

**原因**：Ascend Triton 的 `do_bench_npu` 在 mspti 采样失败后回退到 torch_npu profiler，
后者的 `_collect_prof_result` 在 CSV 没有设备数据时直接返回 `0.0`。通用 benchmark 无法区分
这个值与正常的零耗时，于是将其作为有效结果显示。此前也观察到同类采样问题产生 `inf`。

**解决**：只在本算子的 `FusedHPostBDABenchmark.get_latency` 中隔离昇腾计时：kernel 模式下
统一使用 NPU Event，对 30 次完整 API 调用计时，baseline 和 FlagTrain 使用相同口径，不再进入
不稳定的 mspti/profiler 路径。没有修改通用 `benchmark/base.py` 或其他算子。Event 统计完整
API 的设备时间，与 profiler 的纯 kernel 统计口径不同；昇腾结果不能与历史 profiler 绝对值
直接比较，但同一表格内的 speedup 仍使用一致的计时方式。

### 6.18 昇腾大 grid 超过启动上限

**现象**：完整性能集的 `(512,4,4,4096)` forward 报 `coreDim=131072`，要求不超过 65535。
核心集没有覆盖该 grid 规模，因此之前的核心性能回归无法发现。

**原因**：基础配置 `BLOCK_C=64、BLOCK_S=1` 使该形状的 grid 为 `(2048,64)`，Ascend 将二维
grid 展平为 coreDim 后检查总上限。最大的 `(8192,4,4,7168)` 也需要处理。

**解决**：普通通用 kernel 的 `_grid_a_config` 在 grid 展平大小超过 65535 时按 2 的幂增大
`BLOCK_S`。dot forward/A 最终使用一维 tile 映射和 `tile_offset`，每批最多 4096 个 tile，
通过多次 launch 覆盖任意大 grid。单 kernel 持久循环和单批 65535 tile 曾触发 AICore timeout，
未保留；最终没有原生回退。

### 6.19 昇腾大通道形状加速比低

**现象**：初版昇腾核心性能集中，大通道 forward/backward 的 speedup 多在 0.003–0.04；小形状
也低于 1，但绝对耗时只有约 0.2–0.5 ms。

**定位**：配置扫描必须在实际注册模块 `_ascend.ops.megatron.fused_h_post_bda` 上进行；使用
完整包路径会重复加载文件，得到另一份模块全局配置。修正实验后，N=4 的
`(64,8,4,4096)` BF16 forward 在 `BLOCK_C=64/128/256/512` 下分别约为
20.03/8.85/4.86/2.12 ms。主机入队约 0.34 ms/次，Python 不是主要瓶颈。`REUSE_INPUT=True`
在 512 配置下仅从约 2.11 ms 降到 1.98 ms，N² 重复读取不是当前首要问题。

N=8 的 `(128,1,8,7168)` BF16 backward 分阶段测量为：A 使用 64 时 31.06 ms，使用 512 时
3.25 ms；B 使用 64 时 4.84 ms。B 使用更大通道块会触发编译卡死，因此不能把 A 的配置直接
套用到 B。

**处理**：forward/A 按 C 选择最大 512 的通道块，B 独立保持 64。完整 82 项正确性测试通过，
核心性能集 `2 passed`。BF16 无 bias 的代表数据如下：

| Shape | 方向 | 调整前 (ms) | 调整后 (ms) | 降幅 |
| --- | --- | ---: | ---: | ---: |
| `(64,8,4,4096)` | Forward | 20.064 | 2.084 | 89.6% |
| `(128,1,8,7168)` | Forward | 34.801 | 3.139 | 91.0% |
| `(64,8,4,4096)` | Backward | 23.254 | 4.786 | 79.4% |
| `(128,1,8,7168)` | Backward | 40.194 | 8.020 | 80.0% |

**剩余差距**：组合 baseline 使用高效原生 `bmm` 和归约；当前 Triton kernel 仍采用标量流
循环和 FP32 累加。调整后 N=8 backward 的 B 已占总耗时约 60%，A 和 forward 也仍明显慢于
原生组合。小形状则主要受每次调用的固定开销影响。因此本次修正解决的是异常低效配置，并不
意味着已经达到或超过组合 baseline。

后续完整回归的 forward 48 点全部通过；`(512,4,4,4096)` BF16 forward 从约 87.6 ms 降到
8.86 ms。但最大 `(8192,4,4,7168)` BF16/FP16 仍约为 2242 ms：grid 上限迫使
`BLOCK_S=8`，与 512 通道块形成 4096 元素的大 tile，抵消了通道块收益。FP32 同形状从约
296 ms 降到 88 ms。完整命令在 forward 完成后继续执行 backward，达到 20 分钟外部超时而终止，
不是最新配置下的完整 96 点通过记录。

### 6.20 Ascend `tl.dot` 与原生归约混合优化

继续分阶段测量发现，N=8 原生 B 仅约 0.109 ms，相比 Triton B 的 4.84 ms 快约 44 倍；
A 的 `REUSE_INPUT` 仅约 11% 收益。Ascend 专用 `tl.dot` 将 N 填充到 16，forward 使用
`H_res.T @ R`，backward A 使用 `H_res @ G` 并在同一 kernel 计算 `g_x`。FP32 指定
`input_precision="ieee"`，原型通过 1e-5 容差检查。

中间方案对 `C>=4096、N<=16` 且 grid 合法的形状启用 dot forward/A，并将 B 换成原生
`bmm + 归约`；grid 超限时整个方向使用原生组合。普通形状仍保留固定配置通用 kernel。
该方案完整 82 项正确性通过，核心性能集 `2 passed`。BF16 无 bias 核心代表数据：

| Shape | 方向 | 标量大块版本 (ms) | dot/原生混合 (ms) | 当前 Speedup |
| --- | --- | ---: | ---: | ---: |
| `(64,8,4,4096)` | Forward | 2.084 | 0.864 | 0.220x |
| `(128,1,8,7168)` | Forward | 3.139 | 0.480 | 0.238x |
| `(64,8,4,4096)` | Backward | 4.786 | 0.989 | 0.346x |
| `(128,1,8,7168)` | Backward | 8.020 | 0.536 | 0.414x |

相对最初 64 通道块版本，上述四点的总收益分别约为 23.2x、72.5x、23.5x 和 75.0x。
核心大通道形状仍未全面超过原生 baseline；继续提升需要减少 dot padding、输出写回和多次
原生调用开销，而不是恢复已验证会退化的配置。最终完整 96 点性能集 `2 passed`，耗时
74.59 秒；最大 grid 形状由原生回退执行，speedup 约为 0.95–1.00，不再出现 2 秒级退化。

### 6.21 原生矩阵乘与自定义后置/归约混合实验（已撤回）

dot 方案仍有 N=16 padding 和中间输出开销。实验中曾用原生 `bmm` 负责矩阵乘，自定义 kernel
负责 forward post 和 backward `g_x`，获得下表收益。但该实现不符合“代码中不调用 Torch
原生算子”的约束，因此已撤回，最终版本不使用该路径。历史实验的 82 项正确性和 96 点性能
均通过，耗时 63.50 秒。BF16 无 bias 代表结果：

| Shape | 方向 | dot/原生方案 (ms) | 最终混合方案 (ms) | 最终 Speedup |
| --- | --- | ---: | ---: | ---: |
| `(64,8,4,4096)` | Forward | 0.864 | 0.442 | 0.431x |
| `(128,1,8,7168)` | Forward | 0.480 | 0.270 | 0.425x |
| `(64,8,4,4096)` | Backward | 0.989 | 0.438 | 0.789x |
| `(128,1,8,7168)` | Backward | 0.536 | 0.433 | 0.509x |

该实验中，N=8 FP32 有 bias forward 达到 1.116x；`(512,4,4,4096)` backward 各 dtype
约为 0.73–0.94x；最大 grid 形状约为 1.00x。这些数据不代表当前纯 Triton 实现。

### 6.22 最终纯 Triton dot、分批 grid 和 bias 归约

最终 Ascend 实现不调用 Torch 原生计算算子。大通道 forward/A/B 都使用 Triton dot；forward
计算 `H_res.T @ R + H_post*(x+bias)`，A 计算 `H_res @ G` 和 `g_x`，B 通过
`orig @ go.T` 计算 `g_hr` 并同时计算 `g_hp`。forward 使用 `NP=max(4,N)` 减少 padding，
backward A/B 使用 NP=16；矩阵乘使用 IEEE 精度。
forward/A 每个 program 处理一个线性化 tile，每次 launch 最多 4096 个 tile，通过 `tile_offset`
覆盖大 grid。持久 loop 和单批 65535 tile 曾触发 AICore timeout，未保留。

BF16/FP16 的 forward/A 使用 2048 通道块；2048 在 FP32 出现数值错误，1024 则通过 1e-5
检查，因此 FP32 使用 1024，B 使用 1024/8 warps。4096 通道块存在 UB/CC 溢出，不可使用。Bias 不再
调用 `g_x.sum()`：`sb<=512` 使用单 kernel 归约，较大 sb 使用 16 路 FP32 partial 和 merge。
普通小形状仍使用固定配置通用 Triton kernel。最终 82 项正确性全部通过。

### 6.23 大 grid 分批 dot 和 FP32 配置优化结果

最大形状曾因通用大 `BLOCK_S` 路径只有约 0.01x。改为 4096 tile 分批 dot 后，BF16
`(8192,4,4,7168)` 稳态 forward/backward 约 29.7/52.9 ms，speedup 约 0.78/0.60；
不是首次调用的编译耗时。FP32 从 512 改为 1024 后，最大形状 forward 约 50 ms，speedup
约 0.74；小 sb 的单 kernel bias 进一步减少固定 launch 开销。

最终完整 96 点性能 `2 passed`，耗时 701.30 秒。算术平均 speedup：forward 48 点为
**0.689x**，backward 48 点为 **0.660x**，整体 96 点为 **0.674x**。整体均值由此前通用大
`BLOCK_S` 版本的约 0.501x 提升约 35%；所有数据点仍为纯 Triton 实现。

### 6.24 小形状融合和 B warp 优化结果

`sb<=8、C<=1024、N<=4` 的 backward 改为通用单 kernel 融合，核心小形状无/有 bias 从约
0.42/0.65 ms 降到 0.29/0.42 ms；普通小形状通道块使用 64。大通道 B 使用 1024/8 warps。
A/B 单 kernel 融合在 512 下约 0.80 ms，比分离路径慢，1024 又有 UB 溢出，因此未采用。

最终 82 项正确性通过，完整 96 点性能 `2 passed`，耗时 183.12 秒。算术平均 speedup：
forward **0.652x**，backward **0.838x**，整体 **0.745x**。Backward 均值提升明显，但整体
尚未达到 1.0；小形状固定启动成本和部分 forward 仍是主要限制。

### 6.25 原生 N dot 实验与最终组合

将 forward/A/B 全部改为 `NP=max(4,N)` 后，N=8 forward 原型约 0.170 ms、B 原型约
0.080 ms，但完整回归均值为 forward 0.757x、backward 0.723x、整体 0.740x。随后保留
forward 原生 N，A/B 恢复 NP=16，最新完整 96 点性能 `2 passed`，均值分别为
**0.688x/0.715x/0.702x**。不同轮次 backward 波动较大，因此同时保留 6.24 的 0.745x
历史结果，不将其视为当前轮次值。最终 82 项正确性通过，整体仍未达到 1.0。

### 6.26 Forward 全 dot 优化

小核心 forward 从通用标量路径切换到 dot 后，C=1024 的实测约从 0.240 ms 降到 0.218 ms；
C=1/65/130 也有约 6%–16% 收益。因此最终 `N<=16` 的 forward 全部使用 dot，并按 C
裁剪通道块，backward 的适用范围不变。小 backward 全融合因 `g_hr/g_hp` 需要跨通道归约，
直接按通道分片写回会互相覆盖，已撤回。

最新 82 项正确性通过，完整 96 点性能 `2 passed`，耗时 93.62 秒。算术平均 speedup：
forward **0.714x**、backward **0.726x**、整体 **0.720x**。整体仍未达到 1.0，但 forward
较大形状已有多点超过 1.0，完整回归耗时进一步下降。

### 6.27 固定配置预绑定直启与 backward dot 门槛扩展

裸 JIT 的参数绑定和分发在小形状上占主要时间。单 kernel 对照中，原始启动的主机/Event 时间约为
0.092/0.093 ms，固定配置预编译直启降为 0.052/0.053 ms。曾尝试直接使用 `libentry` 包装，
但首次回归在后续 FP32 编译用例长时间停滞，未进入性能阶段，因此没有采用。最终实现使用
Ascend 本地的 `_launch_fixed`：首次调用通过 `warmup` 取得编译 kernel，后续直接调用编译后的
runner，不恢复 autotune。缓存 key 包含 kernel、设备、grid、warps、constexpr、输入 dtype、
每个指针的 16 字节对齐，以及普通整数的 `1/16` 特化，避免不同输入对齐或 stride 错用 kernel。
forward/A/B dot、小 backward 和 bias kernel 均接入该路径；通用标量回退保持不变。

`bias.contiguous()` 同时移到所有启动循环外，每次公开 API 调用至多执行一次。82 项正确性（包括
非连续输入）全部通过，完整 96 点性能 `2 passed`。算术平均 speedup 为 forward **0.741x**、
backward **0.800x**、整体 **0.770x**，高于上一版的 0.714x/0.726x/0.720x。小 forward 通常从
0.22–0.25 ms 降为 0.19–0.21 ms；FP32 最大 forward 减少约 2 ms 启动开销。小形状完整 API
仍明显慢于单独 kernel，输入校验、输出分配、view/contiguous 和设备上下文是剩余固定成本。

独立扫描通用标量 A、dot A、通用标量 B、dot B 后，将 backward dot 门槛从 `C>=4096` 扩到
`C>=512、N<=16`，小 backward 融合路径仍优先。BF16/FP32、sb=8/128、N=4/8、C=512/2048
的 16 组补充正确性全部通过。FP32、sb=128、N=4、C=512 时，A 从约 0.531 ms 降为
0.088 ms，B 从约 0.119 ms 降为 0.100 ms。个别单项如 FP32、sb=8、N=8、C=512 的 A
约慢 5%，但 B 约快 22%，A+B 合计仍有收益，因此没有继续增加 A/B 分离门槛。该扩展主要改善
非标准 C=512/1024/2048 形状；标准 96 点中的对应小 backward 已由融合 kernel 接管。

B 的 1024 通道块仍优于 512；warps/stages 差异约 0.003 ms，与主机波动同量级，未修改。
最大 BF16 形状将每批 4096 tile 改为 8192/16384 后，主机入队略降，但 forward/backward 的
Event 变化不足 0.4%，因此保留 4096 分批，不采用更大 grid。

### 6.28 Backward B padding hint 和 bias 归约块

`dot_pad_only_k` 必须按方向验证，不能统一加到所有 dot。N=8、sb=128、C=7168 的 BF16
独立 B 对照两轮约快 6%–12%；最大 BF16 `(8192,4,4,7168)` 完整 backward 从约 48.67 ms
降为 44.89 ms，约快 7.8%。相反，N=4、sb=512、C=4096 的 forward 从约 0.245 ms 退化到
0.414 ms。因此只给 B 的 `orig/go` 输入调用 Ascend `compile_hint(...,
"dot_pad_only_k")`，forward 和 A 不添加该 hint。

大 sb bias 归约保持 16 路 FP32 partial + merge，并将 `BLOCK_S` 从 8 调为 16。最大 sb 的
BF16 bias 从约 0.429 ms 降为 0.375 ms，FP32 从约 0.784 ms 降为 0.765 ms。SPLIT=32 没有
收益；SPLIT=64 的 merge tile UB 溢出；BLOCK_S=32 在 BF16 下更慢，在 FP32 下也会 UB
溢出，均未采用。归约输入仍是已经转换为输入 dtype 的 `g_x`，没有改成原生 `sum` 或改变
输入舍入路径。

B-only hint 和 BLOCK_S=16 的最终版本 82 项正确性全部通过，完整 96 点性能 `2 passed`，
耗时 222.19 秒；算术平均 speedup 为 forward **0.774x**、backward **0.801x**、整体
**0.787x**。Backward 均值与仅预绑定版本的 0.800x 基本持平，收益主要体现在部分大形状，
不能声称所有形状都有稳定提升。曾根据单轮逐点对比把 hint 限制为 `C>=7168`，但该版本完整
回归只有 forward 0.729x、backward 0.775x、整体 0.752x，因此撤回条件并恢复 B-only hint。
不同轮次波动较大，配置选择以完整回归和可重复的大形状定向结果共同判断，整体仍未达到 1.0。

### 6.29 扩展行 dot 原型、包装层精简与 B split-C 复验

A 的 `H_res` 增加 `H_post` 行、B 的 `original_residual` 增加 `x` 行在代数上成立，但当前
Ascend 编译器无法 lowering 这些结构。A 分别尝试了 dot 结果归约抽取、二维 masked store、
目标指针选择和单一底层存储的规则/不规则偏移，均在 `TritonToLinalgIncubated` 或地址分析
阶段触发 MLIR 断言。B 的无 bias 原型尝试了通道循环后的 16×16 小归约和 masked store，
也在同一 lowering 阶段失败。上述原型全部撤回，没有加入额外预处理 kernel、Torch `cat`
或大临时张量；N=16 和有 bias 的数值路径未改变。

小 forward 的主机分项测量中，输入校验约 0.013 ms，连续输入上的五个扁平 view 约
0.015–0.017 ms，启动 key 构造约 0.008 ms。dot kernel 只使用指针和显式维度，因此 forward
和 backward 连续化后不再创建扁平 view，输出直接按最终形状分配。小 backward 融合 kernel
仍使用通用 kernel，但 stride 直接由 N/C 计算，输入校验完整保留。`(2,4,4,1024)` BF16
forward 从约 0.179–0.182 ms 降为 0.149 ms；同一进程旧/新 backward 对照中，无 bias 从约
0.200 ms 降为 0.187 ms，有 bias 从约 0.314 ms 降为 0.295 ms。

新增 `_launch_fixed_tiles` 在每次 API 调用内只准备一次参数特化和设备信息，逐批仅替换
`tile_offset/grid`；它不跨调用缓存数据地址，也不复用返回张量。32 批提交的主机时间约从
2.3–2.5 ms 降为 2.0–2.1 ms，但 NPU Event 基本不变，说明大形状仍由设备工作主导，不能把
主机入队收益直接计为 kernel 加速。

对 `(128,1,8,7168)` 独立试验 B split-C=2/4 + FP32 partial/merge，正确性均通过，但第二次
启动和 merge 抵消了并行收益。BF16 无 bias 单阶段约 0.073 ms，split2/4 约为 0.107/0.140 ms；
FP32 无 bias 从约 0.077 ms 增至 0.104/0.142 ms，有 bias 同样退化，因此没有接入公开路径。

新增 C=512 的正式用例还暴露出冷缓存首次编译时 BF16 无 bias B 的 `grad_h_post` 出现 NaN；
使用已有磁盘缓存或完整顺序回归可能掩盖该问题。原因是 C=512 仍强制 BLOCK_C=1024 的配置，
全新 Triton 缓存可以稳定复现。B 的 BLOCK_C 现裁剪到不超过 C 的 next-power-of-two，C=512
不再进行 K 填充，同一冷缓存用例通过。其他标准 C>=1024 配置不变。

最终代码完整正确性为 **95 passed**（27.41 秒），全新缓存的 C=512 单项也通过。完整 96 点
性能 `2 passed`，耗时 305.05 秒。算术平均 speedup 为 forward **0.871x**、
backward **0.970x**、整体 **0.920x**；上一版对应值为 0.774x/0.801x/0.787x。FP32
backward 达到 **1.023x**，但整体仍未达到 1.0。小 backward 无 bias 多数代表点达到
1.1–1.2x，与旧/新包装定向对照一致；不同轮次的 baseline 和设备状态仍会影响均值，因此
不能把全部均值变化都归为某一项 kernel 优化的收益。

### 6.30 标量 B 分批、冷缓存非整齐 C 与 sb=1 bias 融合

通用标量 Backward B 原来直接以 `grid=(sb,)` 启动，合法输入 `(65536,1,4,65)` 会超过
65535 的 grid 上限。昇腾包装层现将该路径按 4096 个 batch 行分批，批间通过连续视图移动
输入和输出基址，并传入实际 `batch_sb`；常规 sb<=4096 仍只有一次启动且不增加 view。
上述 65536 行边界用例已验证五个梯度，不再越界。

新增 `(1367,1,4,4097)` BF16 forward/backward 两种 bias 回归。BF16 每行三个通道块，总计
4101 个 tile，覆盖第二批、不足 4096 的尾部和通道尾块，四项均通过。C=513 的新增测试进一步
发现 BF16 有无 bias 的 `grad_h_post` 都会出现 NaN，说明 6.29 的 C=512 修复不能覆盖所有
大块填充情况。B 的 BLOCK_C 现改为不超过 C 的 2 的幂：C=513 使用 512+1 两轮，C=768
使用 512+256 两轮，C>=1024 的标准配置不变。C=513/768 有无 bias 四项在独立进程、全新
Triton 缓存下全部通过。

大形状尾块独立处理的原型保持主体和尾部总启动次数不增加，但两轮最大 BF16 forward 旧/新
分别约 27.49/27.47 ms，A 约 27.43/27.46 ms，均在波动范围内且主机开销略增。减小逻辑块
没有转化为实际收益，已完整撤回，没有保留通道基址参数或额外启动计划。

sb=1 且有 bias 时，`grad_bias[c] = grad_x[0,c]`。新增昇腾专用 `_triton_hpb_bwd_small_bias1_kernel`
在小融合计算 `g_x_acc` 后，以相同 dtype 转换同时写 `grad_x` 和独立分配的 `grad_bias`，跳过
后续 bias 归约启动；没有共享输出地址或改变舍入路径。BF16 `(1,1,1,1)`、`(1,1,4,130)`、
`(1,1,4,1024)` 从约 0.281–0.284 ms 降为 0.188–0.191 ms，FP32 降为约 0.198–0.208 ms。
三种 dtype 的 N=1/2/4 定向测试均通过，并检查两个输出数值一致且 `data_ptr` 不同。
最终完整正确性为 **107 passed**（115.96 秒）。完整 96 点性能 `2 passed`，耗时 134.43 秒，
算术平均 speedup 为 forward **0.838x**、backward **0.961x**、整体 **0.900x**。sb=1
有 bias 的 BF16/FP16/FP32 点分别达到 **1.387x/1.448x/1.403x**。本轮没有修改 forward，
但其均值仍从上一轮 0.871x 变为 0.838x，最大 BF16 forward 的 Train 时间也从约 27.5 ms
变为 30.7 ms；大 backward 同类波动同样存在。因此整体均值低于上一轮 0.920x 不能归因于
sb=1 融合，定向旧/新对照和该点的正式 benchmark 才反映其实际收益。

### 6.31 拆分分支 NPU Event、sb=2 与 dot A bias 融合

`fused_h_post_bda_ascend` 分支基于拆分后的通用 forward/backward 文件，将昇腾实现和测试也拆成
两个文件。两个 benchmark 恢复 Ascend 专用 NPU Event：每个实现先执行 5 次 warm-up 并同步，
再记录 30 次调用的设备时间，baseline 和候选实现口径一致。`get_latency` 显式接收并消费
`_reference_candidate`，不会把框架参数传给算子；其他 vendor 仍调用通用实现。Ascend 不执行
Megatron fused Triton 探测，避免 CANN 编译停滞，组合 baseline 仍先尝试 `torch.compile`，
当前环境因 `triton_key` 接口不兼容回退 eager。

小融合路径在原有 sb=1 基础上扩展到 sb=2、N<=2。此时 BLOCK_S=2，同一 program 已持有两行
`g_x_acc`，直接执行 `g_x_acc -> 输出 dtype -> FP32 -> 行求和` 写入独立 `grad_bias`，
与先写 `grad_x` 再做归约的舍入路径一致。sb=1 保持原来的单行直写分支。三种 dtype、
C=65/130/1024 的完整 API 均有收益：BF16 C=65 约从 0.236 ms 降为 0.183 ms，C=130 从
0.232 ms 降为 0.181 ms，C=1024 从 0.236 ms 降为 0.186 ms；FP32 C=65 从约 0.260 ms
降为 0.211 ms。sb=1 的 BF16/FP16/FP32 对照约为 0.184/0.184/0.189 ms，未见回退。

dot A 在 sb=1 且有 bias 时也同时写独立 `grad_bias`。`g_x` 先转换为目标 dtype，各 program
只写自己负责的通道块，不修改 dot 或输入精度，也不增加预处理 kernel。N=8、C=512/2048/
4097/7168、非连续输入和三 dtype 定向测试全部通过。多数完整 API 对照从约 0.32 ms 降为
0.26 ms；BF16 C=512 从约 0.460 ms 降为 0.390 ms，C=7168 从约 0.444 ms 降为 0.364 ms。

N=4 多 sb 块对角 forward 原型没有进入正式分发。编译产物中 NP=4/NP=16 的 workspace 分别为
8192/32768，共享缓冲均为 221184；NP=16 实测未呈现 4 倍耗时，因此进行了四行块对角试验。
原型完全通过加载地址和 mask 构造，没有 `cat` 或预处理 kernel，但首次数值检查失败；将地址
简化为 `pid_group` 基址加组内偏移后，单位矩阵输入仍产生错误结果，说明当前编译器不能可靠
lowering 该结构。原型已完整撤回，没有残留实现，也没有根据理论填充收益宣称加速。

正式新增的 25 项 bias 和非连续测试全部通过；另有 12 个代表用例在独立进程、空 Triton 缓存
下通过，覆盖三种 dtype、sb=2、dot A、C=513 历史问题和非连续输入。最终完整正确性为
**129 passed**（228.92 秒）。恢复 NPU Event 后的完整 96 点性能为 `2 passed`，耗时 242.37 秒，
算术平均 speedup 为 forward **0.823x**、backward **0.985x**、整体 **0.904x**，FP32
backward 为 **1.024x**。sb=2 有 bias 的 BF16/FP16/FP32 点分别达到 **1.410x/1.389x/1.426x**；
sb=1 对应点为 **1.425x/1.408x/1.404x**，与定向“未见回退”的结论一致。Forward 未修改，
均值仍随轮次波动，不能把整体均值变化全部归因于本轮 bias 优化。

### 6.32 NPU 时间分解、B NP 自适应与双通道块分组

使用独立 NPU Event 分解最大 FP16 backward：A 约 30.6 ms、B 约 14.2 ms、完整 API 约
45 ms，主机提交约 2 ms，说明大形状主要是设备工作。A 的 NP 从 16 改为 8/4 没有收益，但
B 的 NP=4 降到 8.82 ms，约快 38%；N=8 使用 NP=8 也有稳定收益。因此仅将 B 的 NP 改为
`max(4,N)`，A 保持原配置。中等形状同样有效，C=512/513/768 的空缓存和完整 129 项测试
均通过。该轮 96 点均值为 forward 0.864x、backward 1.008x、整体 0.936x。

Forward 的算法优化来自通道循环结构而不是继续调整 NP。整行 C 循环只加载一次
`H_res/H_post`，小/中 sb 能减少重复 dot 初始化，但最大 sb=32768 时从约 30 ms 退化到
37.6 ms。双通道块分组在输入复用和并行度之间更平衡：最大 BF16 forward 从约 30.7 ms 降为
15.5 ms，FP32 从约 53.7 ms 降为 31.0 ms。最终 `N<=8、C_BLOCKS>1` 时，sb<=8192 使用整行
kernel，更大 sb 使用双块 kernel，单块 C 保持原 tile 路径。

Backward A 采用同样的双块结构，最大 BF16/FP16 A 从约 30.6 ms 降为 15.3 ms，FP32 从
53.6 ms 降为 30.7 ms；sb=2048 也约快 2 倍。A 在所有多通道块 dot 路径使用双块，B 保持
`max(4,N)` 和已有 C 循环。完整 96 点均值进一步提升为 forward **1.189x**、backward
**1.180x**、整体 **1.185x**，其中 FP32 forward 为 1.339x、FP32 backward 为 1.269x。

另外排除了三种结构：增广 K 维 rank-1 dot 在 MLIR lowering 阶段失败；直接使用 dot
accumulator 与现有实现交替测量没有稳定差异；多 sb 块对角加载在单位矩阵输入下就产生错误
结果。它们均未保留。当前接入版本新增 forward row/grouped 回归，完整正确性为 **137 passed**；
row、NP 和双块的独立空缓存分别为 9/9/5 项通过，完整性能 `2 passed`。对应实现提交为
`32f9d02 perf(megatron): improve Ascend h_post bda tiling`。

### 6.33 昇腾 Megatron 原生 Triton baseline 与固定配置性能分析

拆分分支后来恢复了对当前昇腾 Megatron 包的实际探测。当前环境安装的
`megatron/core/fusions/fused_mhc_kernels.py` 中，forward、backward A 和 backward B 都是真正的
`@triton.jit` kernel，并带有 `@triton.autotune`，不是 `torch.compile` 或 eager 组合实现。对应
benchmark 直接导入 `_triton_hpb_fwd_kernel`、`_triton_hpb_bwd_g_x_orig_kernel` 和
`_triton_hpb_bwd_g_hp_hr_kernel`。因此，新的 baseline 与此前 eager 组合 baseline 不是同一个比较
对象，两轮 speedup 不能直接横向比较。

不能将平台差异描述为“昇腾使用 Triton，而 NVIDIA 固定使用 torch.compile”。Megatron 在 NVIDIA
上同样提供该算子的融合 Triton 路径，当前 benchmark 在非昇腾平台也优先调用
`_triton_h_post_bda_fwd/_bwd`。Torch 组合加 `torch.compile` 是 native/fallback 路径，用于未启用
融合算子或融合 kernel 不可用时；它不是 NVIDIA 平台唯一或固定的执行方式。两边的发行包和后端
适配可以不同，但这里比较的 h_post_bda Triton kernel 来源相同。

Ascend 没有直接执行 Megatron 的完整 autotune，而是绕过 Autotuner，以 `.fn[grid]` 调用同一
JITFunction。Forward 和 backward A 固定使用 `BLOCK_C=512、BLOCK_S=8、num_warps=4`，必要时增大
`BLOCK_S` 使总 coreDim 不超过 65535；backward B 固定为 `BLOCK_C=64、BLOCK_S=1、num_warps=4`。
这里改变的是配置选择和启动方式，没有改写 Megatron kernel 的数值计算。所谓“autotune 有问题”
具体包括以下三类，不应笼统理解为 autotune 在所有昇腾形状上都不可用：

1. **调优耗时和停滞不可控**：每个 kernel 的原始搜索空间为 `4 种 BLOCK_C × 4 种 BLOCK_S ×
   3 种 num_warps = 48 组`，每组都可能涉及首次编译和测量。历史实验中 forward/backward 完整调优
   分别记录过约 186.68/392.14 秒；后续独立验证中，BF16 `(128,1,8,7168)` forward 超过 8 分钟
   仍未完成首次调用，只能停止该测试进程。
2. **部分候选配置的 grid 超过启动上限**：最大形状 `sb=32768、C=7168` 在 `BLOCK_S=1、
   BLOCK_C=64` 下，grid 为 `(32768,112)`，总 coreDim 为 `32768×112=3,670,016`，超过当前
   Ascend 启动路径的 65535 上限。Autotuner 会根据当前配置改变 grid，因此不能只检查最终选中
   配置；搜索过程中也可能遇到非法配置。
3. **Backward B 的大块配置发生 UB 溢出**：`BLOCK_C=512、BLOCK_S=8` 的配置需要
   2,478,080 bits，而当前设备 UB 容量为 1,572,864 bits。该配置不能进入昇腾合法候选集合。

因此，正式 benchmark 采用固定合法配置是兼容性处理，不代表 Megatron 官方为该算子规定了这个
固定配置，也不代表完整 autotune 选出的最佳性能。

完整 96 点均使用相同的 NPU Event 口径，预热 5 次、正式测量 30 次，首次编译在正式测量之外；
测试结果为 `2 passed`，总耗时 1044.66 秒，首次编译占主要部分。固定配置 Megatron baseline 的
统计如下：

| 方向 | Baseline 算术平均耗时 (ms) | Ascend 算术平均耗时 (ms) | 算术平均 Speedup | 最低/最高 Speedup |
| --- | ---: | ---: | ---: | ---: |
| Forward，48 点 | 202.554 | 2.732 | 59.783x | 1.208/237.447x |
| Backward，48 点 | 244.968 | 4.047 | 38.838x | 1.411/162.261x |
| 整体，96 点 | 223.761 | 3.390 | 49.311x | 1.208/237.447x |

按 dtype 分组的平均 speedup 为：forward BF16/FP16/FP32 分别 88.124x/88.950x/2.276x，backward
分别 53.878x/55.761x/6.877x。上述数字只表示“当前 Ascend 实现相对固定配置 Megatron Triton
kernel”的收益，不能解释为相对 Megatron 在昇腾上最佳合法配置的收益，也不能与第 6.32 节基于
eager baseline 的 1.185x 直接比较。

为区分主机提交、kernel 计算和配置选择，对 BF16 `(128,1,8,7168)` 做了定向分解。完整 API 使用
30 次 NPU Event；配置扫描使用已分配输出的独立 kernel 和 10 次 Event，不属于新的完整 96 点结果。
Forward 原始 `.fn[grid]` 与缓存底层直启分别为 37.509/37.507 ms，主机提交仅约 0.213/0.173 ms，
Ascend 完整 API 约 0.155 ms。两种启动方式几乎相同，说明大形状 bottleneck 不是裸 JIT 参数绑定或
Python 启动。

Backward 无 bias 的同一形状中，完整 Megatron baseline 为 42.872 ms，A 为 37.869 ms，B 为
4.794 ms，Ascend 完整 API 为 0.267 ms。A 是主要瓶颈，B 次之；即使完全去掉 bias 的
`g_x.sum(dim=0)`，A+B 已占据几乎全部时间，因此 bias 归约不是高 baseline 耗时的主因。

固定配置本身也明显偏离较优点。同一 BF16 forward 形状的定向扫描得到：

| `BLOCK_C/BLOCK_S/num_warps` | Megatron kernel 时间 (ms) |
| --- | ---: |
| 当前固定 `512/8/4` | 约 37.51 |
| `256/1/2` | 6.916 |
| `512/1/2` | 3.126 |

仅把 `BLOCK_S` 从 8 改为 1，在其他维度基本相同的配置下就获得约 12 倍改善。大 `BLOCK_S` 增加单
program 的 FP32 工作块、广播和寄存器压力；小形状还会降低 grid 并行度，例如 `(2,4,4,1024)` 的
固定 forward grid 只有 `(1,2)`。因此当前固定配置是规避编译、grid 和 UB 问题的保守选择，不是
性能最优配置。

原始 Megatron kernel 的计算结构也不同于 Ascend 实现。Forward 和 backward A 使用静态循环、标量
权重与 FP32 向量乘加，没有使用 `tl.dot`；forward 对 original residual 的加载随 `N²` 增长，
backward A 对 `go` 的逻辑加载次数为 `N+N²`。BF16/FP16 输入也会先转成 FP32 再做这些标量乘加。
Ascend forward/A 使用 `tl.dot` 和 FP32 accumulator，减少了重复加载并利用矩阵计算路径，这与
BF16/FP16 的 speedup 显著高于 FP32 一致。Megatron B 在 `C=7168、BLOCK_C=64` 时需要 112 轮通道
循环，并混合大量逐对 `tl.sum` 和 `tl.where`；Ascend B 使用 `BLOCK_C=1024` 和
`tl.dot(orig, trans(go))`，循环降为 7 轮。

仓库中的其他“Megatron baseline”不能作为该异常耗时的直接反证。`fused_gated_norm` 和
`fused_pad_routing_map` 只在 NVIDIA 使用原生 Megatron kernel；不少激活类 baseline 使用 Megatron
的 `torch.compile` 封装；直接使用 Triton 的测试通常走通用 `do_bench_npu` 设备任务计时，部分还
设置 `TRITON_ALL_BLOCKS_PARALLEL=1`。当前 h_post_bda 的 NPU Event 则覆盖完整 API。计时口径确实
不同，但原始启动与底层直启、完整 backward 与 A/B 单项的对照已经表明，主要差距来自 kernel
计算和固定配置，而不是计时器单独造成的假象。

随后直接调用 Megatron 官方 `_triton_h_post_bda_fwd/_bwd` 封装做了独立进程验证，不通过 `.fn`
绕过 Autotuner。BF16、有 bias 的 `(1,1,2,8)` forward 首次调用和完整调优约 7.16 秒，选中
`BLOCK_C=64、BLOCK_S=1、num_warps=4`，调优后 30 次 NPU Event 的完整 API 均值约 0.238 ms；
同形状 backward 首次调用约 15.47 秒，A/B 分别选中 `64/1/4` 和 `256/1/4`，调优后完整 API 约
0.504 ms。Forward 及 backward 五个梯度均通过 BF16 数值对照，说明官方 autotune 在小形状上确实
可用，且选中配置不同于 benchmark 的固定配置。BF16 `(128,1,8,7168)` forward 的官方完整调优
超过 8 分钟仍未完成首次调用，随后停止该独立进程；这不证明它永远无法完成，但说明未裁剪的
48 组配置不适合直接用于常规大形状 benchmark。该实验没有修改正式 benchmark。

后续若需要更公平地评价“对 Megatron 原生 kernel 的加速比”，应在不超限且不溢出 UB 的配置集合
内预先调优或建立按形状查表的配置，编译和调优均放在正式计时之外；B 应使用独立的合法配置集合。
重新测量后，speedup 很可能显著低于 49.311x，但这才是对 Megatron 在昇腾上合理性能的更有解释力
的比较。当前 49.311x 应保留为固定配置实验结果，不应作为最终端到端收益宣传。

### 6.34 完整 autotune 的 grid 容错与耗时复核

进一步直接调用 Megatron 官方 `_triton_h_post_bda_fwd/_bwd`，验证了
`TRITON_ALL_BLOCKS_PARALLEL=1` 对 coreDim 问题的作用。未设置该变量时，BF16
`(512,4,4,4096)` forward 的首个 `BLOCK_C=64、BLOCK_S=1` 配置产生 `coreDim=131072`，以普通
`RuntimeError` 中断调优；设置后，48 组配置全部完成，耗时约 7.25 秒，最终选中 `64/1/4`。
该变量是把逻辑 block 自动映射到物理核，使大 grid 可以运行，不是捕获异常并跳过配置。
UB 溢出则包装为 `MLIRCompilationError`，正常 Autotuner 会将其记为 `inf` 并跳过；只有通过
`.fn` 绕过 Autotuner 时才会直接使调用失败。

实验期间曾将环境变量严格限制在 Megatron baseline 调用内，返回或异常后恢复原值，并实际确认
FlagTrain 昇腾实现运行时没有继承该变量。最终方案已改用 NVIDIA base-data，不再保留这段环境
变量逻辑。

完整 autotune 的“完整”指 48 组 block 配置全部参与搜索，不是 shape 数量多。Megatron 的
key 为 `C、N`，Triton 额外加入 dtype，不包含 `sb` 和 bias；forward 与 backward A/B 的缓存
彼此独立，默认也不跨进程复用最佳配置。两个唯一形状、三种 dtype 会形成 6 个 key，对应
forward 288 次、backward 576 次配置测速；模块探测再增加 144 次。每个配置的 `do_bench`
还会重复执行，因此两个形状也可能产生很长的首次调优时间。

本轮实测的典型耗时如下：

| 方向与形状 | dtype | 调优耗时 |
| --- | --- | ---: |
| Forward `C=7168、N=8` | FP16 | 973.82 秒 |
| Forward `C=7168、N=8` | FP32 | 1603.61 秒 |
| Backward A `C=7168、N=8` | BF16 | 1186.63 秒 |
| Backward A/B `C=1024、N=4` | FP16 | 365.34/668.51 秒 |
| Backward A/B `C=4096、N=4` | FP16 | 371.30/665.62 秒 |
| Backward A/B `C=1024、N=4` | FP32 | 262.63/630.14 秒 |

缩小 `sb` 和 `C` 均没有稳定改善。小 `sb` 的 `C=1024/4096` forward 每组仍需约 246–311 秒；
进一步缩为 `C=256/512` 后，BF16 两组仍分别需要 303.88/311.77 秒。这证明主要成本是每个
新 `(C,N,dtype)` 的完整编译、分发和配置测速，而不是正式形状的计算量。剔除大 `C` 和缩小
shape 的实验均已撤回，未将这些不完整的长跑结果作为最终加速比。

### 6.35 昇腾改用 NVIDIA base-data 与 backward 数据缺口

昇腾代码分支已 rebase 到 `origin/main` 的 `6e1c2ca`。最新框架允许 `torch_op=None`：不执行本机
baseline，`--base-data` 成为唯一比较来源。提交 `cd9a0b7` 仅对昇腾启用该路径，不调用
`_load_baseline`，不导入或探测 Megatron kernel，也不再设置 `TRITON_ALL_BLOCKS_PARALLEL`。
其他 vendor 保持原有 Megatron/Torch baseline，shape 恢复为通用 benchmark 的完整列表。

NV 折算公式为：

```text
compared_speedup = (base_ms / train_ms) / factor
factor = vendor_peak / reference_peak
```

`base_ms` 和瓶颈资源来自 `tools/op_flagtrain_perf_baseline.json`，factor 按当前芯片、dtype 和
baseline 瓶颈资源查找。输出中的 `Torch Latency` 和本地 `Train Speedup` 为 `N/A`，有效比较值
为 `vs Base`。昇腾候选实现仍使用独立 NPU Event，预热 5 次、正式测量 30 次。

全量 pre-commit 和 **137 passed** 正确性均通过，导入拦截检查确认昇腾没有探测 Megatron。
Forward 的 48 个核心加综合点全部匹配 NV 记录并通过，耗时 46.20 秒。Backward 前 4 个 BF16
case 的 `vs Base` 分别为 0.527/0.463/0.506/0.655，但第 5 个 `(128,1,8,7168)` 因 NV 数据缺失
而按框架设计失败。

该失败不是 `baseline/` 接口漏写 shape。`baseline/megatron/fused_h_post_bda_forward.py` 和
`fused_h_post_bda_backward.py` 都定义了完整 16 个 shape/bias case，与通用 benchmark 一致。
当前 JSON 的 forward 有 16 个 case，backward 却只有前 4 个，缺少从 `(128,1,8,7168)` 开始的
后续 12 个 case。`baseline/README.md` 说明这些接口应原样同步到外部 `ops-baseline` 仓库并生成
数据；JSON 不是本地 FlagTrain benchmark 自动生成的，本次没有手工修改 JSON。

因此，当前代码路径已经完成，但 backward 全量 NV 折算仍依赖外部重新采集完整数据。应重点检查
外部采集在第 5 个 `(128,1,8,7168)` 处的日志，并在 JSON 的 backward 也达到 16 个 case 后再
运行全量性能测试。不能通过伪造 NV 延迟或删减通用 shape 绕过该缺口。

### 6.36 测试参考接口统一与昇腾精度参考实测

昇腾分支随后 rebase 到 `origin/main=3f1ffd6`，提交 `1a945df` 统一了拆分后的 forward/backward
测试结构。Benchmark 不再携带 Torch 组合实现：非昇腾平台尝试导入 Megatron 原生接口，失败则令
`torch_op=None` 并依赖 NV base-data；昇腾不探测，直接使用 NV 折算。功能测试在非昇腾平台优先
使用原生 Megatron，导入失败时回退 Torch 组合；NV 专属用例额外比较有原生接口与 Torch 组合在
有无 bias 下的结果，当前昇腾环境未执行这两个 NV 用例。

为评估昇腾功能测试能否同样使用原生参考，曾在完整前后向套件中启用原生 Megatron。运行命令为：

```bash
pytest -s -v tests/megatron/test_fused_h_post_bda_forward.py \
  tests/megatron/test_fused_h_post_bda_backward.py --durations=20
```

pytest 总耗时 **3639.46 秒（60 分 39 秒）**，墙钟时间 **3663.08 秒（61 分 3 秒）**，结果为
**42 passed、95 failed、4 skipped**。第 9 项 BF16 `(1,1,8,257)` forward 不是永久卡死，而是
300.02 秒后才完成；相同 `C=257、N=8` 的 FP16/FP32 no-bias 用例分别耗时 603.49/452.32 秒。
BF16/FP16 medium forward 也分别耗时 219.19/224.40 秒。主要成本仍是每个新 `(C,N,dtype)` 的
48 组配置编译和测速，而不是精度比较本身。

原生 forward 在 `(1367,1,4,4097)` 还选择了产生 **coreDim=88855** 的配置，超过昇腾 65535 上限。
此后 NPU Event 出现 `event recorder null`，后续大量失败是设备状态异常后的连带失败，不能解释为
95 项独立数值错误。鉴于近一小时的耗时和原生 grid 风险，最终昇腾精度测试也不探测原生
Megatron，直接使用 Torch 组合参考；其他平台仍保持原生优先、缺失回退。该路径全量 pre-commit
通过，功能测试为 **137 passed、4 skipped，3.69 秒**，导入拦截确认没有访问 Megatron。

### 6.37 当前 NV 折算性能与后续实验范围

在 rebase 到 `3f1ffd6` 之前的工作区，按第 6.35 节口径重新实测了当前昇腾实现。Forward 48 点
全部成功，各点 `vs Base` 的算术平均为：BF16 **0.309x**、FP16 **0.532x**、FP32 **0.268x**，
总体 **0.370x**。FP16 均值包含 4.249x 和 1.399x 两个使用不同折算系数的高点，不能将均值
理解为所有形状的典型收益。Backward 只有 BF16 前 4 点可用，`vs Base` 为
**0.552/0.477/0.507/0.705**，平均 **0.560x**；第 5 点仍因外部 NV 数据缺失而停止，不能给出
完整反向均值。

小形状单次完整 API 约 0.14–0.16 ms，而大 BF16 `(8192,4,4,7168)` forward 约 14–16 ms，说明
启动开销和设备计算需要分开处理。后续受控实验限定为主机提交/设备时间分项计时、连续输入启动
快路径，以及编译期 `EVEN_C` mask 特化。输出和调用方指针不缓存复用，非连续输入保持支持；
不再重复历史上失败的扩大 grid、4096 通道块、扩展行 dot、B split-C 和扩大 bias split。

### 6.38 启动分项计时、失败原型与 device context 快路径

本轮首先在同一进程中对 30 次完整 API 分别测主机提交、同步后墙钟和 NPU Event。代表性结果为：

| 方向与形状 | Bias | 主机提交 (ms) | NPU Event (ms) |
| --- | --- | ---: | ---: |
| Forward `(2,4,4,1024)` | 无 | 0.141946 | 0.141783 |
| Forward `(2,4,4,1024)` | 有 | 0.140880 | 0.143641 |
| Backward `(2,4,4,1024)` | 无 | 0.171181 | 0.171913 |
| Backward `(2,4,4,1024)` | 有 | 0.280628 | 0.271699 |
| Forward `(8192,4,4,7168)` | 无 | 0.733778 | 15.537602 |
| Backward `(64,8,4,4096)` | 无 | 0.228729 | 0.260551 |

小形状 Event 与主机提交基本一致，说明主机路径是主要瓶颈；大 forward 的设备时间约为主机提交的
21 倍，启动优化不能解决其主要差距。微基准中输入校验约 12.26 μs、一次输出 `torch.empty` 约
6.23 μs、device context 约 4.85 μs；5 次连续张量 `.contiguous()` 合计约 0.94 μs。

连续输入通过 `is_contiguous()` 避免 `.contiguous()` 的原型没有可靠收益。不同进程的主机计时虽有
约 10 μs 变化，但远超微基准可解释的范围，不能排除进程和频率波动，已撤回，没有保留该复杂分支。

`EVEN_C` 原型在 forward 和 backward A 中按编译期整除条件省略通道 mask。首次性能运行还暴露了
Ascend Triton 的约束：`mask=None` 时不能传 `other=0.0`，修正后可以继续实验。冷编译完整回归耗时
620.48 秒，结果为 136 passed、4 skipped、1 failed；失败项是 FP32 `(1,1,8,7168)` backward 的
`grad_h_res`，7163/57344 个元素不匹配。检查实际差异发现 B 的内层 mask 并未启用该特化，因此不能
把此失败直接解释为 `EVEN_C` 的数学问题，但它反映了当前工具链对冷编译/缓存状态的敏感性。恢复
已推送实现后，该用例单独通过，完整套件恢复为 **137 passed、4 skipped**。

性能上，`EVEN_C` forward 没有稳定改善，backward 已有 4 点变为 0.531/0.460/0.506/0.687，低于
原实现的 0.552/0.477/0.507/0.705。综合数值风险和收益，forward/A 的实验代码也全部撤回，正式
分支不保留 `EVEN_C` 参数。

最终仅保留提交 `ba89ea8` 的 device context 快路径：当前 device 与输入一致时使用 `nullcontext`，
跨 device 仍调用原 `torch_device_fn.device`。该校验、输出独立分配、stream 查询和非连续输入处理均
保留，不缓存调用方指针或返回张量。定向 100 次主机提交计时中，小 forward 无 bias 从约
0.145 ms 降至 0.132–0.134 ms，小 backward 无 bias 从约 0.173 ms 降至 0.163–0.165 ms。

完整 NPU Event 对照如下。总体均值没有提高，保留它的依据是小形状主机路径改善，而不是大形状
或跨轮次均值变化：

| 指标 | 原实现 | device context 快路径 |
| --- | ---: | ---: |
| Forward BF16 16 点均值 | 0.309437x | 0.315437x |
| Forward FP16 16 点均值 | 0.532312x | 0.526125x |
| Forward FP32 16 点均值 | 0.267500x | 0.267313x |
| Forward 全 48 点均值 | 0.369750x | 0.369625x |
| Backward 已有 BF16 4 点均值 | 0.560250x | 0.570750x |

最终全量 pre-commit 通过，冷编译完整精度回归 **137 passed、4 skipped**。Backward 全量性能仍受
第 6.35 节的外部 NV 数据缺口限制。没有引入 AOT/C++ 启动层；当前小形状收益不足以证明这类额外
复杂度的必要性，大形状仍需基于生成代码和设备时间线寻找计算侧突破。

### 6.39 其他算子 NV 折算对照与 forward 分组突破

为判断当前低加速比是否主要来自其他算子的优化方法，在同一昇腾环境补测了三个代表算子。仓库的
NV JSON 只保存 H800 原始延迟，不包含这些算子的历史 Ascend `vs Base`，因此下表是本轮实测：

| 算子 | 总体平均 `vs Base` | 大形状表现 | NV 基线 kernel 数 |
| --- | ---: | ---: | ---: |
| h_post_bda forward | 0.370x（48 点） | 约 0.24–0.26x | 1 个融合 Triton kernel |
| h_aggregate forward | 2.845x（20 点） | 0.144–0.293x | 2 个 Torch kernel |
| fused_gated_norm | 2.523x（40 点） | regular 约 0.756–1.194x | 1 个融合 Triton kernel |
| fast_gelu | 21.131x（21 点） | 约 3.275–6.995x | 9 个 Torch kernel |

这些均值不能直接横向解释。h_aggregate 的高均值来自极小形状，大形状并不比 h_post_bda 好；
fast_gelu 是单遍单 kernel 对 NV 的 9 个 Torch kernel；h_post_bda 则已经面对 NV 融合 kernel。
此外，其他算子使用通用 `do_bench_npu` 设备任务计时，h_post_bda 使用完整 API 的 NPU Event，
小形状口径不同。多数点使用相同的 mem 折算系数 0.389412，差异不是由 h_post_bda 单独受到系数
惩罚造成的。

fused_gated_norm 的 persistent 和单遍归约确实更有效。其大形状实测约 795 GB/s；当前最大 BF16
h_post_bda forward 修改前约 304–330 GB/s。要达到该形状 NV 折算 1.0x，约需 3.71 ms、1267 GB/s，
说明仍存在真实的设备效率差距，不能只靠启动优化解决。

本轮先试验 row-dot 通道循环 `num_stages=2`。最大 BF16 bias 从 14.761 ms 变为 15.876 ms，FP32
从约 29.55 ms 变为 31.05 ms，没有收益并撤回。随后保持 kernel 不变进行结构对照：

- 中小 `sb` 全部改用 grouped=2 时，FP32 `(512,4,4,4096)` 从约 0.616 ms 退化到 0.976 ms。
- 原始单块 dot 的 BF16 medium 约 0.975 ms、最大形状约 30.7 ms，均明显慢于现有复用路径。
- 最大 `sb=32768` 的 grouped=4 使 BF16/FP16 从约 15.45/14.76 ms 降至约 10.94/10.98 ms，
  FP32 降至约 20.14 ms。
- grouped=8 时 BF16/FP16 因空块循环退化到约 18.27 ms，FP32 则进一步降至约 17.08 ms。

因此提交 `c7ba7e9` 保留原有 `sb<=8192` row 路径，仅在大 `sb` grouped 路径按 dtype 选择
`C_GROUP=4`（BF16/FP16）或 8（FP32）。中小 `sb` 不改动。没有增加临时张量、预处理 kernel 或
Torch 原生计算，也没有改变矩阵乘精度。

正式 48 点 forward 结果为：BF16 0.300250x、FP16 0.592938x、FP32 0.304875x，总体
**0.399354x**；上一版总体为 0.369625x。BF16 均值仍受跨轮次小形状折算高点波动影响，应以最大
形状的旧/新耗时作为这项优化的直接证据。最大 BF16 正式点为 10.800/10.841 ms，FP32 为
17.081/17.081 ms。

Grouped 精度用例新增 FP32 `C=2049`，与原有 BF16 `C=4097` 一起覆盖 8/4 分组及有无 bias。
全量 pre-commit 通过，最终 **139 passed、4 skipped**。48 点均值仍未达到 0.8x；剩余难点是大量
小形状的完整 API 成本和 medium/大形状的浅小 dot 效率，不能把本轮单点约 30% 的设备收益写成
已达到目标。

## 7. 正确性测试设计

测试文件：`tests/megatron/test_fused_h_post_bda.py`。

### 7.1 参考实现

- Forward：独立的 PyTorch `bmm + 逐元素运算` reference。
- Backward：对 PyTorch reference 前向执行 `torch.autograd.grad`。
- 参考计算中的 `bmm`（含 autograd 反传）在 `_strict_fp32_matmul()` 下执行，临时关闭
  TF32：部分后端（如 MetaX）默认 fp32 matmul 走 TF32，参考值自身误差会超过 fp32 用例
  的 1e-5 容差，造成假失败。
- 使用 `--ref cpu` 时，通用 reference 输入会被独立复制到 CPU，确保参考计算真正在 CPU
  上执行，而不是只把结果移动到 CPU。
- 大形状 forward 专项使用独立 GPU PyTorch reference 后再复制预期结果，避免非连续大
  张量的 CPU reference 和比较使回归超时；通用 CPU reference 覆盖保持不变。
- 随机种子为 2026，随机输入范围为 `[-0.1,0.1]`。

### 7.2 参数化形状和 dtype

| dtype | 覆盖的 `(s,b,n,C)` |
| --- | --- |
| BF16 | `(1,1,1,1)`、`(1,2,2,65)`、`(2,1,4,130)`、`(2,3,4,130)`、`(1,1,8,257)`、`(2,5,8,257)`、`(513,1,4,130)` |
| FP16 | `(1,2,2,65)`、`(2,1,4,130)`、`(1,1,8,257)`、`(513,1,4,257)`、`(2,5,8,257)` |
| FP32 | `(1,2,2,65)`、`(2,1,4,130)`、`(1,1,8,257)`、`(513,1,4,257)`、`(2,5,8,257)` |

这些形状覆盖：

- 最小维度 `s=b=n=C=1`。
- `n=1/2/4/8`。
- 非整块通道 `C=65/130/257/4099`。
- 非整块 `s*b=2/6`。
- 大 `s*b=513`，以及 `s*b=10` 的非小形状 `N=8` 分支。
- 三种 dtype 下的 `n=2/4/8`。
- 三种 dtype 的输入复用路径，以及 BF16/FP16 的中等形状原路径。

每组通用形状分别测试 `bias=None` 和有 bias，forward 和 backward 分开执行。Backward
长通道专项由 BF16 和 FP32 覆盖 `(128,1,8,4096)`，每种 dtype 都测试有无 bias，检查纯
Triton B dot 的通道循环和五个梯度。dot 门槛专项由三种 dtype 覆盖 `(128,1,4,512)` 和
`(128,1,8,2048)` 及有无 bias，固定此前只记录在实验文档中的代表性形状。多批启动专项覆盖
`(1367,1,4,4097)` 两个方向和两种 bias；非整齐 B 块专项覆盖 BF16 的 C=513/768 和两种
bias。Forward 专项由三种 dtype 覆盖 `(64,8,4,4099)` 及有无 bias，验证中等形状分发和尾块。

### 7.3 特殊测试

除参数化用例外，还包括：

1. **固定 transpose 数值案例**：用固定小矩阵验证 `H_res.T`，防止把矩阵乘方向写反。
2. **非连续 forward 输入**：分别覆盖小形状和 `(64,8,4,4096)` 中等形状固定配置路径。
3. **非连续 backward 输入**：覆盖小融合和 `(128,1,8,512)` dot 路径，包含非连续 bias
   和 `grad_output`，逐项检查五个梯度。
4. **无 bias 返回约定**：验证 `grad_bias is None`。
5. **长通道 backward**：覆盖 BF16/FP32、有无 bias，检查五个梯度和 B 的 C 循环归约。
6. **标量 B grid 边界（昇腾专项）**：`(65536,1,4,65)` 验证 4096 行分批和实际批大小。
7. **sb=1 独立 bias 输出**：三种 dtype 检查 `grad_bias` 等于 `grad_x` 且地址不同。

### 7.4 容差

| dtype | Forward `atol/rtol` | Backward `atol/rtol` |
| --- | --- | --- |
| BF16 | 0.02 | 0.05 |
| FP16 | 0.001 | 0.005 |
| FP32 | 1e-5 | 1e-5 |

### 7.5 用例数量

- 17 组通用 dtype/shape 组合 × 2 种 bias 设置 × 2 个方向 = 68 项。
- 2 种长通道 backward dtype × 2 种 bias 设置 = 4 项。
- 3 种 forward dtype × 2 种 bias 设置 = 6 项。
- 3 种 backward dot dtype × 2 组形状 × 2 种 bias 设置 = 12 项。
- 多批启动 2 个方向 × 2 种 bias 设置 = 4 项。
- 非整齐 B 块 2 组 C × 2 种 bias 设置 = 4 项。
- 标量 B grid 边界 = 1 项。
- sb=1 独立 bias 输出 3 种 dtype = 3 项。
- 固定 transpose 案例 = 1 项。
- 非连续 forward = 2 项。
- 非连续 backward = 2 项。

**合计 107 项正确性测试。**

运行命令：

```bash
CUDA_VISIBLE_DEVICES=3 pytest tests/megatron/test_fused_h_post_bda.py \
  -m "fused_h_post_bda_forward or fused_h_post_bda_backward" --ref cpu -q
```

## 8. 性能测试设计

性能文件：`benchmark/megatron/test_fused_h_post_bda.py`。

### 8.1 Baseline 选择

非昇腾平台的 benchmark 在运行时实际探测 Megatron：优先导入并执行
`_triton_h_post_bda_fwd/_bwd`，导入或执行失败时回退到组合实现。昇腾固定使用组合
baseline，不探测 Megatron Triton；否则小 shape 探测可能命中已有编译缓存，而正式 shape
仍会在 48 组 autotune 配置中触发 `bishengir-compile` 卡死。这里描述原分支当时的策略；
拆分后的 `fused_h_post_bda_ascend` 分支后来改为直接运行 Megatron 原生 Triton kernel 的固定合法
配置，其性能结果和配置问题见第 6.33 节。

组合 baseline 先实际探测 `torch.compile`，失败时警告并回退到同一实现的 eager 版本。
测试会打印实际选择，避免把 PyTorch fallback 误报成 Megatron Triton。

当前 NVIDIA 环境选择 Megatron Triton baseline。Fallback 逐行对齐 Megatron
`native_h_post_bda` 和 `_torch_h_post_bda_bwd`，包括矩阵乘方向、view/reshape、bias 展开、
加法顺序和梯度 dtype 转换，并保留 `torch.compile` 以接近未使用融合 Kernel 时的真实
Megatron 执行方式。它显示为 `Megatron-native torch.compile (fallback)`，不是普通 eager
Torch 组合实现；当 `torch.compile` 不可用时显示为 `Megatron-native torch eager
(fallback)`，二者共享同一份实现代码。

### 8.2 计时口径

- NVIDIA 使用 GenericBenchmark 的 kernel 模式和 CUDA Event，预热 25 ms、正式测量 100 ms，
  取中位数。
- 本算子在昇腾 kernel 模式下使用 NPU Event，预热 5 次、正式测量 30 次并取平均值，不再进入
  可能返回空采样结果的 `do_bench_npu`，见第 6.17 节。
- 输入随机生成在计时区域之外。
- 首次 Triton 编译、autotune 和 fallback 的 `torch.compile` 编译不属于正式测量。
- 输出分配、`.contiguous()`、split-C partial 分配和 bias 梯度求和都包含在被测 API 内。
- Backward 直接传入 `grad_output` 调用 backward，**不经过 autograd，也不包含 forward
  耗时**。
- Speedup 定义为 `Megatron latency / FlagTrain latency`，大于 1 表示 FlagTrain 更快。

本轮优化还使用 CUDA Graph replay 和 Profiler 分段计时辅助定位 A、B、bias reduction 的
GPU 时间。Graph 数据只用于分析，不替代上述完整 API 的 48 点性能结果；普通 `do_bench`
和 Graph 的 L2 清理口径不同，Graph 还会展开多轮调用及输出分配，不适合不加限制地用于
大形状综合集。

### 8.3 核心形状

核心集包含三种 dtype：

- BF16
- FP16
- FP32

以及以下形状：

- `(2,4,4,1024)`
- `(64,8,4,4096)`
- `(128,1,8,7168)`

每个形状分别测试 bias 有无。

每个算子有 `3 dtype × 3 shape × 2 bias = 18` 个数据点，forward 和 backward 合计
**36 个核心数据点**。

### 8.4 综合附加形状

综合集额外包含：

- `(1,1,1,1)`
- `(1,2,2,65)`
- `(2,3,4,130)`
- `(512,4,4,4096)`
- `(8192,4,4,7168)`

核心和附加合计 8 组形状，均有 bias 和无 bias 各测一次：

- BF16 综合集：每个算子 16 点，两个算子共 **32 点**。
- 全精度综合集：每个算子 48 点，两个算子共 **96 点**。

当前最终的 forward 和 backward 结论分别来自各自 48 个全精度综合数据点。

### 8.5 运行命令

与最终 GPU 3 验证一致：

```bash
# Forward：默认 comprehensive + 三种 dtype
CUDA_VISIBLE_DEVICES=3 pytest -s -v benchmark/megatron/test_fused_h_post_bda.py \
  -m "fused_h_post_bda_forward"

# Backward：默认 comprehensive + 三种 dtype
CUDA_VISIBLE_DEVICES=3 pytest -s -v benchmark/megatron/test_fused_h_post_bda.py \
  -m "fused_h_post_bda_backward"
```

记录 JSON：

```bash
CUDA_VISIBLE_DEVICES=3 pytest -s -v benchmark/megatron/test_fused_h_post_bda.py \
  -m "fused_h_post_bda_forward" \
  --record json --output /workspace/forward.json -q

CUDA_VISIBLE_DEVICES=3 pytest -s -v benchmark/megatron/test_fused_h_post_bda.py \
  -m "fused_h_post_bda_backward" \
  --record json --output /workspace/backward.json -q
```

建议串行运行性能测试，不要让多个 autotune/benchmark 进程互相争用。

## 9. 当前性能结果

以下结果来自 NVIDIA H20-3e、BF16/FP16/FP32 全精度综合集，时间单位为 ms。Forward
对应当前中等形状固定配置版本；backward 数据对应 FP32 通用原生归约修改前的小形状融合和
N=8 split-C 版本。`1c32531` 之后 FP32 且 `C>=4096` 的 backward 尚未重新测量，下列
backward 统计和代表数据应视为历史结果，不能直接当作当前 FP32 长形状性能。

### 9.1 整体统计

| 算子 | 数据点数 | 最低 Speedup | 算术平均 Speedup | 低于 0.8x 的数量 |
| --- | ---: | ---: | ---: | ---: |
| Forward | 48 | 0.980x | 1.253x | 0 |
| Backward（历史） | 48 | 0.833x | 1.770x | 0 |

这里的平均值是各数据点 Speedup 的**算术平均**，不是按耗时加权，也不代表端到端训练加速比。
Forward 本轮优化前的平均值为 1.240x、最低值为 0.830x；backward 优化前分别为
1.171x 和 0.819x。历史 backward 最低点是 FP16 `(64,8,4,4096)` 无 bias 的 0.833x，
该形状继续使用单阶段 B。

历史 JSON 记录为：

- `/workspace/benchmark_fused_h_post_bda_forward_gpu3_medium_fixed_final.json`
- `/workspace/benchmark_fused_h_post_bda_backward_gpu3_fused_splitc_final.json`

### 9.2 Forward 代表性数据（BF16）

| Shape | Bias | Megatron (ms) | FlagTrain (ms) | Speedup |
| --- | --- | ---: | ---: | ---: |
| `(1,2,2,65)` | 无 | 0.006880 | 0.005216 | 1.319x |
| `(1,2,2,65)` | 有 | 0.006928 | 0.005248 | 1.320x |
| `(2,4,4,1024)` | 无 | 0.008096 | 0.005824 | 1.390x |
| `(2,4,4,1024)` | 有 | 0.007904 | 0.005664 | 1.395x |
| `(64,8,4,4096)` | 无 | 0.019360 | 0.018176 | 1.065x |
| `(64,8,4,4096)` | 有 | 0.019168 | 0.019296 | 0.993x |
| `(128,1,8,7168)` | 无 | 0.022496 | 0.019040 | 1.182x |
| `(128,1,8,7168)` | 有 | 0.024096 | 0.019616 | 1.228x |
| `(512,4,4,4096)` | 无 | 0.058560 | 0.049120 | 1.192x |
| `(8192,4,4,7168)` | 无 | 1.553120 | 1.276832 | 1.216x |
| `(8192,4,4,7168)` | 有 | 1.541664 | 1.240560 | 1.243x |

小形状主要收益来自消除固定 dispatch 开销；`N=8` 和大 `sb` 形状进一步从输入复用中
获益。BF16/FP16 的 `(64,8,4,4096)` 保持原计算结构，通过固定配置和绕过 Autotuner
将耗时由约 19.4–20.0 μs 降至 BF16 的 18.2–19.3 μs、FP16 的 16.8–18.5 μs。

### 9.3 Backward 代表性数据（BF16）

| Shape | Bias | Megatron (ms) | FlagTrain (ms) | Speedup |
| --- | --- | ---: | ---: | ---: |
| `(1,2,2,65)` | 无 | 0.056720 | 0.017472 | 3.246x |
| `(1,2,2,65)` | 有 | 0.066624 | 0.028000 | 2.379x |
| `(2,4,4,1024)` | 无 | 0.056704 | 0.049056 | 1.156x |
| `(2,4,4,1024)` | 有 | 0.067808 | 0.050848 | 1.334x |
| `(64,8,4,4096)` | 无 | 0.062624 | 0.074560 | 0.840x |
| `(64,8,4,4096)` | 有 | 0.072416 | 0.085120 | 0.851x |
| `(128,1,8,7168)` | 无 | 0.101312 | 0.083424 | 1.214x |
| `(128,1,8,7168)` | 有 | 0.126976 | 0.093792 | 1.354x |
| `(512,4,4,4096)` | 无 | 0.121280 | 0.115296 | 1.052x |
| `(512,4,4,4096)` | 有 | 0.143488 | 0.133216 | 1.077x |
| `(8192,4,4,7168)` | 无 | 2.930272 | 2.687840 | 1.090x |
| `(8192,4,4,7168)` | 有 | 3.486624 | 2.947968 | 1.183x |

本轮小形状融合显著降低了极小形状的完整 API 时间；N=8 split-C 将 `(128,1,8,7168)`
无 bias/有 bias 的 FlagTrain 时间从原来的 0.098240/0.125696 ms 降至
0.083424/0.093792 ms。N=4 中等形状保持单阶段 B，大 sb 路径仍主要受益于 backward A
输入复用。Bias reduction 没有改动。

## 10. 当前限制与后续方向

1. **验证平台有限**：当前系统验证集中在 NVIDIA H20-3e，其他 vendor 尚未完成同等覆盖。
2. **小 `n` 限制**：`n` 必须为正的 2 的幂，当前测试覆盖到 8。
3. **无完整 autograd 公共封装**：当前公开的是直接 forward/backward API。
4. **性能具有形状依赖性**：小形状重点优化固定开销，大形状不等于一定超过 Megatron。
5. **测量会受系统状态影响**：不同 GPU、并发进程和 autotune 状态会影响绝对耗时，应以
   同一测试内的 baseline 比值为主。
6. **Bias reduction 仍有优化空间但不宜盲目重写**：Torch 目前优于已尝试的 Triton 和
   atomic 方案。
7. **不适合直接推广 batched dot**：当前 `n<=8` 的典型场景下，Tensor Core 填充和寄存器
   开销抵消了理论收益。
8. **输入复用具有适用范围**：三维 accumulator 会增加寄存器压力，不能对所有形状默认
   启用；新增平台或重要形状后，应重新验证复用条件和配置裁剪规则。
9. **Split-C 不能只看 Graph 收益**：partial 分配和 merge launch 会影响完整 API；当前仅在
   非原生 FP32 的 N=8 受限形状启用，NVIDIA 使用 split-C=4，除昇腾外的通用后端使用 16；
   昇腾独立后端保留单阶段 B；其他 N 或更大的 sb 需要重新测量。
10. **中等形状 forward 配置是受限特调**：固定配置只用于 BF16/FP16 的
    `(64,8,4,4096)`；全流和双流复用曾出现完整 API 回退，不能推广到其他形状或 dtype。
11. **FP32 长归约依赖原生 PyTorch 算子**：该路径优先保证跨平台数值一致性，但会增加对平台
    BLAS/归约实现的依赖；Hygon hipBLAS 问题尚待实机最小复现，且需要补测修改后的性能。

## 11. 相关文件

| 文件 | 作用 |
| --- | --- |
| `src/flag_train/megatron/fused_h_post_bda.py` | 通用 Triton Kernel、输入校验和 forward/backward 包装 |
| `src/flag_train/runtime/backend/_ascend/megatron/fused_h_post_bda.py` | 昇腾纯 Triton dot、固定配置预绑定直启、大 grid 分批、sb=1 bias 融合及 FP32 bias 归约 |
| `src/flag_train/megatron/__init__.py` | 算子导出和 `SpecOpRegistrar` 注册 |
| `tests/megatron/test_fused_h_post_bda.py` | 107 项正确性测试，含跨批、grid 边界和冷编译敏感配置 |
| `benchmark/megatron/test_fused_h_post_bda.py` | Baseline 选择、输入生成和性能测试 |
| `fused_h_post_bda.md` | 本文档 |
