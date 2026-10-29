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

当前分支为 `fused_h_post_bda`。本文描述截至代码提交 `eeabbc4` 的实现和验证结果。

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

满足以下受限条件时，Backward B 使用 split-C=4：

```text
vendor == "nvidia"
且不属于小形状
且 n == 8
且 sb <= 128
且 C >= 4096
```

固定配置为 `BLOCK_C=256`、`BLOCK_S=2`、`num_warps=2`。以 `(128,1,8,7168)` 为例，
单阶段 B 只有 64 个 sb CTA；split-C 第一阶段变为 `64 × 4 = 256` 个 CTA，增加通道方向
并行度。第一阶段写出 `[4,sb,n,n]` 和 `[4,sb,n]` 的 FP32 partial，第二阶段用一个小
Kernel 合并，不使用 atomic，也不改变 bias 的归约来源。

split-C 会改变 FP32 的累加分组，并增加 partial 读写和一次 merge launch，因此没有推广
到其他 N 或大 sb。N=4 的实验虽然降低了 Graph 时间，但完整 API 明显退化，最终仍保留
单阶段 B。

### 4.7 分发路径和其他 vendor

| 输入情况 | Kernel 调用方式 | 计算结构 | 设备上下文 |
| --- | --- | --- | --- |
| NVIDIA 小形状 forward | 原始 JITFunction + 固定配置 | 原 forward | `nullcontext()` |
| NVIDIA `(64,8,4,4096)` forward，BF16/FP16 | 原始 JITFunction + dtype 固定配置 | 原 forward | `nullcontext()` |
| NVIDIA 小形状 backward，`n<=4` | 原始 JITFunction + 固定配置 | 单 Kernel 融合 | `nullcontext()` |
| NVIDIA 小形状 backward，`n>4` | 原始 JITFunction + 固定配置 | 原 A、B | `nullcontext()` |
| NVIDIA N=8 split-C | A 直接调用 Autotuner；B 固定配置 | A 输入复用 + split-C/merge | `nullcontext()` |
| NVIDIA 其他满足复用条件的形状 | 直接调用 Autotuner + 裁剪配置 | 输入复用 | `nullcontext()` |
| NVIDIA 其他形状 | 直接调用 Autotuner + 完整配置 | 原实现 | `nullcontext()` |
| 其他 vendor | `LibEntry` + 通用 autotune | 原实现 | `torch_device_fn.device()` |

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

## 7. 正确性测试设计

测试文件：`tests/megatron/test_fused_h_post_bda.py`。

### 7.1 参考实现

- Forward：独立的 PyTorch `bmm + 逐元素运算` reference。
- Backward：对 PyTorch reference 前向执行 `torch.autograd.grad`。
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
专用 split-C 用例由 BF16 和 FP32 覆盖 `(128,1,8,4096)`，每种 dtype 都测试有无 bias，
直接验证四路 C 分片、FP32 partial 和 merge。Forward 专项由三种 dtype 覆盖
`(64,8,4,4099)` 及有无 bias，验证中等形状分发和非整块 C。

### 7.3 特殊测试

除参数化用例外，还包括：

1. **固定 transpose 数值案例**：用固定小矩阵验证 `H_res.T`，防止把矩阵乘方向写反。
2. **非连续 forward 输入**：分别覆盖小形状和 `(64,8,4,4096)` 中等形状固定配置路径。
3. **非连续 backward 输入**：包含非连续 bias 和 `grad_output`，逐项检查五个梯度。
4. **无 bias 返回约定**：验证 `grad_bias is None`。
5. **Split-C backward**：覆盖 BF16/FP32、有无 bias，检查五个梯度及 partial 合并映射。

### 7.4 容差

| dtype | Forward `atol/rtol` | Backward `atol/rtol` |
| --- | --- | --- |
| BF16 | 0.02 | 0.05 |
| FP16 | 0.001 | 0.005 |
| FP32 | 1e-5 | 1e-5 |

### 7.5 用例数量

- 17 组通用 dtype/shape 组合 × 2 种 bias 设置 × 2 个方向 = 68 项。
- 2 种 split-C dtype × 2 种 bias 设置 = 4 项。
- 3 种 forward dtype × 2 种 bias 设置 = 6 项。
- 固定 transpose 案例 = 1 项。
- 非连续 forward = 2 项。
- 非连续 backward = 1 项。

**合计 82 项正确性测试，当前全部通过。**

运行命令：

```bash
CUDA_VISIBLE_DEVICES=3 pytest tests/megatron/test_fused_h_post_bda.py \
  -m "fused_h_post_bda_forward or fused_h_post_bda_backward" --ref cpu -q
```

## 8. 性能测试设计

性能文件：`benchmark/megatron/test_fused_h_post_bda.py`。

### 8.1 Baseline 选择

benchmark 在运行时实际探测 Megatron：

1. 优先使用 Megatron `_triton_h_post_bda_fwd/_bwd`。
2. 用小张量实际执行 forward、backward 并同步设备。
3. 如果导入或执行失败，则警告并回退到显式 PyTorch `bmm/sum` baseline。
4. 打印实际选择的 baseline，避免把 PyTorch fallback 误报成 Megatron。

当前 NVIDIA 环境选择 Megatron Triton baseline。

### 8.2 计时口径

- 使用 GenericBenchmark 的 kernel 模式和 CUDA Event。
- 预热 25 ms，正式测量 100 ms，取中位数。
- 输入随机生成在计时区域之外。
- 首次编译和 autotune 不属于正式测量。
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
对应当前中等形状固定配置版本；backward 对应小形状融合和 N=8 split-C 最终实现。

### 9.1 整体统计

| 算子 | 数据点数 | 最低 Speedup | 算术平均 Speedup | 低于 0.8x 的数量 |
| --- | ---: | ---: | ---: | ---: |
| Forward | 48 | 0.980x | 1.253x | 0 |
| Backward | 48 | 0.833x | 1.770x | 0 |

这里的平均值是各数据点 Speedup 的**算术平均**，不是按耗时加权，也不代表端到端训练加速比。
Forward 本轮优化前的平均值为 1.240x、最低值为 0.830x；backward 优化前分别为
1.171x 和 0.819x。当前 backward 最低点是 FP16 `(64,8,4,4096)` 无 bias 的 0.833x，
该形状继续使用单阶段 B。

最终 JSON 记录为：

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
9. **Split-C 不能只看 Graph 收益**：partial 分配和 merge launch 会影响完整 API；当前仅
   保留实测有效的 N=8 受限路径，其他 N 或更大的 sb 需要重新测量，不能直接推广。
10. **中等形状 forward 配置是受限特调**：固定配置只用于 BF16/FP16 的
    `(64,8,4,4096)`；全流和双流复用曾出现完整 API 回退，不能推广到其他形状或 dtype。

## 11. 相关文件

| 文件 | 作用 |
| --- | --- |
| `src/flag_train/megatron/fused_h_post_bda.py` | Triton Kernel、输入校验和 forward/backward 包装 |
| `src/flag_train/megatron/__init__.py` | 算子导出和 `SpecOpRegistrar` 注册 |
| `tests/megatron/test_fused_h_post_bda.py` | 82 项正确性测试 |
| `benchmark/megatron/test_fused_h_post_bda.py` | Baseline 选择、输入生成和性能测试 |
| `fused_h_post_bda.md` | 本文档 |
