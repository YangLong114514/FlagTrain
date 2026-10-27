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

当前分支为 `fused_h_post_bda`。本文描述截至提交 `dc00e6d` 的实现和验证结果。

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

当前实现包含三个主要 Triton Kernel，以及一个 Torch bias 归约。

| 部分 | 实现 | 主要工作 |
| --- | --- | --- |
| Forward | `_triton_hpb_fwd_kernel` | 计算 `H_res.T @ R + H_post * (x + bias)` |
| Backward A | `_triton_hpb_bwd_g_x_orig_kernel` | 计算 `grad_x` 和 `grad_original_residual` |
| Backward B | `_triton_hpb_bwd_g_hp_hr_kernel` | 计算 `grad_h_post` 和 `grad_h_res` |
| Bias 归约 | `g_x.sum(dim=0).to(bias.dtype)` | 跨全部 `sb` 计算 `grad_bias` |

Grid 组织方式：

- Forward：二维 grid，分别覆盖 `sb` 和 `C`。
- Backward A：二维 grid，覆盖 `sb` 和 `C`。
- Backward B：一维 grid，覆盖 `sb`，在 Kernel 内沿 `C` 分块归约。

累加使用 FP32，最终写回各输出对应的输入 dtype。

### 4.2 通用 autotune 配置

主要 Kernel 的通用搜索空间为：

- `BLOCK_C ∈ {64,128,256,512}`
- `BLOCK_S ∈ {1,2,4,8}`
- `num_warps ∈ {2,4,8}`

合计 48 组配置，autotune key 为 `C` 和 `N`，Triton 还会把输入 dtype 纳入缓存键。

大形状继续使用这套 autotune 机制。首次编译和调优发生在正式性能测量之前。

### 4.3 NVIDIA 小形状专用路径

小形状对 Python dispatch、Autotuner 和设备上下文非常敏感。当前对满足以下条件的
输入使用专用路径：

```text
vendor == "nvidia" 且 sb <= 8 且 C <= 1024
```

Forward 和 backward 都直接调用原始 `JITFunction`，完全绕过 `LibEntry` 和
Autotuner，并显式传入固定配置：

| 条件 | `BLOCK_C` | `BLOCK_S` | `num_warps` |
| --- | ---: | ---: | ---: |
| `C <= 256` | 64 | `n <= 2` 时为 2，否则为 1 | 4 |
| `256 < C <= 1024` | 128 | `n <= 2` 时为 2，否则为 1 | 4 |

NVIDIA 下不再进入 `torch_device_fn.device()` 上下文，避免小形状上的额外固定开销。

### 4.4 NVIDIA 大形状和其他 vendor

| 输入情况 | Kernel 调用方式 | 设备上下文 |
| --- | --- | --- |
| NVIDIA 小形状 | 原始 JITFunction + 固定配置 | `nullcontext()` |
| NVIDIA 其他形状 | 直接调用 Autotuner | `nullcontext()` |
| 其他 vendor | `LibEntry` + 通用 autotune | `torch_device_fn.device()` |

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

## 7. 正确性测试设计

测试文件：`tests/megatron/test_fused_h_post_bda.py`。

### 7.1 参考实现

- Forward：独立的 PyTorch `bmm + 逐元素运算` reference。
- Backward：对 PyTorch reference 前向执行 `torch.autograd.grad`。
- 使用 `--ref cpu` 时，reference 输入会被独立复制到 CPU，确保参考计算真正在 CPU 上
  执行，而不是只把结果移动到 CPU。
- 随机种子为 2026，随机输入范围为 `[-0.1,0.1]`。

### 7.2 参数化形状和 dtype

| dtype | 覆盖的 `(s,b,n,C)` |
| --- | --- |
| BF16 | `(1,1,1,1)`、`(1,2,2,65)`、`(2,1,4,130)`、`(2,3,4,130)`、`(1,1,8,257)`、`(64,8,4,130)` |
| FP16 | `(1,2,2,65)`、`(2,1,4,130)`、`(1,1,8,257)` |
| FP32 | `(1,2,2,65)`、`(2,1,4,130)`、`(1,1,8,257)` |

这些形状覆盖：

- 最小维度 `s=b=n=C=1`。
- `n=1/2/4/8`。
- 非整块通道 `C=65/130/257`。
- 非整块 `s*b=2/6`。
- 大 `s*b=512`。
- 三种 dtype 下的 `n=2/4/8`。

每组形状分别测试 `bias=None` 和有 bias，forward 和 backward 分开执行。

### 7.3 特殊测试

除参数化用例外，还包括：

1. **固定 transpose 数值案例**：用固定小矩阵验证 `H_res.T`，防止把矩阵乘方向写反。
2. **非连续 forward 输入**：检查输出与 CPU reference 一致。
3. **非连续 backward 输入**：包含非连续 bias 和 `grad_output`，逐项检查五个梯度。
4. **无 bias 返回约定**：验证 `grad_bias is None`。

### 7.4 容差

| dtype | Forward `atol/rtol` | Backward `atol/rtol` |
| --- | --- | --- |
| BF16 | 0.02 | 0.05 |
| FP16 | 0.001 | 0.005 |
| FP32 | 1e-5 | 1e-5 |

### 7.5 用例数量

- 12 组 dtype/shape 组合 × 2 种 bias 设置 × 2 个方向 = 48 项。
- 固定 transpose 案例 = 1 项。
- 非连续 forward = 1 项。
- 非连续 backward = 1 项。

**合计 51 项正确性测试，当前全部通过。**

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
- 输出分配、`.contiguous()` 和 bias 梯度求和包含在被测 API 内。
- Backward 直接传入 `grad_output` 调用 backward，**不经过 autograd，也不包含 forward
  耗时**。
- Speedup 定义为 `Megatron latency / FlagTrain latency`，大于 1 表示 FlagTrain 更快。

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

以下结果来自 NVIDIA H20-3e、BF16/FP16/FP32 全精度综合集。时间单位为 ms。

### 9.1 整体统计

| 算子 | 数据点数 | 最低 Speedup | 算术平均 Speedup | 低于 0.8x 的数量 |
| --- | ---: | ---: | ---: | ---: |
| Forward | 48 | 0.882x | 1.167x | 0 |
| Backward | 48 | 0.848x | 1.161x | 0 |

这里的平均值是各数据点 Speedup 的**算术平均**，不是按耗时加权，也不代表端到端训练加速比。

### 9.2 Forward 代表性数据（BF16）

| Shape | Bias | Megatron (ms) | FlagTrain (ms) | Speedup |
| --- | --- | ---: | ---: | ---: |
| `(1,2,2,65)` | 无 | 0.007616 | 0.005472 | 1.392x |
| `(1,2,2,65)` | 有 | 0.008032 | 0.005472 | 1.468x |
| `(2,4,4,1024)` | 无 | 0.008704 | 0.005632 | 1.545x |
| `(2,4,4,1024)` | 有 | 0.008160 | 0.005888 | 1.386x |
| `(64,8,4,4096)` | 无 | 0.019168 | 0.018848 | 1.017x |
| `(64,8,4,4096)` | 有 | 0.019360 | 0.020000 | 0.968x |
| `(8192,4,4,7168)` | 无 | 1.528384 | 1.509680 | 1.012x |
| `(8192,4,4,7168)` | 有 | 1.515728 | 1.458336 | 1.039x |

小形状主要收益来自消除固定 dispatch 开销；大形状继续维持在接近 Megatron 的水平。

### 9.3 Backward 代表性数据（BF16）

| Shape | Bias | Megatron (ms) | FlagTrain (ms) | Speedup |
| --- | --- | ---: | ---: | ---: |
| `(1,2,2,65)` | 无 | 0.055520 | 0.038528 | 1.441x |
| `(1,2,2,65)` | 有 | 0.065120 | 0.048896 | 1.332x |
| `(2,4,4,1024)` | 无 | 0.058016 | 0.048736 | 1.190x |
| `(2,4,4,1024)` | 有 | 0.068816 | 0.054368 | 1.266x |
| `(64,8,4,4096)` | 无 | 0.061152 | 0.072128 | 0.848x |
| `(64,8,4,4096)` | 有 | 0.071936 | 0.084736 | 0.849x |
| `(8192,4,4,7168)` | 无 | 2.871776 | 2.887904 | 0.994x |
| `(8192,4,4,7168)` | 有 | 3.146928 | 3.064288 | 1.027x |

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

## 11. 相关文件

| 文件 | 作用 |
| --- | --- |
| `src/flag_train/megatron/fused_h_post_bda.py` | Triton Kernel、输入校验和 forward/backward 包装 |
| `src/flag_train/megatron/__init__.py` | 算子导出和 `SpecOpRegistrar` 注册 |
| `tests/megatron/test_fused_h_post_bda.py` | 51 项正确性测试 |
| `benchmark/megatron/test_fused_h_post_bda.py` | Baseline 选择、输入生成和性能测试 |
| `fused_h_post_bda.md` | 本文档 |
