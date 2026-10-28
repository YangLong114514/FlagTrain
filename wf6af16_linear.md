# wf6af16_linear 算子文档

> FP6 权重 / FP16 激活线性层，移植自 DeepSpeed 的 `cuda_wf6af16_linear`（源自 FP6-LLM），
> Triton 实现，支持多后端。本文档覆盖：功能、接口、NVIDIA 实现方案、国产平台后端实现方案、
> 实现历程、历史问题与解决方案、测试设计。

---

## 1. 功能介绍

`wf6af16_linear` 是一个**权重量化线性层**算子，计算：

```
output = hidden_states @ (W * scales)^T
```

其中：

- **W** 是量化到 **FP6-E3M2** 的权重矩阵（1 符号位 + 3 指数位 + 2 尾数位，指数 bias 3，
  无 inf/NaN 编码，指数 7 按正规值处理，absmax = 28，次正规分支 `m * 2^-4`），
  每个输出通道配一个 fp16 scale（per-channel 量化）；
- **hidden_states** 是 FP16 激活；
- kernel 内部把 FP6 解包、反量化为 FP16（FP6 的每个值都能在 FP16 中精确表示，
  反量化无损），以 **FP32 累加** 做 GEMM，输出 **FP16**。

典型用途是推理场景的 weight-only 量化：权重显存占用降到 FP16 的 6/16 = **37.5%**，
decode 阶段（token 数少、带宽瓶颈在权重读取）收益最大。

**职责边界**：算子只负责"解包 + 反量化 + GEMM"这条每步前向都要走的热路径。
权重的**量化**（fp16 → FP6 code）和**打包**（code → 两个 uint8 切片）是模型加载时
一次性完成的输入准备，不属于算子——这一点与 DeepSpeed 的分层完全一致
（DeepSpeed 中是 module 层的 `fp_quantize` + `preprocess_weight`，kernel 不管；
见第 5、6 节的讨论）。FlagTrain 测试目录里提供了这两个操作的 torch 参考实现，
供调用方准备输入。

### 与 DeepSpeed 原版的关系

| 维度 | DeepSpeed `cuda_wf6af16_linear` | FlagTrain `wf6af16_linear` |
|---|---|---|
| 实现 | CUDA C++（InferenceCore 扩展） | Triton |
| 架构 | **锁死 sm_80（A100）**，`__CUDA_ARCH__` 守卫 | 多后端（NVIDIA / 沐曦 / 天数 / 昇腾 / 海光…） |
| 权重打包布局 | 32 线程 MMA 位交织（FP6-LLM 布局） | **密集行主序**（自定义，互不兼容） |
| 量化/打包 | module 层辅助函数 | 同样不在算子内（测试提供参考实现） |
| 功能范围 | 无 bias、无激活融合、无 batch | 相同（逐一对齐） |

注意：**不能把 DeepSpeed `preprocess_weight` 的输出直接喂给本算子**，字节内布局不同。

---

## 2. 接口介绍

唯一公开接口（各后端 override 注册在同一个命名空间下，调用方无感）：

```python
import flag_train

output = torch.empty((tokens, out_channels), dtype=torch.float16, device=dev)
flag_train.deepspeed.wf6af16_linear(
    output, hidden_states, weights_2bit, weights_4bit, scale,
    out_channels=None, tokens=None, in_channels=None,
    split_k=None, wide_load=None,   # 后两个为可选调优参数
)
```

### 参数

| 参数 | 类型 | 形状 | 说明 |
|---|---|---|---|
| `output` | fp16 Tensor | `[tokens, out_channels]` | 输出，调用方预分配，函数返回同一对象 |
| `hidden_states` | fp16 Tensor | `[tokens, in_channels]` | FP16 激活 |
| `weights_2bit` | uint8 Tensor | `[out_channels, in_channels // 4]` | FP6 code 高 2 位切片（布局见下） |
| `weights_4bit` | uint8 Tensor | `[out_channels, in_channels // 2]` | FP6 code 低 4 位切片 |
| `scale` | fp16 Tensor | `[out_channels]` | 每输出通道一个 scale |
| `out_channels` | int，可选 | — | 缺省从 `scale` 推断 |
| `tokens` | int，可选 | — | 缺省从 `hidden_states` 推断 |
| `in_channels` | int，可选 | — | 缺省从 `hidden_states` 推断 |
| `split_k` | int，可选 | — | 显式指定 split-K 因子；缺省走启发式（见 3.3） |
| `wide_load` | bool，可选 | — | 强制开/关 uint32 宽读；缺省按对齐条件自动（通用后端专属，厂商后端不接受此参数） |

### 权重打包布局（FlagTrain 自定义，密集行主序）

6-bit code 拆成"高 2 位（符号 + 指数最高位）+ 低 4 位"两片分别存储：

- `weights_2bit[m, j]`：第 `4j..4j+3` 列共 4 个 code 的高 2 位，第 `4j+i` 列位于字节内
  位 `6-2i`；
- `weights_4bit[m, j]`：第 `2j..2j+1` 列共 2 个 code 的低 4 位，第 `2j+i` 列位于位 `4-4i`。

反量化公式（与 DeepSpeed CUDA kernel 逐位一致）：

```
code -> sign * 2^(e-3) * (1 + m/4)        e != 0
        sign * m * 2^-4                   e == 0（次正规）
再乘以该行 scale
```

### 输入约束（wrapper 断言）

- dtype：激活/输出/scale 为 fp16，权重切片为 uint8；
- `in_channels % 4 == 0`（打包以 4 code 为单位）；
- 所有张量必须 **contiguous**（kernel 按行主序裸指针寻址，非连续输入直接报错而不是算错）；
- `split_k` 需满足 `in_channels % (split_k * 4) == 0`（每个切片起点落在字节边界），
  否则回退为 1。

### 厂商后端注册机制

各后端在 `flag_train/runtime/backend/_<vendor>/ops/deepspeed/wf6af16_linear.py`
定义同名实现并在本包 `__all__` 中列出；`SpecOpRegistrar` 在 `import flag_train` 时
将其覆盖到 `flag_train.deepspeed.wf6af16_linear`。测试与 benchmark 只读这个命名空间，
因此对调用方而言**接口永远只有一个**，后端差异透明。当前后端：

- **通用后端**（`src/flag_train/deepspeed/wf6af16_linear.py`）：NVIDIA、海光及未覆盖平台；
- **沐曦**（`_metax`）：自有 kernel 拷贝 + 自有调优空间（见 4.1）；
- **天数**（`_iluvatar`）：复用通用 kernel，裁剪过的配置空间 + 关闭宽读（见 4.2）；
- **昇腾**（`_ascend`）：自有 kernel（流式拆解的 unpack 结构）。

---

## 3. NVIDIA 平台实现方案

通用后端三个组件：GEMM kernel（autotune）、split-K 归约 kernel、host wrapper。

### 3.1 GEMM kernel 结构

`wf6af16_linear_kernel` 计算 `C = (W * scales) @ A^T` 的一个 tile
（W: `[M,K]` 打包权重，A: `[N,K]` 激活，C: `[N,M]` 输出）：

1. 按 `(pid_m, pid_n, pid_k)` 定位 tile，K 维按 `split_k` 切片；
2. 主循环沿 K 以 `BLOCK_K` 步进：
   - 加载激活 tile（mask 到 `k_end`，越界置 0）；
   - **紧凑字节读取**权重切片：2-bit 片每行只读 `BLOCK_K/4` 字节、4-bit 片只读
     `BLOCK_K/2` 字节，再用 `tl.interleave` 还原列序——消除了"每个 code 读一个字节"
     的 4x/2x 冗余读；
   - 位运算合成 6-bit code（`(top2 << 4) | low4`）；
   - **软件反量化**：`e = (code>>2)&7`，`m = code&3`，`where(e==0, m*0.0625,
     exp2(e-3)*(1+m*0.25))`，加符号、转 fp16、乘行 scale——FP6 值在 fp16 中精确
     表示，此转换无损；
   - `tl.dot(a, w^T, acc)` 喂 tensor core，fp32 累加；
3. 写出：`SPLIT_K_MODE` 下写 fp32 部分和到 `[split_k, N, M]` workspace，
   否则直接写 fp16 输出；split-K 时由 `wf6af16_split_k_reduce_kernel` 做归约。

### 3.2 自动调优空间

12 组 tile 配置（`BLOCK_M × BLOCK_N × BLOCK_K × warps × stages`），
key = `[M, N, K, split_k]`：

- 小 `BLOCK_N`（16/32）服务 decode（带宽瓶颈在打包权重）；
- 大 tile + 深流水（stages 3-4）服务 prefill（计算瓶颈）；
- 其中 `(16,16,64)`、`(64,16,64)` 浅流水、`(32,128,64)` 四组来自沐曦 C550 扫参的
  优胜配置，在 H20 上实测同样更快，反哺进了通用空间（见 4.1）。

### 3.3 split-K 策略（`_pick_split_k`）

1. tokens ≤ 768 时先查 **DeepSpeed 在 A100 上 profile 的 split-K 表**
   （按 token 档 × 输出通道），通过 4 列字节对齐校验才采用；
2. 查不到或对齐失败时，decode 范围（tokens ≤ 64）回退 **2 的幂**：
   tokens ≤ 16 取 8，否则取 4，并带对齐校验与最小切片长度（64 列）守卫；
3. 其余取 1（单趟 + fp16 直写）。

DeepSpeed 表里的奇数因子（13、18…）大多过不了字节对齐，没有 fallback 时
decode 形状几乎全部退化为 split_k=1，几十个 CTA 啃整条 K——pow2 fallback
在 H20 decode 上带来 1.3-1.7x。

### 3.4 WIDE_LOAD（uint32 宽读）

decode 路径上把两个打包切片按 **uint32 字**读取、寄存器内拆字节
（2-bit 片一个字 16 列、4-bit 片一个字 8 列），减少 load 指令数。
门控条件（全部满足才开启）：

- tokens ≤ 64（仅 decode 的窄 tile 有收益，prefill 实测中性到负收益）；
- `K % 16 == 0` 且 `k_per_split % 16 == 0`（每行、每切片起点 4 字节对齐）；
- 两个权重指针 4 字节对齐。

H20 上 tokens=1 约 22% 收益。该路径在通用 kernel 内以 `WIDE_LOAD` constexpr
分支实现，不满足条件走原始字节路径。

### 3.5 性能数据（H20-3e，torch 2.11 / triton 3.6，基线为 torch 组合实现）

| shape (tokens, in, out) | 算子延迟 | 对 torch 基线加速比 |
|---|---|---|
| (1, 4096, 4096) | 0.050 ms | **68.0x** |
| (4, 4096, 4096) | 0.061 ms | 56.1x |
| (8, 4096, 4096) | 0.061 ms | 17.5x |
| (16, 4096, 4096) | 0.058 ms | 59.9x |
| (16, 14336, 4096) | 0.180 ms | 46.6x |
| (32, 4096, 4096) | 0.090 ms | 37.7x |
| (64, 4096, 4096) | 0.099 ms | 34.4x |
| (128, 4096, 14336) | 0.227 ms | 37.4x |
| (256, 8192, 8192) | 0.491 ms | 18.6x |
| (512, 4096, 4096) | 0.247 ms | 14.3x |
| (1024, 4096, 4096) | 0.462 ms | 8.1x |

基线是"解包 → 反量化 → cuBLAS FP16 GEMM"的 torch 组合，代表一个正确但未融合的
实现；sm_80（A100）机器上基线会自动切换为 DeepSpeed 原版 kernel。
优化历程：初版 → interleave 紧凑解包 + autotune + split-K（10-33x）→
反哺沐曦优化（pow2 split-K fallback + 新 tile + WIDE_LOAD）后 decode 再提升
约 2x、prefill 最高 1.5x。

---

## 4. 国产平台后端实现方案

各后端与通用后端的关系：kernel 要么复用通用的裸 JIT（`_wf6af16_linear_gemm`）
自行包装调优空间，要么整体自带 kernel 拷贝；split-K 归约 kernel 一律复用通用实现；
注册机制见 2.5。

### 4.1 沐曦（MetaX C550）

**自有 kernel 拷贝 + 自有调优空间**，是改动最深的一个后端，也是反哺通用后端
优化的来源。与通用 kernel 的差异：

1. **激活 mask 绑定 kernel 参数 K，而非循环内计算的 `k_end`**。两者接受的 lane
   完全一致（split_k=1 时 `k_end == K`；split-K 时越界进入相邻切片的列仍在界内，
   其贡献由显式置 0 的尾部 code 抵消——沐曦版用
   `code = tl.where(offs_k < k_end, code, 0)` 替代通用版的"零激活抵消"）。
   MACA 编译器给参数边界的 mask 生成的代码远好于循环计算值边界的 mask：
   同一 tile 下 (512,4096,4096) 从 3270us → 812us，(128,4096,14336) 从 ~2900us
   → 532us。同时参数边界 mask 可静态分析，**规避了 MACA pipeliner 对
   "循环边界相关 load mask"的误编译**。
2. **全配置 `num_stages=1`**：上述 pipeliner bug 下深流水间歇性算错 tile，
   且扫参显示深流水在 C550 上从未获胜。
3. **专属 tile 空间（4 组）**：decode 用 `(64,16,64)` / `(16,16,64)` 小 tile，
   prefill 用 `(32,128,64)`（4/8 warps 两档）——C550 上全量扫参的优胜者。
4. **decode 的 split-K 用 2 的幂**（tokens≤16 取 8，≤64 取 4，带对齐与最小切片
   长度守卫）：2 的幂天然满足字节对齐，而 DeepSpeed 表的奇数因子大多对齐失败
   退化为 1。1 token、K=M=4096 实测 264us → 123us。
5. **WIDE_LOAD 的首发地**：uint32 宽读 + 寄存器拆字节，decode 收益 5-7%，
   门控条件（对齐 + 仅 decode）后被通用后端原样吸收。
6. **自定义 `do_bench`**：沐曦驱动默认每线程私有内存上限 4KB，超限 config
   启动时报普通 `RuntimeError`（stock autotuner 不容错），记为无穷慢跳过。

整体收益：相对初版通用实现 1.5-4.5x。

### 4.2 天数（Iluvatar CoreX）

**复用通用裸 kernel，自有 launcher + 裁剪过的配置空间**。kernel 逻辑与通用版
完全一致（字节路径在天数上是正确的），差异只在两件事：

1. **强制 `WIDE_LOAD=False`**：CoreX 后端误编译宽读路径的循环内地址计算
   （`tl.cast(ptr, pointer_type(uint32)) + k0//16`），单迭代形状正常、
   多次迭代约 37% 元素算错（详见 6.5）。
2. **配置空间裁剪为 9 个**：经逐 config 诊断（绕过 autotune 逐一定参对拍），
   `(16,16,64,2w)`、`(128,128,32)`、`(128,256,32)` 三个 config 在天数上
   误编译（约 100% 全错），从空间中剔除（详见 6.6）。

split-K 启发式、tile 的其余 9 组、归约 kernel 均与通用后端一致。

### 4.3 海光（Hygon DCU）

**无独立后端，直接走通用后端**（含宽读、pow2 split-K 等全部优化），
11 个功能测试全部通过，未做平台专属调优。注意 DeepSpeed 原版 kernel 在海光
HIP 编译下是空桩（见 6.8），因此海光的正确性参照物和性能基线都是
torch 组合实现。

### 4.4 平头哥 PPU（T-Head，待接入）

**暂未跑通**：FlagTree PPU 后端与机器上 PPU_SDK 的 `ppu-llc` 版本不匹配，
kernel 在编译期失败（汇编器无法解析 formatter 生成的文件头，见 6.10），
属环境问题。待 SDK 版本对齐后按通用后端验证数值；若出现类似天数的误编译，
按同样模式加后端覆盖即可。

---

## 5. 实现历程

| 时间 | 提交 | 内容 |
|---|---|---|
| 10-08 | `c783b58` | **初版移植**：Triton kernel + split-K 归约 + 精度/性能测试，量化打包 helper 在库内 |
| 10-09 | `1da174e` | **NVIDIA 优化**：`tl.interleave` 紧凑解包、8 组 autotune 配置、split-K 对齐守卫 |
| 10-10 | `b035563` | **compute-only 重构**：量化/打包移出库（host 侧不允许 torch 计算），只暴露一个接口 |
| 10-10 | `49fcf9a` | 测试统一为裸 `import flag_train` + `flag_train.deepspeed.wf6af16_linear` 调用 |
| 10-13 | `f4fe038` | **沐曦私有内存问题修复**：自定义 `do_bench`，超限 config 记为无穷慢跳过 |
| 10-13 | `03ec559` | **沐曦 pipeliner 规避**：全空间 `num_stages=1` |
| 10-13 | `f095c6c`~`35ca0e7` | **沐曦独立后端**：K 参数边界 mask、专属 tile 空间、pow2 split-K、WIDE_LOAD |
| 10-14 | `3c7720e` | **沐曦优化反哺通用后端**：新 tile、split-K pow2 fallback、通用 kernel 加 WIDE_LOAD（H20 再获 1.2-2.1x） |
| 10-15 | `8b9f0cb`/`17b1452` | **天数独立后端**：强制关闭宽读 + 裁剪出 9 个验证通过的 config |
| 10-20 | `cde1cd2`/`cf3789e` | **海光 oracle 门控修复**：DeepSpeed kernel 在 HIP 下是空桩，移出基线厂商 |
| 10-20 | `d1c6677` | **测试扩充**：逐 config 扫描、边界 shape、负例，benchmark 6→11 shape |
| 10-20 | `b34cab1` | **昇腾后端**：流式拆解的 unpack 结构绕过 BiShengHIR 限制 |
| 10-22 | `6accb10` | **天数 CI 修复**：FP6 值表改在 CPU 构建，绕开 ixrtc 的 exp2 JIT 失败 |
| 10-22 | `52a83da` | 删除非公开接口（量化/打包 helper）的测试 |
| 10-22 | `2fad49f` | PR review 意见处理（日志前缀、`__all__` 格式等） |

---

## 6. 历史问题与解决方案

### 6.1 DeepSpeed 原版 kernel 锁死 sm_80

**现象**：本机 H20（sm_90）跑 DeepSpeed 原版算子无法执行；而 DeepSpeed 自己的
单测在本机显示 pass——实为测试内 `major != 8` 时 early-return 空跑，pass 是假象。

**原因**：`kernel_matmul.cuh:32` 有
`#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && __CUDA_ARCH__ < 900`，
其他架构编译为空桩。

**方案**：测试/benchmark 的基线与 oracle 做三层降级：vendor 在白名单
→ capability == (8,0) → InferenceCore 可构建，否则用 torch 组合参考实现，
并在 skip 原因里写明。

### 6.2 权重打包布局与 DeepSpeed 不兼容

DeepSpeed 为喂 MMA fragment 把比特按 32 线程交织；Triton kernel 用不上这种布局。
FlagTrain 自定义密集行主序布局（第 2 节），并在模块 docstring 中明确声明
"不得直接喂 DeepSpeed `preprocess_weight` 的输出"。

### 6.3 沐曦：私有内存 4KB 限制导致 autotune 崩溃

**现象**：沐曦 C550 上 benchmark 直接失败：
`private memory size required ... system: 4 KB/Thread, kernel request: 5 KB/thread`，
`mcModuleLaunchKernel: Returned mcErrorMemoryValueTooLarge`。

**原因**：沐曦驱动默认每线程私有内存上限 4KB（内核模块参数），部分大 tile 配置
寄存器溢出后超限；Triton autotuner 只容错 `OutOfResources` 等三种异常，
MACA 后端报的是普通 `RuntimeError`，一个超限 config 就让整个调优中断。

**方案**：autotune 传自定义 `do_bench`，把启动失败的 config 记为无穷大耗时自动跳过。

### 6.4 沐曦：pipeliner 误编译

**现象**：深流水（num_stages ≥ 2）下间歇性算错 tile。

**原因**：MACA 编译器对"循环边界相关（loop-bound-dependent）load mask"的
流水化有 bug；另外它给绑定到 kernel 参数（K）的 mask 生成的代码远好于绑定到
循环内计算值（k_end）的 mask——同一 shape 差 4-5.5x。

**方案**：沐曦后端 kernel 的激活 mask 改绑参数 K（超界 lane 靠显式把尾部 code
置 0 抵消），全配置 `num_stages=1`。该改动在 NVIDIA 上实测无收益，**不回迁**
通用后端。

### 6.5 天数：WIDE_LOAD 多次迭代误编译

**现象**：天数 CoreX 上约 37% 元素大面积算错，且不同 split_k 下错误值相同。

**排查**：失败/通过与"宽读开启 且 K 循环迭代 > 1"完美相关（单迭代全过）——
CoreX Triton 后端对 `tl.cast(ptr, pointer_type(uint32)) + k0//16` 的循环内
地址计算误编译。同样代码在 H20 上全对。

**方案**：天数建独立后端，强制 `WIDE_LOAD=False`（字节路径本身是对的）。

### 6.6 天数：3 个 tile config 误编译

**现象**：宽读修复后仍有两个用例约 100% 全错，且失败集合随代码改动漂移。

**排查**：写诊断脚本绕过 autotune 逐 config 单跑，结论干净：
`(16,16,64,2w)`、`(128,128,32)`、`(128,256,32)` 三个 config 全错，其余 9 个全对。
**autotune 只比速度不校验正确性**，误编译 config 一旦最快就会被选中。

**方案**：天数后端自行 wrap 裸 kernel，配置空间只保留 9 个验证通过的 config。
同时把"逐 config 正确性扫描"固化成正式测试（`test_wf6af16_linear_all_configs`），
防止下一个平台踩同样的坑。

### 6.7 天数 CI：torch.exp2 的 ixrtc JIT 失败

**现象**：天数 CI 上第一个测试就挂，报错在 torch 参考实现的
`torch.exp2(e - 3.0)`：`ixrtcCompileProgram failed`——CI 容器的 PyTorch 栈
编译不了 fp32 exp2 的 elementwise kernel。我们算子本身已经跑通，挂在算参考值。

**方案**：FP6 值表（仅 64 个精确值）改在 CPU 上构建再搬到 device，CPU 不走 ixrtc JIT。

### 6.8 海光：DeepSpeed oracle 是空桩

**现象**：海光 DCU 上我们的算子 11 个功能测试全过，但 oracle 对照测试输出
全是脏数据或报 `out of memory`。

**原因**：6.1 的 `__CUDA_ARCH__` 守卫在 HIP 编译下不成立，kernel 编译成空桩——
小 shape 读到未初始化内存，大 shape 直接报错。

**方案**：把 hygon 从 `_DEEPSPEED_BASELINE_VENDORS` 移除（单测和 benchmark 两处），
海光的参照物固定为 torch 组合实现，并加专门的 skip 说明。

### 6.9 昇腾：BiShengHIR 编译器限制

**现象**：指针位宽扩展 cast 直接被拒（`Casting pointers with unmatched bitwidth!`，
uint32 在 A2/A3 上不是硬件 dtype）；字节路径仅在 BLOCK_K=32 能编译
（192KB 统一缓冲区限制），且 `tl.interleave` 两种布局的 `|` 组合会被
BiShengHIR 插入的布局转换打乱数值。

**方案**：昇腾后端重写 kernel 的 unpack 结构——把每字节 4 列当作 4 条独立 K 流
端到端处理，4-bit 片用偶/奇两路 stride-2 字节读，激活 tile 按 stride-4 读 4 流，
最终 tile 的 dot 为 4 个分流的 dot 之和，全程无 `tl.interleave`。

### 6.10 平头哥 PPU：工具链版本不匹配（未解决）

**现象**：PPU 上编译期报错，`ppu-llc` 无法解析 `irformatter` 生成的汇编文件头
（`line 1:0 mismatched input 'ppu'`）。

**结论**：FlagTree PPU 后端与机器上 PPU_SDK 的 `ppu-llc` 版本不匹配，属环境问题，
需要对齐 SDK 版本；算子代码侧无可绕开关。待环境修复后再验证数值
（若出现与天数类似的误编译，按同样模式加后端覆盖）。

---

## 7. 测试设计

### 7.1 精度测试（`tests/deepspeed/test_wf6af16_linear.py`，20 用例）

双层 oracle 设计：

- **`wf6af16_ref`（主）**：torch 组合实现（解包 → FP6 值表反量化 → fp32 累加 matmul），
  任何平台可跑；
- **DeepSpeed 原版 kernel（辅）**：仅 sm_80 + 白名单厂商，独立实现互证，
  防止"参考实现和 kernel 同源误读"；skip 原因显式区分（Ampere-only / Hygon stub /
  未安装）。

| 测试 | 内容 |
|---|---|
| `test_wf6af16_linear` | 8 个 shape 对拍：含非对齐形状（tokens 3/33/100、out 48/384）、`K%16≠0` 字节 fallback（`(7,260,68)`）、大 K + split-K + 宽读 + 8 次迭代（`(1,4096,512)`）、tokens>768 单趟（`(800,512,256)`） |
| `test_wf6af16_linear_split_k` | split_k ∈ {1,2,4,8} 的部分和 + 归约路径 |
| `test_wf6af16_linear_split_k_misaligned_falls_back` | 显式传 split_k=3（切片不对齐）验证回退 |
| `test_wf6af16_linear_rejects_bad_inputs` | dtype 错误、非连续输入必须抛 AssertionError |
| `test_wf6af16_linear_all_configs` | **逐 config 扫描**：解开当前后端 autotuner，每个 config 在字节路径（及宽读平台的宽读路径）单独对拍——防误编译 config 被 autotune 选中（天数事故的固化防线） |
| `test_matches_deepspeed_oracle` | 5 个 token 数对拍 DeepSpeed 原版（仅 Ampere 白名单平台） |

输入准备 helper（`quantize_weights_fp6` / `pack_weights_fp6` / `unpack_weights_fp6`）
是 torch 参考实现，留在测试文件中供调用方参考，不作为被测对象。

### 7.2 性能测试（`benchmark/deepspeed/test_wf6af16_linear.py`，11 个 shape）

- decode 梯度 1/4/8/16/32/64（卡住 split-K 启发式的分档边界），
  prefill 128/256/512/1024（越过 split-K 表 768 行上限），
  另加 `(16, 14336, 4096)` 大 K decode；
- 基线按平台降级：sm_80 白名单 → DeepSpeed 原版 kernel，否则 torch 组合实现；
- 计时只含 kernel 调用，打包/量化在 input 准备阶段完成（与真实推理一致）。

### 7.3 运行方式

```bash
# 精度（两个目录的 conftest 都注册 --override，不能放一条命令里跑）
python -m pytest tests/deepspeed/test_wf6af16_linear.py -q
# 性能
python -m pytest benchmark/deepspeed/test_wf6af16_linear.py -q -s
```

当前 H20 状态：15 passed / 5 skipped（oracle skip 符合预期），benchmark 全 SUCCESS。
