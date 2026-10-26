# 兼容性与验证记录

以下只记录实际执行过的算子、软件和硬件组合；包能安装或导入不表示能够运行完整框架
或训练模型。

| 组件 | 当前内容 | 已验证框架版本 | 已验证硬件 | 正确性与性能 |
| --- | --- | --- | --- | --- |
| Megatron | fused_h_post_bda 前向、反向 | main `2052769` / Core `0.21.0+2052769` | NVIDIA H20-3e | 23 项正确性测试、2 项核心 benchmark 通过 |
| DeepSpeed | 空包与目录说明 | 无 | 无 | 未验证 |
| Transformer Engine | 空包与目录说明 | 无 | 无 | 未验证 |

## Megatron fused_h_post_bda（2026-10-26）

验证基于 FlagTrain 提交 `2fad49f` 之上的工作区实现，参考上游 Megatron-LM 提交
`2052769557dcf7045c4091d1996f0bc59afeb80e`。环境为 Linux、Python 3.12.3、
PyTorch `2.11.0a0+eb65b36914.nv26.02`、Triton 3.6.0、CUDA 13.1、驱动 570.158.01；
正确性和性能测试各使用一张 NVIDIA H20-3e，均为单进程、单设备，无分布式并行。

正确性测试使用固定随机种子 2026，覆盖 BF16、FP16、FP32，bias 有无，`n=2/4/8`、
非整块通道、非连续输入及固定 transpose 数值案例。参数化形状为 `(1,2,2,65)`、
`(2,1,4,130)`、`(1,1,8,257)`。参考计算在 CPU 上执行；BF16 前向/反向容差分别为
0.02/0.05，FP16 为 0.001/0.005，FP32 均为 1e-5。输出和全部五个梯度共 23 项测试通过：

```bash
pytest tests/megatron/test_fused_h_post_bda.py \
  -m "fused_h_post_bda_forward or fused_h_post_bda_backward" --ref cpu -vs
```

性能测试使用 BF16，核心形状为 `s=64,b=8,n=4,C=4096`；Megatron PR 的综合形状
`s=8192,b=4,n=4,C=7168` 未执行。baseline 经真实执行探测选择 Megatron 的直接 Triton 实现，
不是 cuTile 或 PyTorch fallback。kernel 模式预热 25 ms、测量 100 ms，使用 CUDA Event
并取中位数；输入生成和首次编译/autotune 不计时，公开 API 内的分配、连续化及 bias
梯度求和计入。反向直接调用 backward，不经过 autograd 或前向。未记录峰值显存。

| 算子 | Megatron (ms) | FlagTrain (ms) | FlagTrain / Megatron 速度比 |
| --- | ---: | ---: | ---: |
| forward | 0.019040 | 0.024352 | 0.782x |
| backward | 0.076640 | 0.089760 | 0.854x |

复现命令：

```bash
pytest benchmark/megatron/test_fused_h_post_bda.py \
  -m "fused_h_post_bda_forward or fused_h_post_bda_backward" \
  --level core --mode kernel --warmup 25 --iter 100 -vs
```

这些是首版单算子基线，FlagTrain 当前慢于同语义 Megatron 实现，不能外推为端到端训练
收益。尚未验证其他 GPU/vendor、分布式训练、cuTile 或上述之外的形状与精度组合。

## 添加验证记录

新增记录应包含验证范围、FlagTrain 和上游提交、关键依赖、操作系统、硬件与驱动、输入
形状和精度、参考实现、容差、实际结果、计时条件、复现命令及未覆盖限制。只将实际测过
的组合列为已验证；单个算子或设备的结果不能证明整个框架兼容。验证方法见
[开发说明](development.md)。
