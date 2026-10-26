# Megatron 性能基准

`test_fused_h_post_bda.py` 分别测量 mHC 前向和反向算子。baseline 在运行时做真实
执行探测：优先使用 Megatron-LM 提交 `2052769557dcf7045c4091d1996f0bc59afeb80e`
中的直接 Triton 实现；导入或执行失败时退回 PyTorch 组合实现，并打印实际选择。

反向 benchmark 直接传入 `grad_output` 调用 backward API，不通过 autograd 建图，也不
包含前向耗时。输入生成不计时；公开 API 内的输出分配、连续化和 bias 梯度求和计入。
首次编译与 autotune 不属于正式测量，耗时由项目 GenericBenchmark 的 CUDA Event
预热、重复和中位数统计得到。

先通过 [`tests/megatron/`](../../tests/megatron/) 的正确性测试，再运行：

```bash
pytest benchmark/megatron/test_fused_h_post_bda.py \
  -m "fused_h_post_bda_forward or fused_h_post_bda_backward" \
  --level core --mode kernel --warmup 25 --iter 100 -vs
```

这些结果是单设备算子数据，不代表端到端训练吞吐。实际环境及限制见
[`docs/compatibility.md`](../../docs/compatibility.md)。
