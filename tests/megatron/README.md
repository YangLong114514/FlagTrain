# Megatron 正确性测试

`test_fused_h_post_bda.py` 验证 mHC 的 `fused_h_post_bda_forward` 和
`fused_h_post_bda_backward`。参考实现为独立的 PyTorch 组合前向和 PyTorch autograd
反向，覆盖可选 bias、非连续输入、边界形状及全部输入梯度；`--ref cpu` 会在 CPU 上
执行参考计算。

参考上游为 Megatron-LM 提交 `2052769557dcf7045c4091d1996f0bc59afeb80e` 的
`megatron/core/fusions/fused_mhc_kernels.py`。运行：

```bash
pytest tests/megatron/test_fused_h_post_bda.py \
  -m "fused_h_post_bda_forward or fused_h_post_bda_backward" --ref cpu -vs
```

性能测量见 [`benchmark/megatron/`](../../benchmark/megatron/)。实际验证环境及结果记录在
[`docs/compatibility.md`](../../docs/compatibility.md)。
