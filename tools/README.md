# 开发辅助工具

组件实现归入 `src/flag_train/<组件>/`，正确性验证、性能测量和使用示例分别归入
`tests/`、`benchmark/`、`examples/` 下的对应组件目录。

## `diagnose_kernel_mode.py`

判断 `--mode kernel` 的读数是否可信：它量到的到底是 kernel 时间还是算子自身的 host
派发时间。`triton.testing.do_bench` 的事件窗口会被 `clear_cache` 的 flush 掩盖掉一部分
host 成本，超出部分原样泄漏进读数，因此算子 host 派发较重时该模式会严重高估。

脚本分别测出 host 入队成本、flush kernel 耗时、逐级加大 flush 直到读数不再下降（平台值
= 真实冷 L2 kernel 时间），并给出余量与夸张倍数。

* 用途：为 `benchmark/` 的 KERNEL 模式读数做可信度判据，避免把 host 时间当作 kernel 时间。
* 依赖：`torch`、`triton`、被测仓库自身（`benchmark/deepspeed/test_lamb.py` 提供
  `torch_op` / `train_op`），以及该后端可用的基线。
* 输入：`--op`（当前仅 `lamb`）、`--shape`（展平后的参数元素数，小尺寸最能暴露 host 成本）、
  `--flushes`（flush 大小序列，MB）。
* 输出：仅打印报告，不写文件。包含两个实现的 host 成本、flush 预算、余量、真实 kernel
  时间、kernel 模式读数、夸张倍数，以及一行 verdict 与余量为负时的警告。
* 执行方法：

  ```
  python tools/diagnose_kernel_mode.py --op lamb --shape 1024
  ```

* 实测预期：Hygon DCU 上 n=1024
  余量约 +48 µs、不报警，真实 kernel speedup 约 1.5×，kernel 模式 speedup 约 1.9×。优化前
  该行余量为 −151 µs 并打印警告，真实 kernel speedup 约 1.3×。真实 kernel 时间有约 15% 的
  run-to-run 波动。

脚本只做诊断，不修改被测代码。要扩展到其他算子，把 `torch_op` / `train_op` 参数化即可。

新增工具时，应说明用途、依赖、输入输出和执行方法，并用实际输入验证预期输出。
涉及本地文件写入时说明目标位置；涉及测量结果时保留环境和配置，避免丢失可比条件。
不要用占位脚本或生成的结果声称框架、算子或硬件已经通过验证。
