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

## `gpu_check_nvidia.sh`

CI 辅助脚本：在共享的 NVIDIA 机器上等一张空闲显卡。`.github/workflows/ci.yml` 的 GPU
作业在构建镜像之前调用它，`.github/backends.json` 的 `gpu_check` 字段也指向它。

* 用途：确认有卡可用且显存够用，避免 CI 作业与同时在跑的训练/推理任务抢同一张卡。
* 依赖：`nvidia-smi`；在 runner 宿主机上执行，不在容器内。
* 输入：脚本顶部的三个参数——`mem_threshold`（空闲显存阈值，默认 30000 MB）、
  `sleep_time`（重试间隔，默认 120 秒）、`max_wait`（最长等待，默认 600 秒）；无命令行参数。
* 输出：每轮打印各卡的 总量/已用/空闲 显存表；等到第一张满足阈值的卡后打印
  `Available GPUs: <id,id,...>` 并以 0 退出。`nvidia-smi` 缺失、没有检测到显卡，或等待
  超过 `max_wait` 仍无可用卡时，以非 0 退出。
* 执行方法：

  ```
  bash tools/gpu_check_nvidia.sh
  ```

* 实测预期：机器空闲时第一轮即打印可用卡；整机显存被占用时每 120 秒重试一次，10 分钟后
  报 `Error: Timed out waiting for available GPU.` 并失败。

脚本内容与 FlagGems 的同名脚本一致，两边的 CI 行为保持对齐。

## `gpu_check_{ascend,hygon,iluvatar,metax,thead}.sh`

五个厂商版本的同一个检查：`.github/workflows/backend-test.yaml` 的容器作业在跑测试前调用，
路径由 `.github/backends.json` 的 `gpu_check` 字段指定。

* 用途/输入/输出：与 `gpu_check_nvidia.sh` 完全一致（空闲阈值 30000 MB、重试间隔 120 秒、
  最长等待 600 秒；打印各卡显存表与 `Available GPUs: ...`；无命令行参数）。区别只在查询工具:
  ascend `npu-smi`、hygon `hy-smi`、iluvatar `ixsmi`、metax `mx-smi`、thead `ppu-smi`。
* 依赖：对应的厂商 smi 工具；在容器内执行（作业本身就是容器）。
* 实测预期：与前一个脚本相同（空闲时第一轮通过，被占满时每 120 秒重试、10 分钟后报
  `Timed out` 并失败）。阈值/等待策略未改动，脚本正文与 FlagGems 同名脚本逐字一致（可 diff 验证）。

## `setup_env.sh` 与 `source_env_bash.sh`

CI 环境准备（对标 FlagGems 的 `setup.sh`）：`bash tools/setup_env.sh <backend> [--prebuilt|--no-prebuilt]`。

* 用途：为某个 backend 应用环境并安装依赖——把 `backends.yaml` 里该 backend 的 `env:` 写进
  `$GITHUB_ENV`（本地无该变量时打印 `export K=V`，可 `eval` 使用）、把 `env_source:` 里的厂商
  脚本经 `source_env_bash.sh` 求差集后同样导出，然后 pip 安装 `setuptools>=77`、本仓库
  （`--no-build-isolation -e .`）与测试依赖（pytest/pyyaml/sqlalchemy/packaging）。
  `--prebuilt`（默认）跳过厂商栈，因为 runtime 镜像已带 torch/triton；`--no-prebuilt` 先按
  `backends.yaml` 装该 backend 的 `flagtree`（无则 `triton`）与 `deps:`，走厂商索引
  （`resource.flagos.net/repository/flagos-pypi-<vendor>/simple`）——当前只有 thead 用这条路径。
* 依赖：`python3` + `pyyaml`（镜像内已有）；`source_env_bash.sh` 需要 `bash` 与 `python3`。
* 输入输出：输入是 backend 名与模式；输出是写进 `$GITHUB_ENV` 的环境变量、pip 的安装日志，
  以及最后一条 `import flag_train` 自检（打出 `flag_train: <vendor> / <device>`，失败即非 0 退出）。
  无法展开的 `env:` 值（仍带 `$VAR`）会打印告警并跳过——`$GITHUB_ENV` 不做展开，写进去会污染
  PATH 之类的变量。
* 执行方法：

  ```
  bash tools/setup_env.sh hygon --prebuilt
  eval "$(bash tools/setup_env.sh iluvatar)"     # 本地无 $GITHUB_ENV 时
  ```

* 实测预期：`hygon` 打印一条 `backend env TRITON_HIP_CLANG_PATH=...`，缺失的 DTK 脚本被跳过并提示；
  `iluvatar` 能把同一条目内的 `$COREX_ROOT` 展开进 PATH/LD_LIBRARY_PATH（镜像里已设则原样保留）；
  `thead --no-prebuilt` 先装 flagtree(ppu) 再装 torch，随后装本仓库与测试依赖。

新增工具时，应说明用途、依赖、输入输出和执行方法，并用实际输入验证预期输出。
涉及本地文件写入时说明目标位置；涉及测量结果时保留环境和配置，避免丢失可比条件。
不要用占位脚本或生成的结果声称框架、算子或硬件已经通过验证。
