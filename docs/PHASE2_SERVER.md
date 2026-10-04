# 二阶段服务器实验：π0.5 vs 本地 VLM + Jev

## 实验问题

本实验比较两条彼此独立的策略：

1. `pi05`：相机图像、语言指令和机器人本体状态直接送入 π0.5，模型返回 7 维连续动作块。每次执行前 5 步后重新推理；整条路线不调用 Jev。
2. `vlm-jev`：本地 VLM 只输出有边界的场景描述、XYZ/旋转方向证据和夹爪建议；Jev 随后从固定的 21 个原子动作中选择一个。控制代码执行该动作 5 个环境步后重新观察。

二者使用相同的 LIBERO task、init state、seed、原始双相机图像、最大环境步数以及 `env.check_success()`。与 openpi 官方 LIBERO 示例一致，两条路线会先执行 10 个不计入策略预算的 dummy steps，让场景稳定；驱动程序随后比较本体状态与图像指纹，不一致时直接停止该配对。模型看不到成功标志、对象真值位姿、深度图或力传感器数据。

这是两种不同动作抽象的系统级对比，不应把推理调用次数直接解释成单模型能力。报告成功率和端到端时间时，也应同时报告 π0.5 推理延迟、VLM/Jev 各自的请求数与延迟。

## 推荐的双 H20 分配

```text
H20:0  openpi π0.5 policy server :8000
H20:1  本地 OpenAI-compatible VLM :8001
CPU    LIBERO worker + phase2 experiment driver
网络   Jev API（当前允许远端；没有远端 GPT）
```

H20 显存足以做 π0.5 推理；两个模型分别占用一张卡可避免显存争抢和不可控的调度抖动。正式计时前分别做一次预热，并保持两个服务常驻。

## 1. 准备三个隔离环境

下面假设本仓库和官方 `openpi` 仓库位于同一级目录。openpi 的依赖变化较快，应按其仓库当前 README 创建环境；本项目只通过官方 WebSocket client 与它通信，避免把 openpi、LIBERO 和本项目的 MuJoCo 依赖装在一起。

```bash
# A. 实验驱动环境
cd jev-embodied
python3.11 -m venv .venv-phase2
source .venv-phase2/bin/activate
python -m pip install -U pip
python -m pip install -e '.[phase2]'

# 将官方 client 安装到驱动环境；不在此环境加载 π0.5 权重
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git ../openpi
python -m pip install -e ../openpi/packages/openpi-client
```

LIBERO 使用已有的 `.venv-libero`，至少需要官方 LIBERO、robosuite、NumPy 和 Pillow，并能独立执行：

```bash
.venv-libero/bin/python -c 'import libero, robosuite, PIL, numpy; print("LIBERO worker ready")'
```

本地 VLM 建议使用单独环境，例如 vLLM 的 OpenAI-compatible server。模型不是协议的一部分，但一轮正式对比必须固定模型名、revision、量化方式和服务参数。默认示例为 Qwen2.5-VL-7B-Instruct。

## 2. 启动 π0.5 服务（GPU 0）

按官方 openpi 说明安装完成后，在一个长期运行的终端启动 LIBERO checkpoint：

```bash
cd ../openpi
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=gs://openpi-assets/checkpoints/pi05_libero
```

默认 WebSocket 端口是 `8000`。首次运行会下载权重，不要把下载时间计入实验。驱动侧严格复现官方 LIBERO 输入约定：两路 RGB 图旋转 180°、pad resize 到 `224×224`，输入键为 `observation/image`、`observation/wrist_image`、`observation/state` 和 `prompt`。返回动作必须是有限的 7 维向量；与官方评测一致，π0.5 反归一化后的动作不会被额外裁剪，而 VLM + Jev 的固定候选仍严格限制在 `[-1, 1]`。

## 3. 启动本地 VLM（GPU 1）

以下只是一个可替换的 OpenAI-compatible 示例：

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --host 127.0.0.1 --port 8001 --dtype bfloat16 --api-key local
```

若使用其他本地服务，只需支持 `/v1/chat/completions`、双图输入和 JSON object 输出。设置与服务相同的本地 key：

```bash
export PHASE2_VLM_API_KEY=local
```

## 4. 配置 Jev

当前二阶段允许 Jev 走远端 TypeSafe API：

```bash
export TYPESAFE_API_KEY='你的密钥'
export TYPESAFE_MODEL='固定的模型版本，例如 jev-1.13.0'
```

建议正式实验固定具体 Jev 版本，不使用会漂移的 `jev-latest`。密钥只从环境变量或系统凭据读取，不会写入 protocol 和 episode 文件。

## 5. 先做链路 smoke test

在 `jev-embodied` 目录、驱动环境激活后运行：

```bash
jev-embodied phase2-compare \
  --manifest benchmarks/libero-phase2-smoke.json \
  --output runs/phase2-smoke-001 \
  --worker-python .venv-libero/bin/python \
  --modes pi05 vlm-jev \
  --pi05-host 127.0.0.1 --pi05-port 8000 \
  --pi05-replan-steps 5 \
  --vlm-base-url http://127.0.0.1:8001/v1 \
  --vlm-model Qwen/Qwen2.5-VL-7B-Instruct \
  --max-steps 100 --settle-steps 10 --max-calls 100 --timeout 1800 \
  --action-repeat 5 --candidate-scale 0.5 \
  --continue-on-error
```

每个 `--output` 必须是尚不存在的新目录，避免覆盖实验。检查 `report.json` 中两条路线均为 `complete: true`，并确认 episode 的 `initial_fingerprint` 一致。smoke 成功后再扩大清单和步数。

## 6. 正式配对实验

仓库已有两个开发任务的清单，可以先用于工程对比：

```bash
jev-embodied phase2-compare \
  --manifest benchmarks/libero-vision-compare.json \
  --output runs/phase2-dev-001 \
  --worker-python .venv-libero/bin/python \
  --modes pi05 vlm-jev \
  --pi05-host 127.0.0.1 --pi05-port 8000 \
  --pi05-replan-steps 5 \
  --vlm-base-url http://127.0.0.1:8001/v1 \
  --vlm-model Qwen/Qwen2.5-VL-7B-Instruct \
  --max-steps 400 --settle-steps 10 --max-calls 400 --timeout 3600 --max-usd 10 \
  --action-repeat 5 --candidate-scale 0.5 \
  --request-retries 1 --continue-on-error
```

开发清单只有两局，不能支持有说服力的成功率结论。完成链路验证后，应另外冻结包含多个 task、init index 和 seed 的评测清单；不要根据中途结果挑选样本。两种模式必须始终成对运行，并使用完全相同的预算。

## 输出文件

- `protocol.json`：冻结的参数、manifest、模型端点标识和关键源码 SHA-256。
- `report.json`：按模式聚合的成功率、环境步数、墙钟时间和完整性。
- `<case>--<mode>/episode.json`：逐次决策、实际执行动作、延迟、用量和状态。
- `<case>--<mode>/events.jsonl`：可流式恢复的事件记录。
- `<case>--<mode>/inputs/`：每次决策的原始双相机 PNG；VLM 路线还保存旋转后的实际输入 PNG。π0.5 的记录包含送入服务的 224×224 uint8 数组 SHA-256、shape 和 dtype。
- `<case>--<mode>/worker.log`：LIBERO/robosuite 的隔离日志。
- `reproduction/`：本轮使用的关键 Python 源码快照。

## 正式实验前的检查表

- 固定 openpi commit、π0.5 checkpoint URI、VLM revision、Jev 版本和本仓库 commit。
- 两个模型服务完成预热，GPU 上没有其他任务；记录 `nvidia-smi` 和 CUDA/驱动版本。
- smoke 中没有动作越界、schema validation、reset fingerprint 或 worker 错误。
- 输出盘空间足够，所有 episode 都有两路输入图和完整事件记录。
- 先冻结评测 manifest，再开始批量运行；失败局保留，不人工重跑后择优。
