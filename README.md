# jev-embodied

一个在本机运行的具身智能实验平台：使用 MuJoCo 模拟 Franka Panda 机械臂，在统一任务、种子和指标下比较规则策略、Jev 与其他模型。

当前 README 只记录本仓库内已经可以运行、并重新验证过的 Panda 实验。Meta-World、LIBERO 和真实机械臂不属于当前主实验，因此不在这里展示结果或链接。

## 当前实验范围

| 组成 | 当前内容 |
|---|---|
| 机器人与物理 | MuJoCo 3.13.0 + Franka Panda |
| 内置任务 | 搬运入盘 `transfer`、方块堆叠 `stack`、越障搬运 `barrier` |
| 决策方式 | 规则基线、Jev、OpenAI 兼容接口、本地模型 |
| 观测方式 | 仿真状态 `privileged`、RGB-D 几何估计 `rgbd`、原始图像 `vision` |
| 主要指标 | 成功率、运行时间、控制周期、模型调用等待、估算费用、违规接触 |

三项任务使用同一套物理成功条件：物体与目标中心的水平距离小于 2.5 cm，在目标支撑面稳定至少 0.4 秒，夹爪已张开，并且末端执行器已抬高到 0.17 m 以上。

## 安装与启动

需要 Conda、Python 3.11 或更高版本，以及 Node.js。

```bash
git clone https://github.com/zyh3699/jev-embodied.git
cd jev-embodied

conda create -n jev-embodied python=3.12 -y
conda activate jev-embodied
python -m pip install -e .

npm ci
npm run build
jev-embodied serve --port 8090
```

浏览器打开 <http://127.0.0.1:8090>。以后再次启动只需要：

```bash
cd jev-embodied
conda activate jev-embodied
jev-embodied serve --port 8090
```

## 已重新验证的实验

验证日期：2026-09-24。设备上的实际耗时会随硬件和系统负载变化。

| 实验 | 观测 | 相机 | 局数 | 成功率 | 平均耗时/局 | 总控制周期 | 违规接触 | API 调用 | 费用 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 规则基线 A | 仿真状态 | 无 | 9 | 9/9（100%） | 1.307 s | 72 | 0 | 0 | $0 |
| 规则基线 B | RGB-D 几何估计 | 外部相机 | 9 | 9/9（100%） | 1.618 s | 72 | 0 | 0 | $0 |
| 逐步 XYZ 基线 | 仿真状态 | 无 | 9 | 9/9（100%） | 1.751 s | 360 | 0 | 0 | $0 |
| Jev 1.13.0 技能选择 | 仿真状态 | 无 | 9 | 9/9（100%） | 7.109 s | 72 | 0 | 117 | $0.0049 |
| Jev 1.13.0 逐步 XYZ 冒烟 | 仿真状态 | 无 | 1 | 0/1（0%） | 23.890 s | 50 | 0 | 50 | $0.0073 |
| GPT-6 Astra 技能冒烟 | 仿真状态 | 无 | 1 | 0/1（0%） | 214.120 s | 5 | 0 | 10 | $0.2169 |

前三组是规则策略，均覆盖 3 个任务 × 3 个种子（0、1、2）。Jev 高层技能选择也完成了相同的 9 局；逐步 XYZ 先运行 1 局冒烟测试，失败后停止扩展，避免重复消耗 API 费用。

[查看规则基线逐任务结果](docs/results/PANDA_BASELINE_2026-09-24.md) · [查看 Jev 逐局结果与失败分析](docs/results/PANDA_JEV_2026-09-24.md) · [查看 GPT-6 Astra 冒烟记录](docs/results/PANDA_GPT6_ASTRA_2026-09-24.md)

### 实验 1：内部状态规则基线

```bash
conda activate jev-embodied
jev-embodied benchmark \
  --provider baseline \
  --control-mode skills \
  --observation-mode privileged \
  --cameras none \
  --tasks transfer stack barrier \
  --seeds 0 1 2 \
  --threshold 0 \
  --max-cycles 30 \
  --timeout 120 \
  --output runs/revalidation-2026-09-24/baseline-skills-privileged.json
```

这个实验先确认机器人、任务、规则动作和物理成功判定都能工作。它直接读取仿真中的物体、目标和机器人状态，不使用相机或模型 API。

### 实验 2：外部 RGB-D 规则基线

```bash
conda activate jev-embodied
jev-embodied benchmark \
  --provider baseline \
  --control-mode skills \
  --observation-mode rgbd \
  --cameras external \
  --tasks transfer stack barrier \
  --seeds 0 1 2 \
  --threshold 0 \
  --max-cycles 30 \
  --timeout 120 \
  --output runs/revalidation-2026-09-24/baseline-skills-rgbd-external.json
```

这个实验从外部相机的 RGB 与深度图估计物体和目标坐标，再交给同一个规则策略。macOS 必须在有图形会话的普通终端中运行；SSH 或受限后台进程可能无法建立 CoreGraphics 上下文。

### 实验 3：逐步 XYZ 规则基线

```bash
conda activate jev-embodied
jev-embodied benchmark \
  --provider baseline \
  --control-mode incremental \
  --observation-mode privileged \
  --cameras none \
  --tasks transfer stack barrier \
  --seeds 0 1 2 \
  --threshold 0 \
  --max-cycles 100 \
  --timeout 120 \
  --output runs/revalidation-2026-09-24/baseline-incremental-privileged.json
```

这个实验每轮只选择一次 X/Y/Z 小步移动或夹爪动作，控制粒度与后续 Jev 对照一致。它是比较 Jev 决策效果的直接规则基线。

### 实验 4：Jev 高层技能选择

网页连接测试确认 `jev-latest` 实际返回 `jev-1.13.0`。正式组使用以下命令：

```bash
conda activate jev-embodied
jev-embodied evaluate \
  --manifest benchmarks/builtin-dev.json \
  --output runs/revalidation-2026-09-24/jev-skills-dev \
  --policy jev \
  --connection-source saved \
  --control-mode skills \
  --observation-mode privileged \
  --max-steps 30 \
  --max-calls 60 \
  --timeout 180 \
  --threshold 0
```

结果为 9/9 成功，累计运行 63.98 秒，其中 API 等待约 49.36 秒。117 次调用共报告 115,544 个输入 token；按页面中的 Jev 输入价格 $0.042/百万估算，本组费用约 $0.0049。

逐步 XYZ 冒烟测试则在 50 次调用后耗尽预算，物体没有被抬起。模型在 `close/open` 间反复切换，并移动到了目标上方。该结果表明当前任务中，Jev 适合选择高层技能，但不能据此认为它能稳定完成低层闭环控制。

## GPT-6 Astra 状态

OpenLux 第一次连接测试在 61.253 秒后超时，第二次在 15.158 秒后通过。随后只运行了 `transfer / seed 0` 技能冒烟：前 5 个阶段成功执行并把方块搬到目标上方，但第 10 次 API 调用发生 `ReadTimeout`，任务未能释放方块并完成验证。

该局 214.12 秒中约 212.82 秒用于等待 API，估算费用约 $0.2169。由于单局已经出现高延迟失败且成本约为 Jev 9 局总成本的 44.7 倍，本轮不扩展到 9 局。

## 指标解释

| 指标 | 含义 |
|---|---|
| 成功率 | 达到物理成功条件的局数 / 已运行局数 |
| 总运行时间 | 从实验开始到结束的墙钟时间，包含物理仿真与模型等待 |
| 模型等待 | 仅统计模型请求延迟，用于区分推理等待和仿真耗时 |
| 控制周期 | 完成任务所用的决策—执行循环数，越少不一定越安全 |
| 估算费用 | 按页面中配置的模型单价估算，最终以 API 服务商账单为准 |
| 违规接触 | 机械臂与桌面或障碍发生不允许接触的仿真步数 |

比较模型时不能只看是否成功：至少同时检查成功率、耗时、费用和违规接触。3 个种子只用于快速验证，不足以形成稳定的统计结论；正式比较应扩大种子数量并报告失败案例。

## 开发验证

```bash
conda activate jev-embodied
python -m pytest -q
npm test
```

项目采用 [MIT License](LICENSE)。更详细的实现说明见 [技术指南](docs/TECHNICAL_GUIDE.md)。
