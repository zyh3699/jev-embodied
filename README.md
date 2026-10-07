# jev-embodied

`jev-embodied` 是一个具身智能对比实验项目，研究 Jev 适合在机器人系统中承担什么角色，以及它与通用大模型、视觉语言模型和端到端 VLA 的关系。

项目分为两个阶段：

| 阶段 | 研究问题 | 决策接口 | 主要环境 |
|---|---|---|---|
| 一阶段 | 面对结构化候选动作，Jev 与 GPT-6 Astra 的效率和成功率有何差异？ | 离散技能或分层子目标 | MuJoCo Panda、Meta-World、LIBERO |
| 二阶段 | 当所有路线最终都输出连续机器人动作时，VLM+Jev 与端到端 VLA 的差距在哪里？ | 连续 `H×7` 动作块 | LIBERO、Franka Panda |

所有成功率都来自仿真环境的官方物理判定，不使用模型自评。GIF 和视频由运行时保存的真实相机帧或仿真状态生成，并保留模型等待时间。

## 总体框架

### 一阶段：结构化决策

```text
任务与环境状态
      ↓
证据、阶段状态和候选技能
      ↓
┌──────────────┬──────────────┐
│     Jev      │ GPT-6 Astra  │
└──────────────┴──────────────┘
      ↓
确定性技能控制器
      ↓
机器人连续运动
```

这一阶段不要求模型直接输出关节或末端动作。模型只选择 `approach`、`grasp`、`transport`、`release` 等候选，连续控制由同一个代码控制器完成。

### 二阶段：连续动作对比

```text
π0.5 路线
双相机图像 + 任务 + 本体状态
               ↓
             π0.5
               ↓
          连续 H×7 动作块

VLM + Jev 路线
双相机图像 + 任务 + 本体状态
               ↓
      Qwen3.5 语义关键帧
               ↓
       RGB-D 几何与候选生成
               ↓
      Jev 选择一个 H×7 动作块
               ↓
             执行
```

无 Jev 消融路线使用相同的 VLM 和 RGB-D 候选，但固定执行 `nominal` 动作块。它用于区分“候选生成能力”和“Jev 候选裁决能力”。

## 实验口径

- **配对原则**：同组路线共享任务、seed、初始状态、相机和资源上限。
- **成功判定**：只采用 MuJoCo、Meta-World 或 LIBERO 的官方物理谓词。
- **端到端时间**：包含模型推理、网络请求、仿真和机器人执行。
- **动作接口**：二阶段所有路线最终都执行有限、连续的 7 维动作序列。
- **失败记录**：步数耗尽、超时、模型格式错误和接口错误都保留在原始报告中。
- **费用口径**：依据 API token 用量和统一单价估算，仅用于项目内部比较。

# 一阶段：Jev 与 GPT 的结构化决策对比

## 实验设置

| 环境 | 任务 | 模型看到的输入 | 模型输出 |
|---|---|---|---|
| MuJoCo Panda | `transfer` | 任务目标、物体状态、可用技能 | 下一项技能 |
| Meta-World | reach、push、pick-place、drawer、door | 阶段状态、目标关系、候选子目标 | 子目标或技能 |
| LIBERO | 关闭抽屉 | GPT 视觉规划生成的候选和双相机证据 | 候选动作及速度 |

Jev 与 GPT-6 Astra 使用相同候选集合和确定性控制器。实验重点是局部结构化决策的成功率、墙钟时间和估算费用。

## 汇总结果

| 环境与范围 | Jev 路线 | GPT 路线 | 观察 |
|---|---:|---:|---|
| Panda `transfer`，seed 0 | 1/1；7.39 s；$0.000538 | 1/1；47.55 s；$0.5941 | 成功相同；Jev 快 6.4× |
| Meta-World 3 任务 × 2 seeds | 5/6；126.36 s；$0.0183 | 5/6；1075.66 s；$15.9184 | 成功相同；Jev 快 8.5× |
| Meta-World drawer/door × 2 seeds | 4/4；115.64 s；≥$0.0143 | 3/4；1403.73 s；≥$10.4788 | Jev 快 12.1× |
| LIBERO 关闭抽屉，init 0 | GPT 候选 + Jev：0/1；505 步 | GPT 候选 + GPT：0/1；310 步 | 两条路线均未完成任务 |

样本规模较小。这些结果支持“Jev 适合高频、结构化、候选有限的局部选择”，但不能推导出它对所有具身任务都有更高成功率。

完整数据见[一阶段实验报告](docs/results/reproduction-fixed-2026-09-24/RESULTS.md)。

## Panda：技能选择

两条路线在同一个 Panda 场景中完成相同 transfer 任务。墙钟 GIF 包含 API 等待时间。

![Panda Jev 与 GPT 墙钟对照](docs/results/reproduction-fixed-2026-09-24/panda/wallclock-comparison.gif)

[完整 MP4](docs/results/reproduction-fixed-2026-09-24/panda/wallclock-comparison.mp4) · [Jev 轨迹](docs/results/reproduction-fixed-2026-09-24/panda/jev-transfer.mp4) · [GPT 轨迹](docs/results/reproduction-fixed-2026-09-24/panda/gpt-transfer.mp4)

## Meta-World：分层子目标

`reach-v3`、`push-v3` 和 `pick-place-v3` 各使用两个相同 seeds。两条路线都在 `push-v3` seed 0 达到步数上限，其余五局成功。

![Meta-World 六组配对实验](docs/results/reproduction-fixed-2026-09-24/metaworld/paired-wallclock.gif)

抽屉和门实验为控制器提供通用阶段状态与候选接触动作，成功仍由 Meta-World 官方判定。

![Meta-World 抽屉和门任务](docs/results/reproduction-fixed-2026-09-24/metaworld-fixtures/paired-wallclock.gif)

[统计图](docs/results/reproduction-fixed-2026-09-24/metaworld/comparison/comparison.png) · [fixture 统计](docs/results/reproduction-fixed-2026-09-24/metaworld-fixtures/summary.json)

## LIBERO：视觉候选选择

两条路线共享 GPT 双相机视觉规划器，区别仅在于由 GPT 或 Jev 选择候选动作。两组都未通过关闭抽屉的官方判定，说明候选本身缺少可靠接触轨迹时，替换选择器并不能解决任务。

![一阶段 LIBERO 配对回放](docs/results/reproduction-fixed-2026-09-24/libero/drawer-supervisor-v2/comparison.gif)

# 二阶段：π0.5 与 VLM+Jev 连续动作实验

## 实验设置

| 项目 | 设置 |
|---|---|
| 仿真环境 | LIBERO / robosuite 1.4.1，OSC_POSE，20 Hz |
| 机器人 | Franka Panda |
| VLA | 官方 `pi05_libero` checkpoint |
| 本地 VLM | `Qwen/Qwen3.5-27B`，BF16，关闭 thinking |
| Jev | 远端 Jev API；二阶段不调用远端 GPT |
| 服务器 | 2 × NVIDIA H20 |
| 观测 | external + wrist RGB/RGB-D、任务文本、本体状态 |
| 执行 | 每次决策最多执行 5 个环境步，默认每局最多 400 步 |

## 对照路线

| 路线 | 规划 | 动作选择 | 输出 |
|---|---|---|---|
| `pi05` | π0.5 端到端完成 | π0.5 | 连续 `H×7` |
| `vlm-jev-triggered` | Qwen 只在首次、停滞、阶段切换或低置信时规划 | Jev 每轮选择候选 | 连续 `H×7` |
| `vlm-jev-dense` | Qwen 每轮重新规划 | Jev 每轮选择候选 | 连续 `H×7` |
| `vlm-chunk-no-jev` | Qwen 按需规划 | 固定选择 nominal 候选 | 连续 `H×7` |

## 实验序列

### 实验 A：连续动作接口与机械装置任务

任务为关闭抽屉和关闭微波炉，每个任务使用一个固定初态，四条路线共享 400 步上限。Qwen 输出视觉关键帧，通用生成器将其转换为连续候选动作块。

| 路线 | 成功 | 平均步数 | 平均墙钟时间 | VLM / Jev 请求 |
|---|---:|---:|---:|---:|
| π0.5 | 1/2 | 323.5 | 68.07 s | 0 / 0 |
| Triggered VLM + Jev | 0/2 | 400 | 502.06 s | 129 / 219 |
| Dense VLM + Jev | 0/2 | 400 | 589.99 s | 160 / 160 |
| Triggered VLM, no Jev | 0/2 | 400 | 189.93 s | 40 / 0 |

关闭抽屉：

![连续动作关闭抽屉](docs/results/phase2-action-chunks-2026-10-05/media/drawer-init0/comparison.gif)

关闭微波炉：

![连续动作关闭微波炉](docs/results/phase2-action-chunks-2026-10-05/media/microwave-init0/comparison.gif)

[完整实验 A 报告](docs/results/phase2-action-chunks-2026-10-05/RESULTS.md)

### 实验 B：受控桌面抓放任务

这一实验将任务范围固定为“识别指定包装物体，顶部抓取并放入开放篮子”。公开静态纹理只用于外观匹配，不提供当前场景位姿、分割或成功标签。两个任务均采用固定初态。

| 路线 | 成功 | 平均步数 | 平均墙钟时间 | VLM / Jev 请求 |
|---|---:|---:|---:|---:|
| π0.5 | 2/2 | 129.5 | 34.36 s | 0 / 0 |
| VLM, no Jev | 2/2 | 370.5 | 85.51 s | 22 / 0 |
| Triggered VLM + Jev | 2/2 | 485.5 | 154.43 s | 35 / 196 |
| Dense VLM + Jev | 2/2 | 483.0 | 372.37 s | 96 / 194 |

Alphabet soup：

![Alphabet soup 四路线成功对照](docs/results/phase2-reference-grounded-2026-10-06/media/alphabet-soup-init0/comparison.gif)

Cream cheese：

![Cream cheese 四路线成功对照](docs/results/phase2-reference-grounded-2026-10-06/media/cream-cheese-init0/comparison.gif)

四条路线共 8 局均通过官方成功判定。π0.5 用时和步数最少；Jev 路线能够完成任务，但倾向谨慎候选，且每轮远端请求增加了墙钟时间。

[完整实验 B 报告](docs/results/phase2-reference-grounded-2026-10-06/RESULTS.md)

### 实验 C：任务类型泛化

开发集和预先冻结的 holdout 各包含五类关系与操作：抓取放置、空间指代、打开抽屉、推动物体和操作炉灶。混合路线不使用静态纹理、候选区域编号或任务命名宏技能。

| 数据集 | 路线 | 官方成功 | 平均步数 | 平均墙钟时间 |
|---|---|---:|---:|---:|
| 开发 5 任务 | π0.5 | 5/5 | 106.4 | 37.75 s |
| 开发 5 任务 | VLM, no Jev | 0/5 | 320.0 | 81.39 s |
| 开发 5 任务 | Triggered VLM + Jev | 0/5 | 233.0 | 113.83 s |
| Holdout 5 任务 | π0.5 | 5/5 | 103.4 | 46.21 s |
| Holdout 5 任务 | Triggered VLM + Jev | 0/5 | 37.0 | 75.21 s |

推盘任务：

![推盘三路线实验](docs/results/phase2-generalization-2026-10-07/media/push-three-way.gif)

空间指代任务：

![空间关系三路线实验](docs/results/phase2-generalization-2026-10-07/media/spatial-three-way.gif)

打开抽屉：

![打开抽屉三路线实验](docs/results/phase2-generalization-2026-10-07/media/drawer-three-way.gif)

π0.5 在十个任务上全部成功。通用 VLM+Jev 的主要限制是缺少学习式 6D 末端姿态、接触状态下的短时闭环控制，以及遮挡后的稳定目标追踪。Jev 只能在已有候选中选择，不能创造候选集合中不存在的动作轨迹。

[完整实验 C 报告与五组 GIF](docs/results/phase2-generalization-2026-10-07/RESULTS.md)

### 实验 D：最小假设下的桌面抓放

这一实验使用同一套代码处理四个不同包装物体，并显式声明以下范围：

- 源物体可与公开 LIBERO 静态纹理做外观匹配；
- LIBERO 桌面候选采用世界坐标 `z=0` 支撑面；
- 采用顶部抓取；
- 目的地是开放容器，内部点由 RGB-D 容器几何估计；
- 不使用任务专用坐标、物体真值位姿或成功信号。

| 实验 | alphabet soup | cream cheese | salad dressing | milk | 合计 |
|---|---:|---:|---:|---:|---:|
| 独立 smoke | 成功，353 步 | — | — | — | 1/1 |
| 冻结重复 1 | 成功，473 步 | 运行时中止 | 运行时中止 | 成功，364 步 | 2/4 |
| 冻结重复 2 | 成功，468 步 | 运行时中止 | 运行时中止 | 运行时中止 | 1/4 |

成功轨迹：

![最小假设 alphabet soup 成功](docs/results/phase2-minimal-assumption-2026-10-07/media/alphabet-soup-success.gif)

![最小假设 milk 成功](docs/results/phase2-minimal-assumption-2026-10-07/media/milk-success.gif)

运行时中止轨迹：

![Cream cheese 中止轨迹](docs/results/phase2-minimal-assumption-2026-10-07/media/cream-cheese-failure.gif)

![Salad dressing 中止轨迹](docs/results/phase2-minimal-assumption-2026-10-07/media/salad-dressing-failure.gif)

运行时中止来自本地 Qwen 返回不完整 JSON，不能算成功，也不是完整控制预算下的物理失败。该实验说明加入明确、有限的桌面假设后，VLM+Jev 可以完成部分抓放任务；它不代表开放任务泛化已经解决。

[完整实验 D 报告与原始记录](docs/results/phase2-minimal-assumption-2026-10-07/RESULTS.md)

## 二阶段结果总览

| 实验范围 | π0.5 | VLM, no Jev | VLM + Jev | 可以支持的结论 |
|---|---:|---:|---:|---|
| 机械装置开发任务 | 1/2 | 0/2 | 0/2 | 连续接口一致不等于控制能力一致 |
| 两个受控抓放任务 | 2/2 | 2/2 | 4/4（两种 Jev 调度） | 候选充分时，Jev 可以完成局部动作选择 |
| 十个多类型泛化任务 | 10/10 | 开发集 0/5 | 开发集与 holdout 均 0/5 | 当前通用候选缺少 VLA 的 6D 接触策略 |
| 四物体最小假设抓放 | 未重复运行 π0.5 | 未运行 | 最好一次 2/4 | 有限几何与外观假设可恢复部分成功 |

## 如何理解 Jev 的作用

Jev 的优势是从少量、明确、可执行的候选中进行高频局部裁决。它不负责从图像恢复完整机器人动力学，也不能弥补候选集合中缺失的正确 6D 动作。

无 Jev 路线有时墙钟时间更短并不矛盾：无 Jev 直接调用本地 `nominal_chunk()`，选择几乎没有成本；有 Jev 路线每轮都包含远端推理和网络等待，并可能选择更谨慎的动作。实验 B 中无 Jev 为 85.51 秒，Triggered Jev 为 154.43 秒，但两者都成功。因此应同时观察成功率、环境步数、模型调用时间和端到端时间，而不能把“Jev 单次判断比 VLM 规划轻量”解释为“Jev 一定比本地固定规则快”。

# 运行项目

## 基础安装

主平台使用 Python 3.12；Meta-World 和 LIBERO 建议使用独立 Python 3.11 环境。

```bash
git clone https://github.com/zyh3699/jev-embodied.git
cd jev-embodied

conda create -n jev-embodied python=3.12 -y
conda activate jev-embodied
python -m pip install -e '.[video,plots]'

npm ci
npm run build
jev-embodied serve --port 8090
```

浏览器打开 <http://127.0.0.1:8090>。模型凭据保存在本机连接存储中，不写入实验报告或仓库。

## 一阶段复现

Panda：

```bash
jev-embodied evaluate \
  --manifest benchmarks/builtin-smoke.json \
  --output runs/panda-jev \
  --policy jev --connection-source saved \
  --control-mode skills --observation-mode privileged \
  --max-steps 30 --max-calls 60 --timeout 600 \
  --validation-retries 1 --request-retries 1 --continue-on-error
```

Meta-World：

```bash
python scripts/run_metaworld_hierarchical_comparison.py \
  --output runs/metaworld-paired \
  --worker-python .venv-metaworld/bin/python \
  --plot-python "$(which python)" \
  --validation-retries 1 --request-retries 1
```

## 二阶段复现

双 H20 部署步骤见[服务器实验手册](docs/PHASE2_SERVER.md)。四路线运行示例：

```bash
jev-embodied phase2-compare \
  --manifest benchmarks/libero-phase2-supported-smoke.json \
  --output runs/phase2-smoke \
  --worker-python .venv-libero/bin/python \
  --modes pi05 vlm-jev-triggered vlm-jev-dense vlm-chunk-no-jev \
  --pi05-host 127.0.0.1 --pi05-port 8000 \
  --vlm-base-url http://127.0.0.1:8001/v1 \
  --vlm-model Qwen/Qwen3.5-27B \
  --continue-on-error
```

最小假设抓放实验使用：

```bash
jev-embodied phase2-compare \
  --manifest benchmarks/libero-phase2-assisted-final.json \
  --output runs/phase2-assisted \
  --worker-python .venv-libero/bin/python \
  --modes vlm-jev-triggered \
  --source-grounding appearance-reference \
  --vlm-base-url http://127.0.0.1:8001/v1 \
  --vlm-model Qwen/Qwen3.5-27B \
  --max-steps 550 --request-retries 2 --continue-on-error
```

实验输出目录必须不存在。正式实验前建议先运行单任务 smoke 清单，确认模型服务、仿真 worker、初始状态配对和媒体记录链路正常。

## 仓库结构

```text
frontend/       实验配置与运行界面
src/            Python 服务、策略、动作块生成与评测逻辑
benchmarks/     Panda、Meta-World、LIBERO 冻结任务清单
scripts/        实验运行、回放、绘图和结果校验脚本
docs/results/   协议、原始报告、结构化指标、GIF 和视频
```

## 验证

```bash
npm run build
python -m embodied_jev.cli --help
python -m unittest discover -s tests
```

项目采用 [MIT License](LICENSE)。
