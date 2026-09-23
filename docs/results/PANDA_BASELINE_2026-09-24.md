# Panda 基线复验记录（2026-09-24）

本记录只包含当天在本仓库中实际完成的实验。运行环境为本机 Conda 环境 `jev-embodied`，MuJoCo 3.13.0。

## 固定条件

- 任务：`transfer`、`stack`、`barrier`
- 种子：0、1、2
- 决策：规则基线 `baseline`
- 控制：前两组为预设技能 `skills`，第三组为逐步 XYZ `incremental`
- 最大控制周期：30
- 单局超时：120 秒
- 决策阈值：0

## 汇总

| 观测配置 | 完成 | 成功 | 平均耗时/局 | 中位耗时/局 | 累计耗时 | 总周期 | 违规接触 | 模型调用 | 费用 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `privileged` + 无相机 | 9/9 | 9/9 | 1.307 s | 1.280 s | 11.76 s | 72 | 0 | 0 | $0 |
| `rgbd` + 外部相机 | 9/9 | 9/9 | 1.618 s | 1.600 s | 14.56 s | 72 | 0 | 0 | $0 |
| `privileged` + 无相机，逐步 XYZ | 9/9 | 9/9 | 1.751 s | 1.740 s | 15.76 s | 360 | 0 | 0 | $0 |

## 逐局结果

| 任务 | 种子 | 内部状态技能：成功 / 周期 / 耗时 | RGB-D 技能：成功 / 周期 / 耗时 | 逐步 XYZ：成功 / 周期 / 耗时 |
|---|---:|---|---|---|
| transfer | 0 | 是 / 8 / 1.28 s | 是 / 8 / 1.61 s | 是 / 42 / 1.92 s |
| transfer | 1 | 是 / 8 / 1.29 s | 是 / 8 / 1.60 s | 是 / 39 / 1.69 s |
| transfer | 2 | 是 / 8 / 1.28 s | 是 / 8 / 1.80 s | 是 / 40 / 1.74 s |
| stack | 0 | 是 / 8 / 1.28 s | 是 / 8 / 1.63 s | 是 / 41 / 1.71 s |
| stack | 1 | 是 / 8 / 1.29 s | 是 / 8 / 1.59 s | 是 / 38 / 1.63 s |
| stack | 2 | 是 / 8 / 1.26 s | 是 / 8 / 1.59 s | 是 / 39 / 1.75 s |
| barrier | 0 | 是 / 8 / 1.28 s | 是 / 8 / 1.60 s | 是 / 42 / 1.85 s |
| barrier | 1 | 是 / 8 / 1.50 s | 是 / 8 / 1.58 s | 是 / 39 / 1.79 s |
| barrier | 2 | 是 / 8 / 1.30 s | 是 / 8 / 1.56 s | 是 / 40 / 1.68 s |

技能控制两组全部为 0 次违规接触、0 次模型 API 调用。内部状态组的最大抬升高度为 0.19699–0.19853 m；RGB-D 组为 0.19696–0.19855 m。

逐步 XYZ 组也为 9/9 成功、0 次违规接触和 0 次 API 调用。各局使用 38–42 个周期，平均 40 个；这组是后续 Jev 逐步 XYZ 实验的直接对照。

## 复现命令

```bash
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

原始逐局 JSON 由命令写入 `runs/revalidation-2026-09-24/`。`runs/` 已被 Git 忽略，用于防止把本地运行日志、连接信息或大体积结果误提交到仓库。

## 结论边界

这次实验只证明：三项内置任务、技能与逐步 XYZ 规则动作链、物理成功判定，以及外部 RGB-D 几何估计链路能够在当前机器上完成。它没有调用 Jev 或视觉语言模型，因此不能据此评价任何模型的决策能力。每组 9 局适合冒烟验证，不适合宣称统计优势。
