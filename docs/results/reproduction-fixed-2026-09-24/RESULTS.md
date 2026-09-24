# 2026-09-24 配对复现实验

这批结果来自当前仓库在本机的重新运行，不引用旧项目视频。所有媒体由本批保存轨迹或相机帧生成，导出阶段模型调用数为 0。

## 一句话结论

Jev 在 Panda 和 Meta-World 原三任务中保持了与 GPT-6 Astra 相同的成功数，同时显著减少运行时间和估算费用。新增的开抽屉/开门任务中 Jev 为 4/4，GPT 为 3/4，但只有四个配对样本且 GPT 的失败包含接口/格式异常，不能据此宣称普遍能力提升。LIBERO 两组均未成功。

## 汇总

| 环境 | 方法 | 成功 | 墙钟时间 | 请求 | 估算费用 | 结束原因 |
|---|---|---:|---:|---:|---:|---|
| Panda transfer seed 0 | Jev | 1/1 | 7.39 s | 13 | $0.000538 | 成功 |
| Panda transfer seed 0 | GPT-6 Astra | 1/1 | 47.55 s | 13 | $0.59406 | 成功 |
| Meta-World 六局 | Jev | 5/6 | 126.359 s | 312 | $0.018264 | `push-v3` seed 0 步数耗尽 |
| Meta-World 六局 | GPT-6 Astra | 5/6 | 1075.659 s | 313 | $15.91835 | `push-v3` seed 0 步数耗尽 |
| Meta-World 开抽屉/开门四局 | Jev | 4/4 | 115.639 s | 225 | ≥$0.014334 | 全部成功；1 个失败请求无用量 |
| Meta-World 开抽屉/开门四局 | GPT-6 Astra | 3/4 | 1403.734 s | 208 | ≥$10.47884 | 抽屉 seed 1 运行中断；2 个请求无用量 |
| LIBERO drawer init 0 | GPT-6 Astra | 0/1 | 998.838 s | 59 | $5.05621 | 费用保护；310 步 |
| LIBERO drawer init 0 | GPT-6 + Jev | 0/1 | 1068.402 s | 103 | $5.09312 | 费用保护；505 步 |

LIBERO 混合组共有 52 次 GPT 请求和 51 次 Jev 请求，用量完整。在相近费用保护下，它比纯 GPT 多执行 195 步（约 63%），但两组成功数仍是 0。

本次修复与复跑还包含两轮未作为主结果展示的 LIBERO waypoint 诊断。把 Panda、Meta-World、最终 LIBERO 对照及这些诊断全部相加，按公开标准价可计算的费用下界为 **$36.2794**；其中两个失败请求没有返回 token 用量。该数字是估算实验消耗，不是 OpenLux 实收账单。

## 修复了什么

此前失败不全是“GPT 调不了”：

1. Panda 的 GPT 冒烟实验是 60 秒客户端读取超时。聊天请求上限提高到 120 秒后，同一任务成功。
2. Meta-World 的旧运行收到 HTTP 200，但偶发动作 JSON 不满足结构。现在只对验证失败重试，并把失败次数写入 episode；本轮 625 次请求中只有 GPT 的最后一局发生 1 次验证重试，最终成功。
3. LIBERO 的旧运行把错误像素反投影到工作区外，并在执行前终止。现在最多重试两次，把拒绝原因和原回答反馈给视觉规划器，同时仍由几何安全检查阻止越界动作。
4. 模型读取超时及瞬时 429/500/502/503/504/529 可在动作执行前重试一次；每次尝试仍单独计入请求和费用，避免把重试当免费调用。
5. 批量评测增加 `--continue-on-error`，单局运行错误不再抹掉后续配对用例。
6. Panda、Meta-World、LIBERO 都有从真实记录生成媒体的脚本；没有补造成功轨迹。

## Panda

- 环境：项目内置 MuJoCo Panda。
- 控制：模型选择高层离散技能，代码执行连续轨迹。
- 观测：仿真状态，不使用相机。
- 配对依据：同一 `scene_hash`、任务和 seed。

[同一墙钟时间轴动图](panda/wallclock-comparison.gif) · [墙钟对照视频](panda/wallclock-comparison.mp4) · [Jev 单独轨迹](panda/jev-transfer.mp4) · [GPT 单独轨迹](panda/gpt-transfer.mp4)

墙钟对照按记录的逐次 API 延迟显示等待，以总墙钟时间与模型延迟之差分配真实 qpos 帧；两侧统一 4× 播放。Jev 在 7.39 秒完成，GPT-6 Astra 在 47.55 秒完成。

额外的 Jev 九局开发集结果为 9/9、63.98 秒、估算 $0.004853；由于 GPT 本轮只运行一局，它不进入主配对图。

## Meta-World

- 版本：Meta-World 3.1.1、MuJoCo 3.3.0、Python 3.11 worker。
- 子集：`reach-v3`、`push-v3`、`pick-place-v3` × seeds 0/1；这是自定义开发子集，不是 MT10/ML10 榜单分数。
- 预算：每局 200 环境步、80 次调用尝试、`action-repeat=5`。
- 配对依据：六组 `initial_observation_sha256` 全部一致。
- 回放验证：12 条轨迹的全部保存观测、官方成功标记和初始哈希重新执行后完全一致，最大数值误差为 0。

[墙钟对照 GIF](metaworld/paired-wallclock.gif) · [墙钟对照视频](metaworld/paired-wallclock.mp4) · [按环境步核对的原回放](metaworld/paired-grid.mp4) · [统计图](metaworld/comparison/comparison.png) · [逐局数据](metaworld/comparison/episodes.csv) · [图表说明](metaworld/comparison/FIGURE_NOTES.md)

两组产生了相同的 765 个环境步和 5/6 成功数；差异主要来自接口效率。Jev 请求延迟中位数为 365 ms，GPT 为 3019 ms。
墙钟动图让每排六局按实际顺序运行，在统一时间轴上 40× 播放；每次模型等待由记录的两层调用延迟重建，动作仍来自逐步验证过的原始轨迹。

### Meta-World 固定装置任务扩展

- 子集：`drawer-open-v3`、`door-open-v3` × seeds 0/1。
- 协议：与原实验相同的特权状态观测、两层离散决策、200 步、80 次调用尝试和 `action-repeat=5`。
- 适配器：新增接近把手、贴合把手、拉抽屉和转动门四类子目标，并锁存已经完成的阶段，避免模型在接近/贴合之间来回切换。
- 配对依据：四组初始状态哈希全部一致；8 条保存轨迹逐步重放，观测和官方成功标记最大误差为 0。
- 结果：Jev 4/4，115.639 秒；GPT 3/4，1403.734 秒。GPT 抽屉 seed 1 在一次 120 秒读取超时后重试，随后连续两次没有返回约束内的规划选项，按原协议保留为运行失败。
- 费用：Jev 已知下界 $0.014334，GPT 已知下界 $10.47884。失败请求没有 token 用量，因此不把缺失部分记为 0。

[墙钟对照 GIF](metaworld-fixtures/paired-wallclock.gif) · [墙钟对照视频](metaworld-fixtures/paired-wallclock.mp4) · [成功/时间/价格图](metaworld-fixtures/summary.png) · [绘图审计数据](metaworld-fixtures/summary.json) · [完整统计图](metaworld-fixtures/comparison/comparison.png) · [逐局数据](metaworld-fixtures/comparison/episodes.csv)

## LIBERO

- 版本：固定 LIBERO revision `8f1084e`、MuJoCo 3.5.0、Python 3.11 worker。
- 任务：LIBERO-90 task 0，`close the top drawer of the cabinet`，init 0、seed 0。
- 观测：external + wrist RGB-D、TCP 本体反馈；模型不读取物体真值或官方成功信号。
- 设计：`supervisor-v2`。两组共享 GPT 时序视觉候选生成和确定性数值伺服；GPT 或 Jev 只选择候选、正常/谨慎速度或重新观测。
- 配对依据：初始状态、settled 状态、相机变换、控制器和仿真版本哈希一致。
- 保护：1200 秒墙钟上限、每局 $5 费用 admission guard；费用可能略超阈值，因为先完成当前请求再检查。

[README 加速动图](libero/drawer-supervisor-v2/comparison.gif) · [包含真实等待的配对视频](libero/drawer-supervisor-v2/comparison.mp4) · [末帧](libero/drawer-supervisor-v2/final.png)

视觉越界与瞬时网络错误已有动作前重试，两组都真实执行了机械臂。纯 GPT 在 310 步、混合组在 505 步达到约 `$5` 费用保护；抽屉都未达到官方关闭阈值。逐检查点记录长期为 `visual_progress=unchanged`，主要瓶颈是共享 GPT 视觉候选没有稳定形成有效接触几何，而不是接口完全不可用。

## 图表与原始记录

- [三环境总览 PNG](final-overview-v2/overview.png)
- [三环境总览 SVG](final-overview-v2/overview.svg)
- [总览绘图数据](final-overview-v2/metrics.json)
- Panda 原始 episode：`runs/revalidation-2026-09-24/jev-skills-dev/transfer-0-episode.json` 与 `runs/reproduction-fixed-2026-09-24/panda-gpt/transfer-0-episode.json`
- Meta-World 原始批次：`runs/reproduction-fixed-2026-09-24/metaworld-paired/`
- Meta-World 固定装置任务原始批次：`runs/metaworld-fixtures-paired-2026-09-24/`
- LIBERO 主结果原始批次：`runs/reproduction-fixed-2026-09-24/libero-drawer-supervisor-v2/`

`runs/` 默认作为本机大体积原始记录使用；README 引用的图、视频、CSV 和指标 JSON 已放在本目录。
