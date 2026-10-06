# 二阶段参考外观 grounding 实验（2026-10-06）

## 结论

在两个固定的 LIBERO-90 `libero_object` 开发初态上，`π0.5`、本地 `Qwen3.5-27B` 无 Jev、按需 `Qwen3.5 + Jev`、逐轮 `Qwen3.5 + Jev` 共 8 局全部通过官方 `env.check_success()`。这说明修正后的混合控制器能够完成这两类“取物并放入篮子”任务，但样本规模只有 2 个固定初态，不能解释为全任务或跨场景泛化已经解决。

| 模式 | 成功 | 平均步数 | 平均墙钟时间 | VLM 请求 | Jev 请求 | 已知 Jev 费用 |
|---|---:|---:|---:|---:|---:|---:|
| `pi05` | 2/2 | 129.5 | 34.36 s | 0 | 0 | $0 |
| `vlm-chunk-no-jev` | 2/2 | 370.5 | 85.51 s | 22 | 0 | $0 |
| `vlm-jev-triggered` | 2/2 | 485.5 | 154.43 s | 35 | 196 | $0.026125 |
| `vlm-jev-dense` | 2/2 | 483.0 | 372.37 s | 96 | 194 | $0.025575 |

π0.5 在这两局中步数和时间都明显更少。按需 Jev 与 dense Jev 都成功，但 Jev 经常选择更谨慎的进度尺度，所以本轮不能声称 Jev 提升了速度；dense 的主要额外开销来自频繁调用本地 VLM。Jev 的实验意义是验证“从多个可执行连续动作块中做低延迟局部裁决”的架构，而不是替代视觉 grounding 或连续控制器。

## 逐局结果

| 任务 | 模式 | 成功 | 步数 | 墙钟时间 | 模型调用 |
|---|---|---:|---:|---:|---|
| alphabet soup → basket | `pi05` | 是 | 132 | 34.57 s | π0.5 27 次 |
| alphabet soup → basket | `vlm-chunk-no-jev` | 是 | 379 | 86.55 s | Qwen 11 次 |
| alphabet soup → basket | `vlm-jev-triggered` | 是 | 499 | 161.24 s | Qwen 19；Jev 100 |
| alphabet soup → basket | `vlm-jev-dense` | 是 | 494 | 405.83 s | Qwen 55；Jev 99 |
| cream cheese → basket | `pi05` | 是 | 127 | 34.14 s | π0.5 26 次 |
| cream cheese → basket | `vlm-chunk-no-jev` | 是 | 362 | 84.46 s | Qwen 11 次 |
| cream cheese → basket | `vlm-jev-triggered` | 是 | 472 | 147.62 s | Qwen 16；Jev 96 |
| cream cheese → basket | `vlm-jev-dense` | 是 | 472 | 338.90 s | Qwen 41；Jev 95 |

同一任务的四种模式具有相同初始状态 SHA-256：alphabet soup 为 `b23dc6…8808`，cream cheese 为 `e8d0d5…4a3f`。成功只取 LIBERO 环境判定，不使用 VLM/Jev 自评。

## 为什么上一版失败、这一版成功

上一版虽然已经把输出统一成连续 `H×7`，但把视觉方向直接当作接触控制依据，导致目标漏检、桌面深度泄漏、抓取点过高、抓住后重新规划回源物体，以及释放高度不稳定。修正版保留统一动作接口，同时补齐了任务无关的控制层：

- 从 RGB-D 构建与任务无关的候选区域，使用相机标定反投影到世界坐标；过滤桌面泄漏，并针对扁平物体降低最小高度阈值。
- Qwen 只做语义像素/关键帧建议；公开静态 LIBERO 纹理只用于外观匹配，不提供当前场景位姿、分割或成功标签。
- 抓取拆成高位 XY 对齐、下降、闭合、夹紧和抬升；放置拆成安全抬升、XY 运输、下降和张爪。
- 显式维护 `source → destination → verification` 阶段。接触和运输阶段锁定几何目标，避免中途视觉重规划破坏已经建立的接触。
- Jev 看到的是多个具体 `H×7` 数组及观测证据；自由空间候选均保持正向进度，夹爪候选只在接触或释放阶段出现。

这里仍不存在 `pick_alphabet_soup`、`put_in_basket` 等任务命名宏技能。混合路线比 π0.5 多了一层通用几何与状态机归纳偏置，因此是“相同连续动作输出接口”而不是“相同内部结构”。

## 动态结果

Alphabet soup：

![Alphabet soup 四模式成功对比](media/alphabet-soup-init0/comparison.gif)

Cream cheese：

![Cream cheese 四模式成功对比](media/cream-cheese-init0/comparison.gif)

每组还保存 `comparison.mp4`、`poster.png`、`final.png` 与 `media.json`。媒体由实验记录的帧和墙钟时间生成，没有再次调用模型。

## 边界与下一步

本轮使用 `init_index=0`、`seed=7` 的两个开发任务，且混合控制器使用了公开资产的外观参考，因此适合证明端到端链路与控制机制，不能证明零样本或跨纹理泛化。Qwen 服务记录中的 revision 为 `unspecified`，后续正式扩展实验必须固定权重 revision，并至少覆盖更多任务、多个初态、无外观参考消融、关闭深度消融和重复运行置信区间。

[机器可读指标](metrics.json) · [协议摘要](protocol.json) · [上一版失败消融](../phase2-action-chunks-2026-10-05/RESULTS.md)
