# Panda + GPT-6 Astra 冒烟记录（2026-09-24）

本记录只包含当天通过 OpenLux 的 OpenAI 兼容接口实际运行的实验。模型 ID 为 `gpt-6-astra`，观测为仿真真值，无相机，控制方式为高层技能选择。

## 连接检查

| 次数 | 结果 | 耗时 |
|---|---|---:|
| 第一次 | 超时 | 61.253 s |
| 第二次 | 通过 | 15.158 s |

连接能够成功，但延迟不稳定。

## 冒烟结果

| 任务 | 种子 | 成功 | 已完成周期 | API 调用 | API 等待 | 总耗时 | 违规接触 | 估算费用 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| transfer | 0 | 否 | 5 | 10 | 212.816 s | 214.12 s | 0 | $0.2169 |

模型共报告 18,384 个输入 token 和 661 个输出 token。按页面配置的 $10/百万输入 token、$50/百万输出 token 估算：

```text
18,384 × $10 / 1,000,000 + 661 × $50 / 1,000,000 = $0.21689
```

这个估算只包含正式冒烟局，不包含两次连接测试可能产生的费用。

单次调用延迟中位数为 19.655 秒，最大值为 60.003 秒。最终错误为 `ReadTimeout`。

## 执行到哪里

模型已经依次完成：

```text
approach → descend → grasp → lift → carry
```

失败时夹爪仍闭合并持有方块；末端位置为 `[0.42911, 0.17987, 0.21993]`，方块位置为 `[0.43259, 0.17988, 0.21593]`。也就是说，方块已经移动到目标上方，但后续模型请求超时，未执行下降、释放和撤离，因此物理成功条件没有通过。

## 结论

- 适配器能够调用 GPT-6 Astra，前 5 个技能阶段的选择有效。
- 当前 OpenLux 链路延迟高且不稳定；本局 99.4% 的时间用于等待 API。
- 一局成本约为 Jev 完整 9 局技能组成本的 44.7 倍。
- 因单局已经高延迟失败，本轮没有扩展到 9 局，不能把 0/1 当作稳定成功率结论。

## 复现命令

使用仓库内只包含 `transfer / seed 0` 的冒烟清单：

```bash
jev-embodied evaluate \
  --manifest benchmarks/builtin-smoke.json \
  --output runs/revalidation-2026-09-24/gpt6-astra-skills-smoke \
  --policy chat \
  --connection-source saved \
  --control-mode skills \
  --observation-mode privileged \
  --max-steps 30 \
  --max-calls 60 \
  --timeout 300 \
  --threshold 0
```

输出目录必须不存在。API Key 从本机保存连接读取，不写入实验结果。
