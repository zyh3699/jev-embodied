"""Plot the three paired reproduction experiments from their recorded JSON files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


PRICES = {
    "jev": (0.042, 0.0),
    "chat": (10.0, 50.0),
}


def load(path):
    return json.loads(path.read_text())


def token_cost(provider, row):
    input_tokens, output_tokens = row.get("input_tokens"), row.get("output_tokens")
    if input_tokens is None or output_tokens is None:
        return None
    input_rate, output_rate = PRICES[provider]
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000


def panda_record(path, label):
    row = load(path)
    return {"label": label, "successes": int(row["success"]), "n": 1,
            "wall_seconds": row["wall_seconds"], "estimated_usd": token_cost(row["provider"], row),
            "cost_is_lower_bound": False,
            "calls": row["model_calls"], "environment_steps": len(row["history"]),
            "source": str(path), "pair_key": row["scene_hash"]}


def metaworld_record(path, label, provider):
    row = load(path)
    aggregate = row["aggregate"]
    return {"label": label, "successes": aggregate["successes"], "n": aggregate["planned"],
            "wall_seconds": aggregate["wall_seconds"], "estimated_usd": token_cost(provider, aggregate),
            "cost_is_lower_bound": False,
            "calls": aggregate["model_calls"], "environment_steps": aggregate["environment_steps"],
            "source": str(path),
            "pair_key": sorted((e["id"], e["initial_observation_sha256"]) for e in row["episodes"])}


def libero_record(path, label):
    row = load(path)
    metrics = row["metrics"]
    complete_cost = metrics["estimated_usd"]
    return {"label": label, "successes": int(row["success"]), "n": 1,
            "wall_seconds": row["wall_seconds"],
            "estimated_usd": complete_cost if complete_cost is not None else metrics["known_cost_subtotal_usd"],
            "cost_is_lower_bound": complete_cost is None,
            "calls": metrics["requests"], "environment_steps": row["steps"], "source": str(path),
            "pair_key": (row["metadata"]["initial_state_sha256"], row["metadata"]["settled_state_sha256"]),
            "status": row["status"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("panda_jev", "panda_gpt", "metaworld_jev", "metaworld_gpt", "libero_gpt", "libero_hybrid"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    for candidate in (Path("/System/Library/Fonts/PingFang.ttc"),
                      Path("/System/Library/Fonts/STHeiti Medium.ttc")):
        if candidate.exists():
            font_manager.fontManager.addfont(candidate)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=candidate).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    groups = [
        ("Panda · transfer seed 0", [panda_record(args.panda_jev, "Jev"), panda_record(args.panda_gpt, "GPT-6 Astra")]),
        ("Meta-World · 6 episodes", [metaworld_record(args.metaworld_jev, "Jev", "jev"),
                                      metaworld_record(args.metaworld_gpt, "GPT-6 Astra", "chat")]),
        ("LIBERO · drawer init 0", [libero_record(args.libero_gpt, "GPT-6 Astra"),
                                     libero_record(args.libero_hybrid, "GPT-6 + Jev")]),
    ]
    for title, pair in groups:
        if pair[0]["pair_key"] != pair[1]["pair_key"]:
            raise ValueError(f"Unpaired initial state: {title}")
    args.output.mkdir(parents=True, exist_ok=False)

    method_colors = {"Jev": "#23765d", "GPT-6 Astra": "#4169a8", "GPT-6 + Jev": "#7b5ca7"}
    fig, axes = plt.subplots(3, 4, figsize=(15.5, 10), constrained_layout=True)
    for row_index, (title, pair) in enumerate(groups):
        labels = [p["label"] for p in pair]
        success = [100 * p["successes"] / p["n"] for p in pair]
        wall = [p["wall_seconds"] for p in pair]
        cost = [p["estimated_usd"] for p in pair]
        steps = [p["environment_steps"] for p in pair]
        values = (success, wall, cost, steps)
        for column, (axis, data) in enumerate(zip(axes[row_index], values)):
            bars = axis.bar(np.arange(2), data, color=[method_colors[label] for label in labels], width=.62)
            axis.set_xticks(np.arange(2), labels, rotation=8, ha="right")
            axis.grid(axis="y", alpha=.22)
            axis.set_axisbelow(True)
            if column == 0:
                axis.set_ylim(0, 110)
                axis.set_ylabel(title + "\n成功率 (%)", fontweight="bold")
                annotations = [f"{p['successes']}/{p['n']}" for p in pair]
            elif column == 1:
                axis.set_ylabel("总墙钟时间 (s)")
                annotations = [f"{v:.1f}s" for v in data]
            elif column == 2:
                if any(v is None or v <= 0 for v in data):
                    raise ValueError("Cost is incomplete or non-positive")
                if max(data) / min(data) >= 10:
                    axis.set_yscale("log")
                    axis.set_ylabel("估算 API 费用 (USD，对数轴)")
                else:
                    axis.set_ylim(0, max(data) * 1.12)
                    axis.set_ylabel("估算 API 费用 (USD)")
                annotations = [("≥" if p["cost_is_lower_bound"] else "") + f"${v:.4g}"
                               for p, v in zip(pair, data)]
            else:
                axis.set_ylabel("已执行环境步 / 决策周期")
                annotations = [str(v) for v in data]
            for bar, text in zip(bars, annotations):
                axis.annotate(text, (bar.get_x() + bar.get_width()/2, bar.get_height()),
                              xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    axes[0, 0].set_title("官方成功判定", fontweight="bold")
    axes[0, 1].set_title("真实端到端运行时间", fontweight="bold")
    axes[0, 2].set_title("按公开标准价估算", fontweight="bold")
    axes[0, 3].set_title("预算内完成的物理推进", fontweight="bold")
    fig.suptitle("jev-embodied · 三组配对复现实验", fontsize=17, fontweight="bold")
    fig.text(.5, -.01, "同一行共享初始状态；Meta-World 为开发子集，LIBERO 为单个开发样本。费用不含中转加价或缓存折扣。",
             ha="center", fontsize=9)
    for suffix in ("png", "svg"):
        fig.savefig(args.output / f"overview.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    output = {"pricing_usd_per_million": {"jev_input": .042, "jev_output": 0,
              "gpt6_astra_input": 10, "gpt6_astra_output": 50},
              "groups": [{"name": title, "methods": pair} for title, pair in groups]}
    (args.output / "metrics.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "paired_groups": len(groups)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
