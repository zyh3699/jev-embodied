"""Plot the compact outcome, wall-time and price summary for fixture tasks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PRICES = {"jev": (0.042, 0.0), "chat": (10.0, 50.0)}
COLORS = {"jev": "#19765a", "chat": "#3c63a8"}
LABELS = {"jev": "Jev", "chat": "GPT-6 Astra"}


def load(path: Path):
    report = json.loads(path.read_text())
    if report.get("format") != "embodied-jev-evaluation-v1":
        raise ValueError(f"Unsupported report: {path}")
    if len(report.get("episodes", [])) != len(report["manifest"]["cases"]):
        raise ValueError(f"Incomplete report rows: {path}")
    return report


def known_cost(report):
    provider = report["configuration"]["policy"]
    input_rate, output_rate = PRICES[provider]
    calls = [call for episode in report["episodes"] for call in episode.get("api_calls", [])]
    known = sum(((call.get("input_tokens") or 0) * input_rate
                 + (call.get("output_tokens") or 0) * output_rate) / 1_000_000
                for call in calls)
    complete = all(call.get("input_tokens") is not None and call.get("output_tokens") is not None
                   for call in calls)
    return known, complete, len(calls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs=2, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Output basename; writes PNG, SVG and JSON")
    args = parser.parse_args()
    reports = [load(path) for path in args.reports]
    if reports[0]["manifest"] != reports[1]["manifest"]:
        raise ValueError("Reports do not share the same manifest")
    if reports[0]["source_sha256"] != reports[1]["source_sha256"]:
        raise ValueError("Reports do not share the same evaluated source")
    by_provider = {report["configuration"]["policy"]: report for report in reports}
    if set(by_provider) != set(PRICES):
        raise ValueError("Expected one Jev and one chat report")
    cases = reports[0]["manifest"]["cases"]
    for case in cases:
        rows = [next(e for e in report["episodes"] if e["id"] == case["id"]) for report in reports]
        if rows[0]["initial_observation_sha256"] != rows[1]["initial_observation_sha256"]:
            raise ValueError(f"Initial state differs for {case['id']}")

    rows = []
    for provider in ("jev", "chat"):
        report = by_provider[provider]
        cost, cost_complete, requests = known_cost(report)
        episodes = report["episodes"]
        rows.append({
            "provider": provider,
            "label": LABELS[provider],
            "successes": sum(e.get("success") is True for e in episodes),
            "planned": len(cases),
            "wall_seconds": sum(e["wall_seconds"] for e in episodes),
            "known_cost_usd": cost,
            "cost_complete": cost_complete,
            "requests": requests,
            "statuses": {status: sum(e["status"] == status for e in episodes)
                         for status in sorted({e["status"] for e in episodes})},
        })

    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    mpl.rcParams.update({"font.family": "sans-serif", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.0))
    fig.subplots_adjust(left=.075, right=.98, top=.67, bottom=.22, wspace=.42)
    names = [row["label"] for row in rows]
    colors = [COLORS[row["provider"]] for row in rows]

    values = [row["successes"] for row in rows]
    axes[0].bar(names, values, color=colors, width=.58)
    axes[0].set(ylim=(0, len(cases) + .65), ylabel="Verified successful episodes")
    axes[0].set_title("Official task success", loc="left")
    for i, row in enumerate(rows):
        suffix = "*" if any(status == "runtime_error" for status in row["statuses"]) else ""
        axes[0].text(i, row["successes"] + .12, f"{row['successes']}/{row['planned']}{suffix}", ha="center")

    values = [row["wall_seconds"] for row in rows]
    axes[1].bar(names, values, color=colors, width=.58)
    axes[1].set(ylabel="Wall time (seconds)")
    axes[1].set_title("End-to-end runtime", loc="left")
    for i, value in enumerate(values):
        axes[1].text(i, value + max(values) * .035, f"{value:.1f}s", ha="center")

    values = [row["known_cost_usd"] for row in rows]
    axes[2].bar(names, values, color=colors, width=.58)
    axes[2].set(ylabel="Estimated USD")
    axes[2].set_title("Known API price subtotal", loc="left")
    for i, (row, value) in enumerate(zip(rows, values)):
        prefix = "" if row["cost_complete"] else "≥"
        axes[2].text(i, value + max(values) * .035, prefix + f"${value:.4f}", ha="center")

    for letter, ax in zip("abc", axes):
        ax.text(-.12, 1.08, letter, transform=ax.transAxes, weight="bold")
        ax.tick_params(axis="x", labelrotation=0)
        ax.grid(axis="y", color="#e7ebef", linewidth=.7)
        ax.set_axisbelow(True)
    speedup = rows[1]["wall_seconds"] / rows[0]["wall_seconds"]
    fig.suptitle("Meta-World fixture manipulation · paired comparison", x=.075, ha="left",
                 y=.97, fontsize=14, weight="bold")
    fig.text(.075, .82, f"Open drawer + open door · 2 seeds each · Jev completed the batch {speedup:.1f}× faster")
    fig.text(.075, .055,
             "Same task seeds, privileged state, action contract and budgets. * GPT includes one retained runtime/format failure.\n"
             "Price uses Jev $0.042/M input and GPT-6 Astra $10/M input + $50/M output; ≥ means requests with missing usage.",
             fontsize=8, color="#536273")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=220)
    fig.savefig(args.output.with_suffix(".svg"))
    plt.close(fig)
    audit = {
        "format": "jev-embodied-fixture-summary-v1",
        "manifest_sha256": reports[0]["manifest_sha256"],
        "source_sha256": reports[0]["source_sha256"],
        "report_sha256": {path.name + f"-{i}": hashlib.sha256(path.read_bytes()).hexdigest()
                          for i, path in enumerate(args.reports)},
        "pricing_usd_per_million": {"jev_input": .042, "jev_output": 0,
                                    "gpt_input": 10, "gpt_output": 50},
        "methods": rows,
        "wall_time_ratio_gpt_over_jev": speedup,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
