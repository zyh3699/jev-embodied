"""Create a compact audited summary from a phase-two v4 result directory."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(source, output):
    report_path = source / "report.json"
    protocol_path = source / "protocol.json"
    report = json.loads(report_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    rows, episodes = report["rows"], []
    for row in rows:
        path = source / f"{row['id']}--{row['mode']}" / "episode.json"
        episode = json.loads(path.read_text())
        calls = episode.get("api_calls", [])
        decisions = episode.get("decisions", [])
        families = Counter()
        triggers = Counter()
        for decision in decisions:
            selected = decision.get("selection")
            candidate = decision.get("candidate_chunks", {}).get(selected, {})
            if candidate.get("family"):
                families[candidate["family"]] += 1
            triggers.update(decision.get("vlm_trigger_reasons", []))
        episodes.append({"case": row["id"], "task": episode.get("metadata", {}).get("task_name")
                         or episode["case"].get("suite") + "/" + str(episode["case"].get("task_id")),
            "mode": row["mode"], "success": row["success"], "status": row["status"],
            "steps": row["steps"], "wall_seconds": row.get("wall_seconds"),
            "decisions": len(decisions), "frames": len(episode.get("frames", [])),
            "vlm_requests": sum(call.get("provider") == "chat" for call in calls),
            "jev_requests": sum(call.get("provider") == "jev" for call in calls),
            "vlm_plans": sum(bool(decision.get("vlm_called")) for decision in decisions),
            "chunk_families": dict(sorted(families.items())), "trigger_reasons": dict(sorted(triggers.items())),
            "initial_fingerprint": episode.get("initial_fingerprint"), "episode_sha256": digest(path)})

    modes = {}
    for mode in protocol["modes"]:
        group = [row for row in episodes if row["mode"] == mode]
        modes[mode] = {"episodes": len(group), "successes": sum(row["success"] for row in group),
            "success_rate": sum(row["success"] for row in group) / len(group),
            "mean_steps": statistics.mean(row["steps"] for row in group),
            "mean_wall_seconds": statistics.mean(row["wall_seconds"] for row in group),
            "vlm_requests": sum(row["vlm_requests"] for row in group),
            "jev_requests": sum(row["jev_requests"] for row in group),
            "complete": all(row["status"] not in {"setup_error", "runtime_error"} for row in group)}
    pair_fingerprints = {case: sorted({row["initial_fingerprint"] for row in episodes if row["case"] == case})
                         for case in sorted({row["case"] for row in episodes})}
    value = {"protocol": report["protocol"], "source": str(source),
             "protocol_sha256": digest(protocol_path), "report_sha256": digest(report_path),
             "modes": modes, "episodes": episodes, "pair_fingerprints": pair_fingerprints,
             "all_pairs_match": all(len(values) == 1 for values in pair_fingerprints.values())}
    output.mkdir(parents=True, exist_ok=False)
    (output / "metrics.json").write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    (output / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "episodes": len(episodes),
                      "all_pairs_match": value["all_pairs_match"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.source.resolve(), args.output.resolve())
