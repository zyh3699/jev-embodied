"""Run the frozen Meta-World hierarchical Jev/GPT-6 comparison.

The two providers use the same privileged observation, subgoal contract,
motor questions, simulator seeds and per-episode budgets. Connections are
loaded from the app's saved store; credentials never enter the command line,
manifest, or exported report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "benchmarks/metaworld-hierarchical-compare.json"
SOURCE_FILES = (
    "evaluation.py",
    "evaluation_meter.py",
    "benchmark_worker.py",
    "benchmark_hierarchy.py",
    "policies.py",
    "hierarchical.py",
    "physics.py",
)


def source_hashes():
    return {
        name: hashlib.sha256((REPO / "src/embodied_jev" / name).read_bytes()).hexdigest()
        for name in SOURCE_FILES
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worker-python", type=Path, default=REPO / ".venv-metaworld/bin/python")
    parser.add_argument("--plot-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--validation-retries", type=int, default=1)
    parser.add_argument("--request-retries", type=int, default=1)
    args = parser.parse_args()
    if not args.worker_python.is_file() or not args.plot_python.is_file():
        parser.error("Prepared Meta-World or plotting environment is unavailable")
    output = args.output.resolve()
    if output.exists():
        parser.error("Output already exists; previous trials must not be overwritten")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    protocol = {
        "name": "metaworld-hierarchical-compare",
        "status": "prepared_not_run",
        "providers": ["jev", "chat"],
        "manifest": manifest,
        "manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
        "source_sha256": source_hashes(),
        "configuration": {
            "max_steps": 200,
            "max_calls": 80,
            "timeout": 600.0,
            "threshold": 0.0,
            "action_scale": 0.5,
            "action_repeat": 5,
            "observation_mode": "privileged",
            "control_mode": "hierarchical",
            "connection_source": "saved",
            "validation_retries": args.validation_retries,
            "request_retries": args.request_retries,
            "continue_on_error": True,
        },
        "hierarchy_version": "metaworld-subgoal-v3",
        "request_contract": "one subgoal request followed by one XYZ/gripper request per decision round",
    }
    sys.path.insert(0, str(REPO / "src"))
    from embodied_jev.evaluation import run, saved_connection

    connections = {provider: saved_connection(provider) for provider in protocol["providers"]}
    expected = {"jev": "jev-latest", "chat": "gpt-6-astra"}
    for provider, model in expected.items():
        if connections[provider].get("model") != model:
            parser.error(f"Saved {provider} model differs from frozen protocol")

    output.mkdir(parents=True, exist_ok=False)
    (output / "reproduction").mkdir()
    for name in SOURCE_FILES:
        shutil.copy2(REPO / "src/embodied_jev" / name, output / "reproduction" / name)
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    reports = []
    for provider in protocol["providers"]:
        namespace = SimpleNamespace(
            manifest=str(MANIFEST),
            output=str(output / provider),
            worker_python=str(args.worker_python),
            policy=provider,
            **protocol["configuration"],
        )
        report = run(namespace, connection=connections[provider])
        reports.append(output / provider / "summary.json")
        if not report["aggregate"]["complete"]:
            print(f"{provider}: incomplete batch preserved; no automatic retry.")
            return 2

    environment = dict(
        os.environ,
        PYTHONPATH=str(REPO / "src"),
        MPLCONFIGDIR=str(output / "plot-cache"),
    )
    return subprocess.call(
        [
            str(args.plot_python),
            "-m",
            "embodied_jev.evaluation_charts",
            "--reports",
            *map(str, reports),
            "--output",
            str(output / "figures"),
        ],
        env=environment,
    )


if __name__ == "__main__":
    raise SystemExit(main())
