"""Phase-two paired LIBERO evaluation: pi0.5 versus local VLM plus Jev.

The two policies are intentionally independent. pi0.5 emits continuous action
chunks directly and never calls Jev. The comparison policy asks a local VLM for
bounded visual evidence, then asks Jev to choose exactly one action from a frozen
high-cardinality action set. The simulator alone supplies task success.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import time

from .evaluation import Worker, load_manifest, saved_connection
from .libero_policy import ModelClient, RequestBudget, usage_summary
from .policies import environment_connection, validate_answer


PROTOCOL = "libero-pi05-vs-local-vlm-jev-v1"
MODES = ("pi05", "vlm-jev")
SOURCE_FILES = ("phase2_compare.py", "benchmark_worker.py", "evaluation.py",
                "libero_policy.py", "evaluation_meter.py", "policies.py")
PHASES = {"approach", "align", "contact", "manipulate", "release", "recover", "uncertain"}
DIRECTIONS = {"negative", "hold", "positive", "unknown"}
GRIPPER = {"open", "hold", "close", "unknown"}

VLM_SYSTEM = """You are the visual perception module in a robot-control experiment.
Inspect the current external and wrist RGB images, the language task, proprioceptive
state, and recent executed actions. Report visible evidence and directional advice;
do not claim task success, contact force, hidden object state, or exact metric depth.
The controller uses world-frame x/y/z translation and axis-angle rx/ry/rz. Use
'unknown' whenever the images do not support a direction. Return one JSON object
with exactly:
{"phase":"approach|align|contact|manipulate|release|recover|uncertain",
 "summary":"brief current scene description",
 "visible_evidence":"brief image-grounded evidence",
 "translation":{"x":"negative|hold|positive|unknown","y":"...","z":"..."},
 "rotation":{"rx":"negative|hold|positive|unknown","ry":"...","rz":"..."},
 "gripper":"open|hold|close|unknown","risk":"brief risk or uncertainty"}
Do not output an action vector or candidate ID. A separate typed selector chooses
one action from a fixed candidate set."""


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def decode_png(packet):
    if not isinstance(packet, dict) or packet.get("encoding") != "png":
        raise ValueError("Expected an encoded PNG camera packet")
    data = base64.b64decode(packet.get("data", ""), validate=True)
    if not data.startswith(b"\x89PNG\r\n\x1a\n") or hashlib.sha256(data).hexdigest() != packet.get("sha256"):
        raise ValueError("Camera packet failed PNG or SHA-256 validation")
    return data


def rotate_png_180(data):
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("The phase2 extra (Pillow) is required") from exc
    source = Image.open(io.BytesIO(data)).convert("RGB")
    output = io.BytesIO()
    source.rotate(180).save(output, format="PNG")
    return output.getvalue()


def initial_fingerprint(packet):
    policy = packet["policy_input"]
    public = {"prompt": policy["prompt"], "state": policy["state"],
              "images": {name: image["sha256"] for name, image in sorted(policy["images"].items())}}
    return hashlib.sha256(json.dumps(public, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def action_candidates(scale, previous_gripper):
    """Return the frozen 21-way normalized LIBERO action set."""
    if not 0 < scale <= .5 or not -1 <= previous_gripper <= 1:
        raise ValueError("Invalid candidate scale or previous gripper command")
    # Half effort is intentionally distinct from the saturated open/close
    # candidates while still preserving the current gripper direction.
    pause_gripper = previous_gripper / 2 if previous_gripper else .25
    result = {"hold": {"description": "Hold pose and reduce current gripper effort by half.",
                       "action": [0., 0., 0., 0., 0., 0., float(pause_gripper)]}}
    for magnitude_name, magnitude in (("small", scale / 2), ("full", scale)):
        for index, axis in enumerate("xyz"):
            for sign_name, sign in (("negative", -1), ("positive", 1)):
                action = [0.] * 6 + [float(previous_gripper)]
                action[index] = sign * magnitude
                key = f"translate_{axis}_{sign_name}_{magnitude_name}"
                result[key] = {"description": f"Translate world {axis.upper()} {sign_name}, {magnitude_name} step; retain gripper.",
                               "action": action}
    for index, axis in enumerate(("rx", "ry", "rz"), start=3):
        for sign_name, sign in (("negative", -1), ("positive", 1)):
            action = [0.] * 6 + [float(previous_gripper)]
            action[index] = sign * scale / 2
            key = f"rotate_{axis}_{sign_name}"
            result[key] = {"description": f"Rotate world {axis.upper()} {sign_name}, small step; retain gripper.",
                           "action": action}
    for name, value in (("open", -1.), ("close", 1.)):
        result["gripper_" + name] = {"description": f"Hold pose and {name} the gripper.",
                                     "action": [0.] * 6 + [value]}
    if len(result) != 21:
        raise AssertionError("The frozen candidate set must contain 21 actions")
    return result


def validate_vlm_analysis(answer):
    fields = {"phase", "summary", "visible_evidence", "translation", "rotation", "gripper", "risk"}
    if not isinstance(answer, dict) or set(answer) != fields or answer["phase"] not in PHASES:
        raise ValueError("Local VLM returned an invalid scene-analysis schema")
    for field in ("summary", "visible_evidence", "risk"):
        if not isinstance(answer[field], str) or not 1 <= len(answer[field]) <= 600:
            raise ValueError(f"Local VLM returned invalid {field}")
    if (not isinstance(answer["translation"], dict) or set(answer["translation"]) != set("xyz")
            or any(value not in DIRECTIONS for value in answer["translation"].values())):
        raise ValueError("Local VLM returned invalid translation evidence")
    if (not isinstance(answer["rotation"], dict) or set(answer["rotation"]) != {"rx", "ry", "rz"}
            or any(value not in DIRECTIONS for value in answer["rotation"].values())):
        raise ValueError("Local VLM returned invalid rotation evidence")
    if answer["gripper"] not in GRIPPER:
        raise ValueError("Local VLM returned invalid gripper evidence")
    return answer


def validate_action_chunk(value, minimum=1):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) < minimum:
        raise ValueError("pi0.5 returned an empty or too-short action chunk")
    result = []
    for action in value:
        if hasattr(action, "tolist"):
            action = action.tolist()
        if not isinstance(action, (list, tuple)) or len(action) != 7:
            raise ValueError("pi0.5 actions must be finite normalized 7D vectors in [-1, 1]")
        try:
            converted = [float(number) for number in action]
        except (TypeError, ValueError, OverflowError):
            raise ValueError("pi0.5 actions must be finite normalized 7D vectors in [-1, 1]") from None
        if any(isinstance(number, bool) or not math.isfinite(value) or abs(value) > 1
               for number, value in zip(action, converted)):
            raise ValueError("pi0.5 actions must be finite normalized 7D vectors in [-1, 1]")
        result.append(converted)
    return result


class Pi05Client:
    def __init__(self, host, port):
        try:
            from openpi_client import image_tools, websocket_client_policy
        except ImportError as exc:
            raise RuntimeError("Install openpi-client from the official openpi repository before running pi05") from exc
        self.image_tools = image_tools
        self.policy = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)

    def _image(self, packet):
        try:
            import numpy as np
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("The phase2 extra (NumPy and Pillow) is required") from exc
        image = np.asarray(Image.open(io.BytesIO(decode_png(packet))).convert("RGB"))
        image = image[::-1, ::-1]  # Official openpi LIBERO evaluation transform.
        image = self.image_tools.resize_with_pad(image, 224, 224)
        return self.image_tools.convert_to_uint8(image)

    def infer(self, policy_input):
        external = self._image(policy_input["images"]["external"])
        wrist = self._image(policy_input["images"]["wrist"])
        observation = {"observation/image": external,
                       "observation/wrist_image": wrist,
                       "observation/state": policy_input["state"], "prompt": policy_input["prompt"]}
        started = time.perf_counter()
        response = self.policy.infer(observation)
        latency = (time.perf_counter() - started) * 1000
        if not isinstance(response, dict) or "actions" not in response:
            raise ValueError("pi0.5 response must contain an actions field")
        audit = {"external": {"sha256": hashlib.sha256(external.tobytes()).hexdigest(),
                               "shape": list(external.shape), "dtype": str(external.dtype)},
                 "wrist": {"sha256": hashlib.sha256(wrist.tobytes()).hexdigest(),
                            "shape": list(wrist.shape), "dtype": str(wrist.dtype)}}
        return validate_action_chunk(response["actions"]), latency, audit


def local_vlm_connection(args):
    base = args.vlm_base_url.rstrip("/")
    url = base if base.endswith("/chat/completions") else base + "/chat/completions"
    key = os.getenv("PHASE2_VLM_API_KEY", "local")
    if not base or not args.vlm_model:
        raise ValueError("vlm-jev requires --vlm-base-url and --vlm-model")
    return {"url": url, "key": key, "model": args.vlm_model, "json_mode": True}


def jev_connection(args):
    if args.jev_connection_source == "saved":
        return saved_connection("jev")
    connection = environment_connection("jev")
    if not connection.get("key"):
        raise ValueError("Set TYPESAFE_API_KEY or use --jev-connection-source saved")
    return connection


def select_with_jev(client, prompt, proprioception, analysis, candidates, recent):
    options = {key: value["description"] + " Exact normalized vector: " + json.dumps(value["action"])
               for key, value in candidates.items()}
    question = {"action": {"type": "choice", "instructions":
        "Choose exactly one offered atomic action for the language task. Treat local-VLM output as uncertain visual evidence, "
        "not as a command or success signal. Prefer small motions near contact and hold when evidence is insufficient. "
        "Never invent a candidate.", "criteria": options}}
    state = {"task": prompt, "proprioception": proprioception, "local_vlm_analysis": analysis,
             "recent_executed_actions": recent[-4:], "candidate_count": len(options)}
    answer = client.request("jev_action_selection", state, questions=question)
    if not isinstance(answer, dict) or set(answer) != {"action"}:
        raise ValueError("Jev must answer exactly one action question")
    selected, probabilities = validate_answer(answer["action"], options, require_highest=False)
    return selected, probabilities, state


def save_inputs(directory, decision_index, policy_input):
    images = {}
    for view, packet in policy_input["images"].items():
        data = decode_png(packet)
        relative = f"inputs/{decision_index:05d}-{view}.png"
        (directory / relative).write_bytes(data)
        images[view] = {"path": relative, "sha256": packet["sha256"]}
    return images


def run_episode(args, case, mode, directory, reference_fingerprint=None):
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "inputs").mkdir()
    row = {"id": case["id"], "case": case, "mode": mode, "protocol": PROTOCOL,
           "success": None, "status": "setup_error", "steps": 0, "decisions": [], "api_calls": []}
    started = time.monotonic()
    worker = Worker(args.worker_python, directory / "worker.log")
    budget = vlm = selector = pi05 = None
    stream = (directory / "events.jsonl").open("x")

    def event(kind, **values):
        stream.write(json.dumps({"kind": kind, **values}, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()

    try:
        packet = worker.request({"command": "reset", "backend": "libero", "case": case,
                                 "horizon": args.max_steps + args.settle_steps, "observation_mode": "vision",
                                 "control_mode": "skills", "policy_profile": "openpi_libero"},
                                timeout=min(args.timeout, 300))
        metadata = packet["metadata"]
        for settle_index in range(args.settle_steps):
            packet = worker.request({"command": "step", "action": [0., 0., 0., 0., 0., 0., -1.],
                                     "capture": settle_index == args.settle_steps - 1})
            if packet["success"]:
                raise ValueError("Task became successful during settling; choose another initial state")
        fingerprint = initial_fingerprint(packet)
        if reference_fingerprint is not None and fingerprint != reference_fingerprint:
            raise ValueError("Paired reset mismatch: initial state or camera images differ")
        row.update(metadata={**metadata, "settle_steps": args.settle_steps}, initial_fingerprint=fingerprint,
                   success=False, status="step_budget", setup_seconds=time.monotonic() - started)
        rollout_started = time.monotonic()
        if mode == "pi05":
            pi05 = Pi05Client(args.pi05_host, args.pi05_port)
        else:
            budget = RequestBudget(args.max_calls, args.timeout, args.max_usd)
            vlm = ModelClient("chat", local_vlm_connection(args), budget, request_retries=args.request_retries)
            selector = ModelClient("jev", jev_connection(args), budget, request_retries=args.request_retries)
        previous_gripper, recent = -1., []
        event("reset", metadata=row["metadata"], initial_fingerprint=fingerprint)
        while row["steps"] < args.max_steps:
            if time.monotonic() - rollout_started >= args.timeout:
                row["status"] = "time_budget"
                break
            policy_input = packet["policy_input"]
            index = len(row["decisions"])
            inputs = save_inputs(directory, index, policy_input)
            decision = {"index": index, "step": row["steps"], "input_images": inputs,
                        "start_seconds": time.monotonic() - rollout_started}
            if mode == "pi05":
                chunk, latency, model_inputs = pi05.infer(policy_input)
                actions = chunk[:args.pi05_replan_steps]
                decision.update(source="pi0.5 direct action chunk", chunk_length=len(chunk),
                                executed_chunk_length=len(actions), inference_latency_ms=latency,
                                model_input_images=model_inputs,
                                image_transform="rotate 180 degrees, resize_with_pad 224x224, uint8")
            else:
                budget.check()
                visual_state = {"task": policy_input["prompt"], "proprioception": policy_input["state"],
                                "recent_executed_actions": recent[-4:],
                                "note": "No task success flag, object pose, depth, or force sensor is provided."}
                images = {view: rotate_png_180(decode_png(image)) for view, image in policy_input["images"].items()}
                model_inputs = {}
                for view, image in images.items():
                    relative = f"inputs/{index:05d}-{view}-vlm.png"
                    (directory / relative).write_bytes(image)
                    model_inputs[view] = {"path": relative, "sha256": hashlib.sha256(image).hexdigest()}
                analysis = validate_vlm_analysis(vlm.request("local_vlm_scene_analysis", visual_state,
                                                              system=VLM_SYSTEM, images=images))
                candidates = action_candidates(args.candidate_scale, previous_gripper)
                selected, probabilities, selector_state = select_with_jev(
                    selector, policy_input["prompt"], policy_input["state"], analysis, candidates, recent)
                actions = [candidates[selected]["action"]] * args.action_repeat
                decision.update(source="local VLM evidence plus Jev 21-way selection", vlm_analysis=analysis,
                                selection=selected, probabilities=probabilities,
                                selected_action=candidates[selected]["action"], selector_state=selector_state,
                                candidate_count=len(candidates), model_input_images=model_inputs,
                                image_transform="rotate 180 degrees; local VLM server owns resize/tokenization")
            decision["inference_end_seconds"] = time.monotonic() - rollout_started
            row["decisions"].append(decision)
            event("decision", **decision)
            for action in actions:
                if row["steps"] >= args.max_steps or time.monotonic() - rollout_started >= args.timeout:
                    break
                packet = worker.request({"command": "step", "action": action, "capture": True})
                row["steps"] = packet["policy_input"]["step"] - args.settle_steps
                previous_gripper = float(action[-1])
                executed = {"step": row["steps"], "action": action, "success": packet["success"]}
                recent.append(executed)
                event("step", **executed)
                if packet["success"] or packet["truncated"]:
                    row["success"] = bool(packet["success"])
                    row["status"] = "success" if packet["success"] else "step_budget"
                    break
            decision.update(end_step=row["steps"], end_seconds=time.monotonic() - rollout_started)
            row["api_calls"] = budget.calls if budget else []
            write_json(directory / "episode.json", row)
            print(json.dumps({"case": case["id"], "mode": mode, "step": row["steps"],
                              "success": row["success"], "decision": index}, ensure_ascii=False), flush=True)
            if row["success"] or packet["truncated"]:
                break
        row["rollout_seconds"] = time.monotonic() - rollout_started
    except KeyboardInterrupt:
        row["status"] = "interrupted"
    except Exception as exc:
        budget_status = str(exc) if str(exc) in {"request_budget", "time_budget", "cost_budget"} else None
        row["status"] = budget_status or ("setup_error" if row["success"] is None else "runtime_error")
        row["error_type"] = type(exc).__name__
        row["error"] = str(exc)[:500]
        print(json.dumps({"case": case["id"], "mode": mode, "status": row["status"],
                          "error_type": row["error_type"]}), flush=True)
    finally:
        for client in (vlm, selector):
            if client is not None:
                client.close()
        worker.close()
        stream.close()
        row["wall_seconds"] = time.monotonic() - started
        row["api_calls"] = budget.calls if budget else []
        row["metrics"] = usage_summary(row["api_calls"])
        write_json(directory / "episode.json", row)
    return row


def aggregate(rows, modes):
    result = {}
    for mode in modes:
        group = [row for row in rows if row["mode"] == mode]
        scored = [row for row in group if row.get("success") is not None]
        result[mode] = {"planned": len(group), "scored": len(scored),
                        "successes": sum(bool(row["success"]) for row in scored),
                        "success_rate": (sum(bool(row["success"]) for row in scored) / len(scored)) if scored else None,
                        "mean_environment_steps": statistics.mean(row["steps"] for row in scored) if scored else None,
                        "mean_wall_seconds": statistics.mean(row["wall_seconds"] for row in group) if group else None,
                        "complete": len(scored) == len(group) and all(row["status"] not in {"setup_error", "runtime_error"} for row in group)}
    return result


def run(args):
    manifest = load_manifest(args.manifest)
    if manifest["backend"] != "libero":
        raise ValueError("phase2-compare requires a LIBERO manifest")
    modes = list(args.modes)
    if not modes or len(modes) != len(set(modes)) or any(mode not in MODES for mode in modes):
        raise ValueError("Modes must be unique values from pi05 and vlm-jev")
    if (not 1 <= args.max_steps <= 1000 or not 1 <= args.max_calls <= 1000
            or not 1 <= args.action_repeat <= 20 or not 1 <= args.pi05_replan_steps <= 50
            or not 0 <= args.settle_steps <= 50 or not 0 < args.candidate_scale <= .5 or not 1 <= args.timeout <= 14400
            or not 0 < args.max_usd <= 100 or not 0 <= args.request_retries <= 2):
        raise ValueError("Invalid phase-two experiment bounds")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "reproduction").mkdir()
    hashes = {}
    for name in SOURCE_FILES:
        source = Path(__file__).with_name(name)
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        shutil.copy2(source, output / "reproduction" / name)
    protocol = {"version": PROTOCOL, "created_at": datetime.now(timezone.utc).isoformat(),
                "manifest": manifest, "modes": modes, "source_sha256": hashes,
                "policy_contract": {"pi05": "direct 7D normalized action chunks; Jev is never called",
                    "vlm-jev": "local VLM visual evidence; Jev selects one of 21 frozen atomic actions"},
                "budget": {key: getattr(args, key) for key in ("max_steps", "max_calls", "timeout", "max_usd",
                    "action_repeat", "candidate_scale", "pi05_replan_steps", "settle_steps")},
                "endpoints": {"pi05": {"host": args.pi05_host, "port": args.pi05_port},
                              "vlm": {"base_url": args.vlm_base_url, "model": args.vlm_model},
                              "jev_connection_source": args.jev_connection_source}}
    write_json(output / "protocol.json", protocol)
    rows = []
    for case in manifest["cases"]:
        reference = None
        for mode in modes:
            row = run_episode(args, case, mode, output / f"{case['id']}--{mode}", reference)
            reference = reference or row.get("initial_fingerprint")
            rows.append(row)
            report = {"protocol": PROTOCOL, "rows": rows, "aggregate": aggregate(rows, modes)}
            write_json(output / "report.json", report)
            if row["status"] in {"setup_error", "runtime_error"} and not args.continue_on_error:
                return report
    report = {"protocol": PROTOCOL, "rows": rows, "aggregate": aggregate(rows, modes)}
    write_json(output / "report.json", report)
    return report


def add_arguments(parser):
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True, help="New output directory; never overwritten")
    parser.add_argument("--worker-python", required=True, help="Python executable in the LIBERO environment")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--pi05-host", default="127.0.0.1")
    parser.add_argument("--pi05-port", type=int, default=8000)
    parser.add_argument("--pi05-replan-steps", type=int, default=5,
                        help="Execute this many actions from each pi0.5 chunk before replanning")
    parser.add_argument("--vlm-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--vlm-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--jev-connection-source", choices=["environment", "saved"], default="environment")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--settle-steps", type=int, default=10,
                        help="Official pre-policy LIBERO dummy steps; excluded from max-steps")
    parser.add_argument("--max-calls", type=int, default=400)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--max-usd", type=float, default=10)
    parser.add_argument("--action-repeat", type=int, default=5)
    parser.add_argument("--candidate-scale", type=float, default=.5)
    parser.add_argument("--request-retries", type=int, default=1)
    parser.add_argument("--continue-on-error", action="store_true")
