"""Phase-two paired LIBERO evaluation with three independent policies.

pi0.5 emits continuous action chunks directly and never calls Jev. The dense
hybrid asks a local VLM to plan before every Jev selection. The triggered hybrid
reuses a bounded VLM plan while Jev remains confident, and replans only on an
explicit confidence, horizon, or stagnation trigger. The simulator alone supplies
task success.
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


PROTOCOL = "libero-pi05-vs-local-vlm-jev-v2"
MODES = ("pi05", "vlm-jev-triggered", "vlm-jev-dense")
SOURCE_FILES = ("phase2_compare.py", "benchmark_worker.py", "evaluation.py",
                "libero_policy.py", "evaluation_meter.py", "policies.py")
PHASES = {"approach", "align", "contact", "manipulate", "release", "recover", "uncertain"}
DIRECTIONS = {"negative", "hold", "positive", "unknown"}
GRIPPER = {"open", "hold", "close", "unknown"}

VLM_SYSTEM = """You are the low-frequency visual planner in a robot-control experiment.
Inspect the current external and wrist RGB images, the language task, proprioceptive
state, recent executed actions, and the offered atomic-action catalogue. Report a
short-lived visual plan and choose 3-8 candidate action IDs for a separate selector.
Do not claim task success, contact force, hidden object state, or exact metric depth.
The controller uses world-frame x/y/z translation and axis-angle rx/ry/rz. Use
'unknown' whenever the images do not support a direction. Return one JSON object
with exactly:
{"phase":"approach|align|contact|manipulate|release|recover|uncertain",
 "summary":"brief current scene description",
 "visible_evidence":"brief image-grounded evidence",
 "translation":{"x":"negative|hold|positive|unknown","y":"...","z":"..."},
 "rotation":{"rx":"negative|hold|positive|unknown","ry":"...","rz":"..."},
 "gripper":"open|hold|close|unknown","risk":"brief risk or uncertainty",
 "candidate_actions":["3-8 exact IDs copied from available_actions"],
 "plan_horizon_decisions":1,"replan_condition":"one observable reason to replan"}
Never output an action vector or invent an ID. Include hold when visual evidence is
weak. A separate typed selector chooses exactly one of your candidate IDs."""


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


def validate_vlm_analysis(answer, candidates):
    fields = {"phase", "summary", "visible_evidence", "translation", "rotation", "gripper", "risk",
              "candidate_actions", "plan_horizon_decisions", "replan_condition"}
    if not isinstance(answer, dict) or set(answer) != fields or answer["phase"] not in PHASES:
        raise ValueError("Local VLM returned an invalid planning schema")
    for field in ("summary", "visible_evidence", "risk", "replan_condition"):
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
    offered = answer["candidate_actions"]
    if (not isinstance(offered, list) or not 3 <= len(offered) <= 8
            or len(offered) != len(set(offered)) or any(key not in candidates for key in offered)):
        raise ValueError("Local VLM must offer 3-8 unique known candidate IDs")
    if type(answer["plan_horizon_decisions"]) is not int or not 1 <= answer["plan_horizon_decisions"] <= 8:
        raise ValueError("Local VLM returned an invalid plan horizon")
    return answer


def confidence_metrics(probabilities, selected):
    if not probabilities or selected not in probabilities:
        raise ValueError("Jev probabilities are required for triggered planning")
    ordered = sorted((float(value) for value in probabilities.values()), reverse=True)
    entropy = -sum(value * math.log(value) for value in ordered if value > 0)
    normalized_entropy = entropy / math.log(len(ordered)) if len(ordered) > 1 else 0.
    return {"selected_probability": float(probabilities[selected]),
            "max_probability": ordered[0],
            "top_two_margin": ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0],
            "normalized_entropy": normalized_entropy}


def confidence_triggers(metrics, args):
    reasons = []
    if metrics["selected_probability"] < args.jev_confidence_threshold:
        reasons.append("low_selected_probability")
    if metrics["max_probability"] < args.jev_confidence_threshold:
        reasons.append("low_max_probability")
    if metrics["top_two_margin"] < args.jev_margin_threshold:
        reasons.append("low_top_two_margin")
    if metrics["normalized_entropy"] > args.jev_entropy_threshold:
        reasons.append("high_normalized_entropy")
    return reasons


def stalled(recent, actions, threshold):
    if actions <= 0 or len(recent) < actions:
        return False
    window = recent[-actions:]
    return all(item.get("state_delta_l2", math.inf) < threshold for item in window)


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
            raise ValueError("pi0.5 actions must be finite 7D vectors")
        try:
            converted = [float(number) for number in action]
        except (TypeError, ValueError, OverflowError):
            raise ValueError("pi0.5 actions must be finite 7D vectors") from None
        if any(isinstance(number, bool) or not math.isfinite(value)
               for number, value in zip(action, converted)):
            raise ValueError("pi0.5 actions must be finite 7D vectors")
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
        import numpy as np
        external = self._image(policy_input["images"]["external"])
        wrist = self._image(policy_input["images"]["wrist"])
        observation = {"observation/image": external,
                       "observation/wrist_image": wrist,
                       "observation/state": np.asarray(policy_input["state"], dtype=np.float32),
                       "prompt": policy_input["prompt"]}
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
        raise ValueError("VLM + Jev modes require --vlm-base-url and --vlm-model")
    return {"url": url, "key": key, "model": args.vlm_model, "json_mode": True,
            "request_overrides": {"temperature": .2, "top_p": .8, "max_tokens": 1024,
                                  "chat_template_kwargs": {"enable_thinking": False}}}


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
    state = {"task": prompt, "proprioception": proprioception, "cached_vlm_plan": analysis,
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


def save_frame(directory, frame_index, policy_input, wall_seconds, *, step=None, action=None, decision_index=None):
    images = {}
    for view, packet in policy_input["images"].items():
        relative = f"frames/{frame_index:05d}-{view}.png"
        data = decode_png(packet)
        (directory / relative).write_bytes(data)
        images[view] = {"path": relative, "sha256": packet["sha256"]}
    return {"index": frame_index, "step": policy_input["step"] if step is None else step,
            "wall_seconds": wall_seconds,
            "state": policy_input["state"], "images": images, "action": action,
            "decision_index": decision_index}


def plan_with_vlm(vlm, directory, decision_index, plan_revision, policy_input, recent, candidates):
    visual_state = {"task": policy_input["prompt"], "proprioception": policy_input["state"],
                    "recent_executed_actions": recent[-4:],
                    "available_actions": {key: value["description"] for key, value in candidates.items()},
                    "note": "No task success flag, object pose, depth, or force sensor is provided."}
    images = {view: rotate_png_180(decode_png(image)) for view, image in policy_input["images"].items()}
    model_inputs = {}
    for view, image in images.items():
        relative = f"inputs/{decision_index:05d}-plan{plan_revision:03d}-{view}-vlm.png"
        (directory / relative).write_bytes(image)
        model_inputs[view] = {"path": relative, "sha256": hashlib.sha256(image).hexdigest()}
    plan = validate_vlm_analysis(vlm.request("local_vlm_plan", visual_state,
                                             system=VLM_SYSTEM, images=images), candidates)
    return plan, model_inputs


def run_episode(args, case, mode, directory, reference_fingerprint=None):
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "inputs").mkdir()
    (directory / "frames").mkdir()
    row = {"id": case["id"], "case": case, "mode": mode, "protocol": PROTOCOL,
           "success": None, "status": "setup_error", "steps": 0, "decisions": [], "frames": [], "api_calls": []}
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
        plan = None
        plan_revision = 0
        plan_age = 0
        event("reset", metadata=row["metadata"], initial_fingerprint=fingerprint)
        row["frames"].append(save_frame(directory, 0, packet["policy_input"], 0., step=0))
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
                all_candidates = action_candidates(args.candidate_scale, previous_gripper)
                trigger_reasons = []
                if mode == "vlm-jev-dense":
                    trigger_reasons.append("dense_schedule")
                elif plan is None:
                    trigger_reasons.append("initial_plan")
                elif plan_age >= min(plan["plan_horizon_decisions"], args.vlm_max_plan_decisions):
                    trigger_reasons.append("plan_horizon")
                if mode == "vlm-jev-triggered" and stalled(
                        recent, args.stagnation_actions, args.stagnation_threshold):
                    trigger_reasons.append("proprioceptive_stagnation")
                vlm_called = bool(trigger_reasons)
                model_inputs = {}
                if vlm_called:
                    plan_revision += 1
                    plan, model_inputs = plan_with_vlm(
                        vlm, directory, index, plan_revision, policy_input, recent, all_candidates)
                    plan_age = 0
                if plan is None:
                    raise AssertionError("The hybrid policy requires an initialized VLM plan")
                candidates = {key: all_candidates[key] for key in plan["candidate_actions"]}
                selected, probabilities, selector_state = select_with_jev(
                    selector, policy_input["prompt"], policy_input["state"], plan, candidates, recent)
                metrics = confidence_metrics(probabilities, selected)
                jev_attempts = [{"plan_revision": plan_revision, "selection": selected,
                                 "probabilities": probabilities, "confidence": metrics}]
                low_confidence = confidence_triggers(metrics, args)
                if mode == "vlm-jev-triggered" and low_confidence and not vlm_called:
                    trigger_reasons.extend("jev_" + reason for reason in low_confidence)
                    plan_revision += 1
                    plan, model_inputs = plan_with_vlm(
                        vlm, directory, index, plan_revision, policy_input, recent, all_candidates)
                    plan_age = 0
                    vlm_called = True
                    candidates = {key: all_candidates[key] for key in plan["candidate_actions"]}
                    selected, probabilities, selector_state = select_with_jev(
                        selector, policy_input["prompt"], policy_input["state"], plan, candidates, recent)
                    metrics = confidence_metrics(probabilities, selected)
                    jev_attempts.append({"plan_revision": plan_revision, "selection": selected,
                                         "probabilities": probabilities, "confidence": metrics,
                                         "after_vlm_replan": True})
                actions = [candidates[selected]["action"]] * args.action_repeat
                decision.update(source=("on-demand local VLM planning plus Jev selection"
                                        if mode == "vlm-jev-triggered" else
                                        "dense local VLM planning plus Jev selection"),
                                vlm_called=vlm_called, vlm_trigger_reasons=trigger_reasons,
                                vlm_plan=plan, plan_revision=plan_revision, plan_age=plan_age,
                                jev_attempts=jev_attempts, confidence=metrics,
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
                state_before = packet["policy_input"]["state"]
                packet = worker.request({"command": "step", "action": action, "capture": True,
                                         "allow_unbounded": mode == "pi05"})
                row["steps"] = packet["policy_input"]["step"] - args.settle_steps
                previous_gripper = float(action[-1])
                state_after = packet["policy_input"]["state"]
                delta = math.sqrt(sum((float(after) - float(before)) ** 2
                                      for before, after in zip(state_before, state_after)))
                executed = {"step": row["steps"], "action": action, "success": packet["success"],
                            "state_delta_l2": delta}
                recent.append(executed)
                row["frames"].append(save_frame(
                    directory, len(row["frames"]), packet["policy_input"],
                    time.monotonic() - rollout_started, step=row["steps"], action=action, decision_index=index))
                event("step", **executed)
                if packet["success"] or packet["truncated"]:
                    row["success"] = bool(packet["success"])
                    row["status"] = "success" if packet["success"] else "step_budget"
                    break
            if mode != "pi05":
                plan_age += 1
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
        vlm_calls = [sum(call.get("provider") == "chat" for call in row.get("api_calls", [])) for row in group]
        jev_calls = [sum(call.get("provider") == "jev" for call in row.get("api_calls", [])) for row in group]
        replans = [sum(bool(decision.get("vlm_called")) for decision in row.get("decisions", [])) for row in group]
        result[mode] = {"planned": len(group), "scored": len(scored),
                        "successes": sum(bool(row["success"]) for row in scored),
                        "success_rate": (sum(bool(row["success"]) for row in scored) / len(scored)) if scored else None,
                        "mean_environment_steps": statistics.mean(row["steps"] for row in scored) if scored else None,
                        "mean_wall_seconds": statistics.mean(row["wall_seconds"] for row in group) if group else None,
                        "total_vlm_requests": sum(vlm_calls), "total_jev_requests": sum(jev_calls),
                        "mean_vlm_requests": statistics.mean(vlm_calls) if vlm_calls else None,
                        "mean_jev_requests": statistics.mean(jev_calls) if jev_calls else None,
                        "total_vlm_plans": sum(replans),
                        "complete": len(scored) == len(group) and all(row["status"] not in {"setup_error", "runtime_error"} for row in group)}
    return result


def run(args):
    manifest = load_manifest(args.manifest)
    if manifest["backend"] != "libero":
        raise ValueError("phase2-compare requires a LIBERO manifest")
    modes = list(args.modes)
    if not modes or len(modes) != len(set(modes)) or any(mode not in MODES for mode in modes):
        raise ValueError("Modes must be unique values from the phase-two mode set")
    if (not 1 <= args.max_steps <= 1000 or not 1 <= args.max_calls <= 1000
            or not 1 <= args.action_repeat <= 20 or not 1 <= args.pi05_replan_steps <= 50
            or not 0 <= args.settle_steps <= 50 or not 0 < args.candidate_scale <= .5 or not 1 <= args.timeout <= 14400
            or not 0 < args.max_usd <= 100 or not 0 <= args.request_retries <= 2
            or not 0 <= args.jev_confidence_threshold <= 1 or not 0 <= args.jev_margin_threshold <= 1
            or not 0 <= args.jev_entropy_threshold <= 1 or not 1 <= args.vlm_max_plan_decisions <= 8
            or not 0 <= args.stagnation_actions <= 20 or not 0 < args.stagnation_threshold <= 1):
        raise ValueError("Invalid phase-two experiment bounds")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "reproduction").mkdir()
    hashes = {}
    for name in SOURCE_FILES:
        source = Path(__file__).with_name(name)
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        shutil.copy2(source, output / "reproduction" / name)
    renderer = Path(__file__).parents[2] / "scripts" / "render_phase2_comparison.py"
    hashes[renderer.name] = hashlib.sha256(renderer.read_bytes()).hexdigest()
    shutil.copy2(renderer, output / "reproduction" / renderer.name)
    protocol = {"version": PROTOCOL, "created_at": datetime.now(timezone.utc).isoformat(),
                "manifest": manifest, "modes": modes, "source_sha256": hashes,
                "policy_contract": {"pi05": "direct finite 7D action chunks with official unclipped execution; Jev is never called",
                    "vlm-jev-triggered": "cached local-VLM candidate plan; Jev selects until an explicit replan trigger",
                    "vlm-jev-dense": "fresh local-VLM candidate plan before every Jev selection"},
                "budget": {key: getattr(args, key) for key in ("max_steps", "max_calls", "timeout", "max_usd",
                    "action_repeat", "candidate_scale", "pi05_replan_steps", "settle_steps",
                    "jev_confidence_threshold", "jev_margin_threshold", "jev_entropy_threshold",
                    "vlm_max_plan_decisions", "stagnation_actions", "stagnation_threshold")},
                "endpoints": {"pi05": {"host": args.pi05_host, "port": args.pi05_port},
                              "vlm": {"base_url": args.vlm_base_url, "model": args.vlm_model,
                                      "revision": args.vlm_revision},
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
    parser.add_argument("--vlm-model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--vlm-revision", default="unspecified")
    parser.add_argument("--jev-connection-source", choices=["environment", "saved"], default="environment")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--settle-steps", type=int, default=10,
                        help="Official pre-policy LIBERO dummy steps; excluded from max-steps")
    parser.add_argument("--max-calls", type=int, default=400)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--max-usd", type=float, default=10)
    parser.add_argument("--action-repeat", type=int, default=5)
    parser.add_argument("--candidate-scale", type=float, default=.5)
    parser.add_argument("--jev-confidence-threshold", type=float, default=.35)
    parser.add_argument("--jev-margin-threshold", type=float, default=.08)
    parser.add_argument("--jev-entropy-threshold", type=float, default=.90)
    parser.add_argument("--vlm-max-plan-decisions", type=int, default=6)
    parser.add_argument("--stagnation-actions", type=int, default=3)
    parser.add_argument("--stagnation-threshold", type=float, default=.001)
    parser.add_argument("--request-retries", type=int, default=1)
    parser.add_argument("--continue-on-error", action="store_true")
