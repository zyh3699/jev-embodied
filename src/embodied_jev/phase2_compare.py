"""Phase-two LIBERO evaluation with interface-aligned continuous actions.

pi0.5 and every hybrid route ultimately emit Hx7 action chunks. The local VLM
defines a task-agnostic visual keyframe in an observed camera frame; calibrated
geometry generates multiple numeric chunks, and Jev judges those chunks. No
task-named skill or privileged object pose is available to the hybrid policy.
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
from .phase2_chunks import generate_action_chunks, nominal_chunk, validate_keyframe
from .policies import environment_connection, validate_answer


PROTOCOL = "libero-pi05-vs-local-vlm-jev-v4-action-chunks"
MODES = ("pi05", "vlm-jev-triggered", "vlm-jev-dense", "vlm-chunk-no-jev")
SOURCE_FILES = ("phase2_compare.py", "benchmark_worker.py", "evaluation.py",
                "libero_policy.py", "phase2_chunks.py", "evaluation_meter.py", "policies.py")

VLM_SYSTEM = """You are the low-frequency visual keyframe planner in a robot-control experiment.
Inspect the current upright external and wrist RGB images, the language task,
proprioceptive state and recent executed action chunks. Describe only the next
short, observable keyframe. Never output a named robot skill or a world-frame
action. Choose motion_frame as the camera whose image best supports the motion.
In that exact upright image, translation_camera x is image-right, y is image-down,
and z is depth away from the camera; rotation_camera uses the same camera axes.
Use unknown when RGB does not support a direction. The calibrated executor, not
you, maps the camera directions into world coordinates. Do not claim task success,
contact force, hidden object state, exact depth, or object pose. Return exactly:
{"phase":"approach|align|contact|manipulate|release|recover|verify|uncertain",
 "summary":"brief current scene description",
 "visible_evidence":"brief image-grounded evidence",
 "motion_frame":"external|wrist",
 "translation_camera":{"x":"negative|hold|positive|unknown","y":"...","z":"..."},
 "rotation_camera":{"x":"negative|hold|positive|unknown","y":"...","z":"..."},
 "gripper":"open|hold|close|unknown","magnitude":"fine|medium|coarse",
 "chunk_horizon":5,"plan_horizon_decisions":4,
 "completion_evidence":"one visible relation that would complete this keyframe",
 "risk":"brief risk or uncertainty","replan_condition":"one observable reason to replan"}
The action-chunk generator always creates bounded simultaneous 7D trajectories,
including cautious, nominal, assertive, translation-only, rotation-only, recovery,
gripper-only and hold alternatives. A separate typed judge chooses among their
exact numeric arrays. Prefer fine motion near contact and a short horizon when
uncertain. A keyframe must cover at least two judge decisions."""


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


def select_with_jev(client, prompt, proprioception, keyframe, candidates, recent, visual_change):
    options = {key: (value["description"] + " Family=" + value["family"]
                     + "; exact normalized Hx7 sequence=" + json.dumps(value["actions"]))
               for key, value in candidates.items()}
    question = {"action_chunk": {"type": "choice", "instructions":
        "Choose exactly one offered numeric action chunk. The local-VLM keyframe is uncertain visual evidence, not a command. "
        "Judge the exact arrays against current proprioception, recent measured effects, visual-change evidence and risk. "
        "Prefer image-plane-only motion when RGB supports lateral alignment but not depth, camera-depth-only motion when "
        "depth direction is clear, cautious or hold under ambiguity, nominal when full 3D progress is supported, assertive "
        "only with clear free-space evidence, and recovery after adverse or stalled motion. Never invent an option.",
        "criteria": options}}
    state = {"task": prompt, "proprioception": proprioception, "cached_visual_keyframe": keyframe,
             "recent_executed_actions": recent[-6:], "recent_visual_change": visual_change,
             "candidate_chunk_count": len(options)}
    answer = client.request("jev_action_chunk_selection", state, questions=question)
    if not isinstance(answer, dict) or set(answer) != {"action_chunk"}:
        raise ValueError("Jev must answer exactly one action-chunk question")
    selected, probabilities = validate_answer(answer["action_chunk"], options, require_highest=False)
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


def visual_change(previous, current):
    """Mean normalized RGB change in the two observable cameras."""
    if previous is None:
        return None
    try:
        from PIL import Image, ImageChops, ImageStat
    except ImportError as exc:
        raise RuntimeError("The phase2 extra (Pillow) is required") from exc
    values = []
    for view in sorted(current["images"]):
        before = Image.open(io.BytesIO(decode_png(previous["images"][view]))).convert("RGB").resize((64, 64))
        after = Image.open(io.BytesIO(decode_png(current["images"][view]))).convert("RGB").resize((64, 64))
        values.append(sum(ImageStat.Stat(ImageChops.difference(before, after)).mean) / (3 * 255))
    return statistics.mean(values)


def plan_with_vlm(vlm, directory, decision_index, plan_revision, policy_input, recent):
    visual_state = {"task": policy_input["prompt"], "proprioception": policy_input["state"],
                    "recent_executed_actions": recent[-4:],
                    "camera_frames": ["external", "wrist"],
                    "note": ("Images are upright after a 180-degree source transform. Known camera calibration is used "
                             "only after your response. No task success flag, object pose, scene depth, force sensor, "
                             "task-named skill or candidate action is provided.")}
    images = {view: rotate_png_180(decode_png(image)) for view, image in policy_input["images"].items()}
    model_inputs = {}
    for view, image in images.items():
        relative = f"inputs/{decision_index:05d}-plan{plan_revision:03d}-{view}-vlm.png"
        (directory / relative).write_bytes(image)
        model_inputs[view] = {"path": relative, "sha256": hashlib.sha256(image).hexdigest()}
    plan = validate_keyframe(vlm.request("local_vlm_keyframe", visual_state,
                                         system=VLM_SYSTEM, images=images))
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
            if mode != "vlm-chunk-no-jev":
                selector = ModelClient("jev", jev_connection(args), budget, request_retries=args.request_retries)
        previous_gripper, recent = -1., []
        plan = None
        plan_revision = 0
        plan_age = 0
        previous_decision_input = None
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
            current_visual_change = visual_change(previous_decision_input, policy_input)
            previous_decision_input = policy_input
            if mode == "pi05":
                chunk, latency, model_inputs = pi05.infer(policy_input)
                actions = chunk[:args.pi05_replan_steps]
                decision.update(source="pi0.5 direct action chunk", chunk_length=len(chunk),
                                executed_chunk_length=len(actions), inference_latency_ms=latency,
                                model_input_images=model_inputs,
                                image_transform="rotate 180 degrees, resize_with_pad 224x224, uint8")
            else:
                budget.check()
                trigger_reasons = []
                if mode == "vlm-jev-dense":
                    trigger_reasons.append("dense_schedule")
                elif plan is None:
                    trigger_reasons.append("initial_plan")
                elif plan_age >= min(plan["plan_horizon_decisions"], args.vlm_max_plan_decisions):
                    trigger_reasons.append("plan_horizon")
                if (mode in {"vlm-jev-triggered", "vlm-chunk-no-jev"} and recent
                        and recent[-1].get("chunk_family") in {"hold", "recover"}):
                    trigger_reasons.append("previous_hold_or_recovery")
                if (mode in {"vlm-jev-triggered", "vlm-chunk-no-jev"}
                        and current_visual_change is not None
                        and current_visual_change < args.visual_stagnation_threshold
                        and stalled(recent, args.stagnation_actions, args.stagnation_threshold)):
                    trigger_reasons.append("visual_and_proprioceptive_stagnation")
                vlm_called = bool(trigger_reasons)
                model_inputs = {}
                if vlm_called:
                    plan_revision += 1
                    plan, model_inputs = plan_with_vlm(
                        vlm, directory, index, plan_revision, policy_input, recent)
                    plan_age = 0
                if plan is None:
                    raise AssertionError("The hybrid policy requires an initialized VLM plan")
                camera_packet = policy_input["images"][plan["motion_frame"]]
                candidates = generate_action_chunks(
                    plan, camera_packet["camera_to_world"], previous_gripper, args.candidate_scale)
                calibration = {"motion_frame": plan["motion_frame"],
                               "camera_to_world": camera_packet["camera_to_world"],
                               "intrinsics": camera_packet["intrinsics"],
                               "vlm_image_transform": "rotate_180",
                               "scene_depth_used": False}
                if mode == "vlm-chunk-no-jev":
                    selected = nominal_chunk(candidates)
                    probabilities, metrics, selector_state, jev_attempts, low_confidence = {}, None, None, [], []
                else:
                    selected, probabilities, selector_state = select_with_jev(
                        selector, policy_input["prompt"], policy_input["state"], plan, candidates, recent,
                        current_visual_change)
                    metrics = confidence_metrics(probabilities, selected)
                    jev_attempts = [{"plan_revision": plan_revision, "selection": selected,
                                     "probabilities": probabilities, "confidence": metrics}]
                    low_confidence = confidence_triggers(metrics, args)
                if mode == "vlm-jev-triggered" and low_confidence and not vlm_called:
                    trigger_reasons.extend("jev_" + reason for reason in low_confidence)
                    plan_revision += 1
                    plan, model_inputs = plan_with_vlm(
                        vlm, directory, index, plan_revision, policy_input, recent)
                    plan_age = 0
                    vlm_called = True
                    camera_packet = policy_input["images"][plan["motion_frame"]]
                    candidates = generate_action_chunks(
                        plan, camera_packet["camera_to_world"], previous_gripper, args.candidate_scale)
                    calibration = {"motion_frame": plan["motion_frame"],
                                   "camera_to_world": camera_packet["camera_to_world"],
                                   "intrinsics": camera_packet["intrinsics"],
                                   "vlm_image_transform": "rotate_180", "scene_depth_used": False}
                    selected, probabilities, selector_state = select_with_jev(
                        selector, policy_input["prompt"], policy_input["state"], plan, candidates, recent,
                        current_visual_change)
                    metrics = confidence_metrics(probabilities, selected)
                    jev_attempts.append({"plan_revision": plan_revision, "selection": selected,
                                         "probabilities": probabilities, "confidence": metrics,
                                         "after_vlm_replan": True})
                actions = candidates[selected]["actions"][:args.action_repeat]
                source = {"vlm-jev-triggered": "on-demand visual keyframe plus Jev action-chunk judgment",
                          "vlm-jev-dense": "dense visual keyframe plus Jev action-chunk judgment",
                          "vlm-chunk-no-jev": "on-demand visual keyframe plus deterministic nominal chunk"}[mode]
                decision.update(source=source,
                                vlm_called=vlm_called, vlm_trigger_reasons=trigger_reasons,
                                vlm_keyframe=plan, plan_revision=plan_revision, plan_age=plan_age,
                                jev_attempts=jev_attempts, confidence=metrics,
                                selection=selected, probabilities=probabilities,
                                selected_chunk=selected, candidate_chunks=candidates,
                                action_chunk=actions, chunk_length=len(candidates[selected]["actions"]),
                                executed_chunk_length=len(actions), selector_state=selector_state,
                                candidate_count=len(candidates), model_input_images=model_inputs,
                                calibration=calibration, recent_visual_change=current_visual_change,
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
                if mode != "pi05":
                    executed["chunk"] = selected
                    executed["chunk_family"] = candidates[selected]["family"]
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
            or not 0 <= args.stagnation_actions <= 20 or not 0 < args.stagnation_threshold <= 1
            or not 0 <= args.visual_stagnation_threshold <= 1):
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
                    "vlm-jev-triggered": "cached camera-frame visual keyframe; Jev selects an exact Hx7 numeric chunk until an explicit replan trigger",
                    "vlm-jev-dense": "fresh camera-frame visual keyframe before every Jev Hx7 chunk judgment",
                    "vlm-chunk-no-jev": "cached camera-frame visual keyframe with deterministic nominal Hx7 chunk selection",
                    "hybrid_executor": "calibrated camera directions generate bounded simultaneous 7D OSC chunks without task-named skills or scene depth"},
                "budget": {key: getattr(args, key) for key in ("max_steps", "max_calls", "timeout", "max_usd",
                    "action_repeat", "candidate_scale", "pi05_replan_steps", "settle_steps",
                    "jev_confidence_threshold", "jev_margin_threshold", "jev_entropy_threshold",
                    "vlm_max_plan_decisions", "stagnation_actions", "stagnation_threshold",
                    "visual_stagnation_threshold")},
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
    parser.add_argument("--action-repeat", type=int, default=5,
                        help="Maximum environment steps executed from each hybrid Hx7 action chunk")
    parser.add_argument("--candidate-scale", type=float, default=.5,
                        help="Maximum normalized channel magnitude used by the generic chunk generator")
    parser.add_argument("--jev-confidence-threshold", type=float, default=.35)
    parser.add_argument("--jev-margin-threshold", type=float, default=.08)
    parser.add_argument("--jev-entropy-threshold", type=float, default=.90)
    parser.add_argument("--vlm-max-plan-decisions", type=int, default=6)
    parser.add_argument("--stagnation-actions", type=int, default=3)
    parser.add_argument("--stagnation-threshold", type=float, default=.001)
    parser.add_argument("--visual-stagnation-threshold", type=float, default=.002,
                        help="Mean normalized two-camera RGB change below which a stalled robot triggers replanning")
    parser.add_argument("--request-retries", type=int, default=1)
    parser.add_argument("--continue-on-error", action="store_true")
