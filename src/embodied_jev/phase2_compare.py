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
from .phase2_chunks import generate_grounded_action_chunks, nominal_chunk, validate_keyframe
from .policies import environment_connection, validate_answer


PROTOCOL = "libero-pi05-vs-local-vlm-jev-v6-reference-grounded-action-chunks"
MODES = ("pi05", "vlm-jev-triggered", "vlm-jev-dense", "vlm-chunk-no-jev")
SOURCE_FILES = ("phase2_compare.py", "benchmark_worker.py", "evaluation.py",
                "libero_policy.py", "phase2_chunks.py", "evaluation_meter.py", "policies.py")

VLM_INVENTORY_SYSTEM = """You are a task-blind visual inventory observer.
The user intentionally withholds the robot task so that it cannot bias recognition.
Inspect the full external scene and the magnified region gallery. For every supplied
proposal ID, report only ordinary visible category, package shape, genuinely legible
text (or "unreadable"), and confidence. Do not guess a desired object, action, pose,
coordinate, or task. Blurry strokes are not words. Use commonplace package priors:
soup is usually in a cylindrical food can, milk in a tall carton, cream cheese in a
small box or tub, and dressing in a bottle, unless clear visible evidence contradicts
the prior. Return exactly:
{"regions":[{"proposal_id":"region_00","category":"basket","shape":"open woven bin",
"visible_text":"unreadable","confidence":0.9}]}"""

VLM_REFERENCE_MATCH_SYSTEM = """You are an appearance-matching juror.
The target_reference is a public static texture for the language-named object and
contains no scene pose, segmentation, state, or success information. Match its colors,
artwork and package appearance to exactly one current region-gallery proposal. Compare
same-shape distractors directly. Return exactly:
{"choice":"region_00","evidence":"brief visual correspondence","confidence":0.8}"""

VLM_SYSTEM = """You are the semantic visual keypoint planner in a robot-control experiment.
Inspect the current upright external and wrist RGB images, the language task,
proprioceptive state and recent executed action chunks. Identify one visible physical
point on the object where the gripper should next make or maintain contact. Never
output a named robot skill, world coordinate, depth or robot action. Coordinates are
normalized within the exact upright image: u=0 left, u=1 right, v=0 top, v=1 bottom.
Grid lines mark normalized quarters and the cyan TCP marker is measured robot
proprioception, not a semantic object label.
The state includes external_rgbd_regions produced without simulator segmentation and
an external_region_gallery with a magnified tile for every candidate. When selecting
an external target, inspect that gallery, return its exact region ID as proposal_id,
and copy the corresponding u,v center. Numbered dots are proposals, not identities.
For a wrist target proposal_id must be null. Never invent an ID or point between
proposals. Reason from ordinary package shape and visible appearance; blurry marks
are not readable labels and must not be expanded into convenient text.
For release into an open receptacle, prefer a visibly correct proposal whose
candidate_kind is geometric_interior_probe; never target the exterior wall and never
use an interior probe to grasp a source object.
The state also includes task_blind_region_inventory generated before the task was
revealed to that observer. Use it to resist goal-induced label hallucinations. Apply
ordinary category priors: soup is normally a cylindrical can, milk a tall carton,
cream cheese a small box or tub, and dressing a bottle, absent clear contrary evidence.
When source_appearance_match is present, it is the independently matched source object
and must be used for grasp-stage targeting. It is not the destination.
Prefer a broad, rigid, visible interaction surface over an occluded edge. For grasp,
place the semantic seed well inside the exact object's visible identity-bearing body,
away from its silhouette and the support surface. The RGB-D controller grows a
depth-continuous component from that seed and refines the final TCP waypoint to the
object's grasp center, so you should not
try to point at a tiny top edge. For close
tasks, contact_mode push means press the movable surface toward its closed state;
for open or pick tasks use pull or grasp only when the language and image support it.
Before choosing a pixel, identify the exact task object by label, shape and context,
and explicitly distinguish it from every nearby distractor and from the destination
receptacle. Do not choose the basket, plate or other destination while the task object
has not yet been grasped. Do not substitute a visually similar product.
The RGB-D controller, not you, deprojects the semantic pixel and estimates a local
surface normal. Do not claim task success, force, hidden state or object pose.
Planning is stageful. If replan_reasons contains contact_transition_complete and
the previous keyframe used grasp, the controller has closed and executed a
closed-finger lift. Do not reacquire the source: locate the language-specified
destination and return contact_mode release while keeping the gripper closed until
arrival. After a completed release, visually verify the task.
If replan_reasons contains empty_grasp_recovery or grasp_lost_recovery, the object is
not held: reacquire the appearance-matched source with contact_mode grasp. Never
return release for an empty or lost grasp.
Return exactly:
{"phase":"approach|align|contact|manipulate|release|recover|verify|uncertain",
 "summary":"brief current scene description",
 "visible_evidence":"brief image-grounded evidence",
 "target_identity":"exact object or destination surface selected for this stage",
 "distractor_check":"why the pixel is not on a nearby distractor or the wrong stage target",
 "target":{"view":"external|wrist","proposal_id":"region_00 or null","u":0.50,"v":0.50,"confidence":0.80},
 "contact_mode":"push|pull|grasp|release|none",
 "gripper":"open|hold|close|unknown","magnitude":"fine|medium|coarse",
 "chunk_horizon":5,"plan_horizon_decisions":20,
 "completion_evidence":"one visible relation that would complete this keyframe",
 "risk":"brief risk or uncertainty","replan_condition":"one observable reason to replan"}
The generic controller produces bounded Hx7 trajectories from that measured 3D
contact geometry. A separate typed judge chooses among their exact numeric arrays.
Use external when it clearly sees the relevant object surface; use wrist only when
the target is actually visible there. Prefer fine motion near contact. A keyframe
should cover twenty judge decisions. Its verified RGB-D point is locked in world
coordinates until a contact transition, stall, loss, or horizon trigger; never
reinterpret a cached wrist pixel after the wrist camera moves."""

VLM_VERIFY_SYSTEM = """You independently verify one proposed semantic contact pixel.
The target_crop image is magnified around that exact pixel without a grid; external_context
shows the full scene for comparison. Use the language task and current interaction stage,
not any identity claim from the planner. Accept only if visible crop evidence
supports the exact task object or the correct destination for the current stage. The
central quarter itself must lie on that target; a target visible only near an edge is a
rejection. Reject
lookalike products, receptacles selected before grasp, robot parts, table, floor, empty
space, and crops whose identity is not actually supported. Use ordinary category and
package-shape knowledge, and never hallucinate words from blurry texture. Never trust
the proposal text over the crop. When target_reference is supplied, it is a static
appearance-only image of the language-named source object; compare the crop to it
directly. It is not a scene location and is not supplied for destination verification.
Return exactly:
{"accept":true,"observed_identity":"what is visibly centered","evidence":"brief reason"}"""


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


def annotated_planner_png(packet, tcp, proposals=()):
    """Rotate the model image, add normalized grid lines, and project measured TCP."""
    try:
        import numpy as np
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("The phase2 extra (NumPy and Pillow) is required") from exc
    image = Image.open(io.BytesIO(rotate_png_180(decode_png(packet)))).convert("RGB")
    draw = ImageDraw.Draw(image)
    for fraction in (.25, .5, .75):
        x, y = fraction * (image.width - 1), fraction * (image.height - 1)
        draw.line((x, 0, x, image.height), fill=(80, 160, 180), width=1)
        draw.line((0, y, image.width, y), fill=(80, 160, 180), width=1)
        draw.text((x + 2, 2), f"u={fraction:.2f}", fill="white")
        draw.text((2, y + 2), f"v={fraction:.2f}", fill="white")
    camera = np.linalg.solve(np.asarray(packet["camera_to_world"], dtype=float), np.r_[tcp[:3], 1.])
    if camera[2] > 0:
        projected = np.asarray(packet["intrinsics"], dtype=float) @ camera[:3]
        raw_u, raw_v = (projected[:2] / projected[2]).tolist()
        # LIBERO's stored observation is vertically flipped relative to the
        # calibrated OpenCV row convention; the subsequent 180-degree policy
        # transform therefore flips model X but restores camera Y.
        u, v = image.width - 1 - raw_u, raw_v
        if 0 <= u < image.width and 0 <= v < image.height:
            draw.ellipse((u - 8, v - 8, u + 8, v + 8), outline=(0, 255, 255), width=3)
            draw.text((u + 10, v - 6), "TCP", fill=(0, 255, 255))
    for proposal in proposals:
        x, y = proposal["u"] * (image.width - 1), proposal["v"] * (image.height - 1)
        label = proposal["id"].removeprefix("region_")
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(255, 210, 0), outline=(0, 0, 0))
        draw.text((x + 4, y - 6), label, fill=(255, 255, 255), stroke_width=1,
                  stroke_fill=(0, 0, 0))
    output = io.BytesIO()
    image.resize((512, 512), Image.Resampling.BICUBIC).save(output, format="PNG")
    return output.getvalue()


def inspection_png(packet):
    """Large unannotated view for reading labels; never used as a coordinate frame."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("The phase2 extra (Pillow) is required") from exc
    image = Image.open(io.BytesIO(rotate_png_180(decode_png(packet)))).convert("RGB")
    output = io.BytesIO()
    image.resize((1024, 1024), Image.Resampling.LANCZOS).save(output, format="PNG")
    return output.getvalue()


def proposal_gallery_png(packet, proposals):
    """Build a legible object-candidate gallery from the upright external frame."""
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("The phase2 extra (Pillow) is required") from exc
    source = Image.open(io.BytesIO(rotate_png_180(decode_png(packet)))).convert("RGB")
    tile_size, columns = 256, 3
    rows = max(1, math.ceil(len(proposals) / columns))
    gallery = Image.new("RGB", (columns * tile_size, rows * tile_size), (24, 24, 24))
    for index, proposal in enumerate(proposals):
        left, top, right, bottom = proposal["bbox_uv"]
        cx, cy = (left + right) / 2, (top + bottom) / 2
        half = max(.055, .65 * max(right - left, bottom - top))
        box = (max(0, int((cx - half) * source.width)),
               max(0, int((cy - half) * source.height)),
               min(source.width, int((cx + half) * source.width)),
               min(source.height, int((cy + half) * source.height)))
        crop = source.crop(box).resize((tile_size - 16, tile_size - 16),
                                       Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (tile_size, tile_size), (24, 24, 24))
        tile.paste(crop, (8, 8))
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, 96, 22), fill=(0, 0, 0))
        draw.text((5, 4), proposal["id"], fill=(255, 225, 0))
        gallery.paste(tile, ((index % columns) * tile_size, (index // columns) * tile_size))
    output = io.BytesIO()
    gallery.save(output, format="PNG")
    return output.getvalue()


def target_crop_png(packet, u, v):
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("The phase2 extra (Pillow) is required") from exc
    image = Image.open(io.BytesIO(rotate_png_180(decode_png(packet)))).convert("RGB")
    x, y = float(u) * (image.width - 1), float(v) * (image.height - 1)
    # A wide crop lets a nearby correct object falsely validate a pixel that is
    # actually on a distractor. Keep the verifier centered on the proposed point.
    radius = max(16, min(image.width, image.height) // 12)
    left, top = max(0, int(x - radius)), max(0, int(y - radius))
    right, bottom = min(image.width, int(x + radius)), min(image.height, int(y + radius))
    crop = image.crop((left, top, right, bottom)).resize((512, 512), Image.Resampling.BICUBIC)
    output = io.BytesIO()
    crop.save(output, format="PNG")
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
        "Choose exactly one offered numeric action chunk. The VLM supplied only a semantic pixel; calibrated RGB-D supplied "
        "the metric point, local surface normal and distances attached to each option. Judge the exact arrays against current "
        "proprioception, recent measured effects, visual change and risk. Prefer nominal grounded progress in free space, "
        "cautious progress near contact, and a measured surface-normal option during manipulation. Retreat or tangential "
        "motion requires evidence of adverse motion or a bad normal. Never invent an option.",
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


def validate_semantic_grounding(plan, geometry):
    if not geometry.get("valid"):
        raise ValueError("semantic pixel has invalid depth geometry")
    point = geometry.get("point_world", [])
    if len(point) != 3 or any(not math.isfinite(float(value)) for value in point):
        raise ValueError("semantic pixel has no finite world point")
    height = geometry.get("height_above_support_m")
    if plan["contact_mode"] == "grasp" and not geometry.get("grasp_geometry_refined"):
        raise ValueError("grasp seed did not produce a depth-connected object-center geometry")
    if plan["contact_mode"] == "grasp" and (height is None or height < .008):
        raise ValueError("grasp pixel landed on the dominant support surface, not above it on the task object")


def plan_with_vlm(vlm, worker, directory, decision_index, plan_revision, policy_input, recent,
                  previous_plan=None, trigger_reasons=None, interaction_stage=None,
                  semantic_cache=None):
    proposal_packet = worker.request({"command": "grounded_proposals", "view": "external",
                                      "image_rotated_180": True})
    private_proposals = proposal_packet["proposals"]
    proposal_geometry = {item["id"]: item.get("_interior_point_world")
                         for item in private_proposals}
    # Private metric geometry is consumed only after semantic selection.  The
    # VLM sees the same task-blind pixel candidates as before, never XYZ.
    proposals = [{key: value for key, value in item.items() if not key.startswith("_")}
                 for item in private_proposals]
    if not proposals:
        raise ValueError("RGB-D produced no external object-region proposals")
    visual_state = {"task": policy_input["prompt"], "proprioception": policy_input["state"],
                    "recent_executed_actions": recent[-4:],
                    "camera_frames": ["external", "wrist"],
                    "note": ("Images are upright after a 180-degree source transform. Metric RGB-D and camera calibration "
                             "are used only after you return a semantic pixel. No task success flag, object pose, force sensor, "
                             "task-named skill or candidate action is provided.")}
    if previous_plan is not None:
        visual_state["previous_keyframe"] = {
            field: previous_plan[field] for field in
            ("phase", "target_identity", "contact_mode", "gripper", "completion_evidence")}
    visual_state["replan_reasons"] = list(trigger_reasons or [])
    if interaction_stage is not None:
        visual_state["interaction_stage"] = interaction_stage
    visual_state["external_rgbd_regions"] = proposals
    images = {view: annotated_planner_png(
        image, policy_input["state"], proposals if view == "external" else ())
        for view, image in policy_input["images"].items()}
    images["external_detail_reference_only"] = inspection_png(policy_input["images"]["external"])
    images["external_region_gallery"] = proposal_gallery_png(
        policy_input["images"]["external"], proposals)
    model_inputs = {}
    for view, image in images.items():
        relative = f"inputs/{decision_index:05d}-plan{plan_revision:03d}-{view}-vlm.png"
        (directory / relative).write_bytes(image)
        model_inputs[view] = {"path": relative, "sha256": hashlib.sha256(image).hexdigest()}
    expected_ids = {proposal["id"] for proposal in proposals}
    cache_valid = bool(semantic_cache is not None
                       and semantic_cache.get("interaction_stage") == interaction_stage
                       and set(semantic_cache.get("proposal_ids", [])) == expected_ids
                       and isinstance(semantic_cache.get("inventory"), list))
    if cache_valid:
        inventory = semantic_cache["inventory"]
    else:
        inventory_answer = vlm.request("local_vlm_task_blind_inventory", {
            "proposal_ids": [proposal["id"] for proposal in proposals],
            "note": "The robot task is deliberately withheld. Describe visible objects only."
        }, system=VLM_INVENTORY_SYSTEM,
           images={"external_context": images["external_detail_reference_only"],
                   "external_region_gallery": images["external_region_gallery"]})
        inventory = inventory_answer.get("regions") if isinstance(inventory_answer, dict) else None
    if (not isinstance(inventory, list) or len(inventory) != len(expected_ids)
            or {item.get("proposal_id") for item in inventory if isinstance(item, dict)} != expected_ids):
        raise ValueError("Task-blind VLM inventory did not cover every RGB-D proposal exactly once")
    for item in inventory:
        if (set(item) != {"proposal_id", "category", "shape", "visible_text", "confidence"}
                or any(not isinstance(item[field], str) or not item[field]
                       for field in ("proposal_id", "category", "shape", "visible_text"))
                or isinstance(item["confidence"], bool)
                or not isinstance(item["confidence"], (int, float))
                or not 0 <= float(item["confidence"]) <= 1):
            raise ValueError("Task-blind VLM inventory returned an invalid region record")
    visual_state["task_blind_region_inventory"] = inventory
    source_stage = (interaction_stage == "source" if interaction_stage is not None else
                    not (previous_plan is not None
                         and previous_plan.get("contact_mode") == "grasp"
                         and "contact_transition_complete" in (trigger_reasons or [])))
    required_source_proposal = None
    reference_image = None
    if source_stage:
        reference = worker.request({"command": "appearance_reference"})
        if reference.get("available"):
            reference_image = decode_png(reference["image"])
            reference_relative = (f"inputs/{decision_index:05d}-plan{plan_revision:03d}"
                                  "-target_appearance_reference-vlm.png")
            (directory / reference_relative).write_bytes(reference_image)
            model_inputs["target_appearance_reference"] = {
                "path": reference_relative,
                "sha256": hashlib.sha256(reference_image).hexdigest()}
            match = (semantic_cache.get("source_appearance_match") if cache_valid else None)
            if not isinstance(match, dict) or match.get("choice") not in expected_ids:
                match = vlm.request("local_vlm_reference_match", {
                    "target_noun_phrase": reference["target_noun_phrase"],
                    "available_proposal_ids": sorted(expected_ids),
                    "task_blind_region_inventory": inventory,
                    "reference_source": reference["source"]},
                    system=VLM_REFERENCE_MATCH_SYSTEM,
                    images={"target_reference": reference_image,
                            "current_region_gallery": images["external_region_gallery"]})
                if (not isinstance(match, dict)
                        or set(match) != {"choice", "evidence", "confidence"}
                        or match["choice"] not in expected_ids
                        or not isinstance(match["evidence"], str) or not match["evidence"]
                        or isinstance(match["confidence"], bool)
                        or not isinstance(match["confidence"], (int, float))
                        or not 0 <= float(match["confidence"]) <= 1):
                    raise ValueError("Appearance-reference matcher returned an invalid choice")
            required_source_proposal = match["choice"]
            visual_state["source_appearance_match"] = {
                **match, "target_noun_phrase": reference["target_noun_phrase"],
                "source": reference["source"]}
    if semantic_cache is not None and not cache_valid:
        semantic_cache.clear()
        semantic_cache.update(interaction_stage=interaction_stage,
                              proposal_ids=sorted(expected_ids), inventory=inventory,
                              source_appearance_match=(match if source_stage and reference_image is not None
                                                       else None))
    rejected = []
    for attempt in range(6):
        state = dict(visual_state)
        if rejected:
            state["rejected_groundings"] = rejected
            state["correction"] = ("The prior semantic pixel was rejected by measured RGB-D geometry before motion. "
                                   "Choose a visibly different point on the exact current-stage target (source before "
                                   "grasp, destination after the closed-finger lift); never repeat that pixel. "
                                   "Use the external view for this correction because it shows the full object inventory; "
                                   "do not claim an object is centered in the wrist view without readable evidence.")
            state["correction_target_view"] = "external"
        answer = vlm.request("local_vlm_keyframe", state, system=VLM_SYSTEM, images=images)
        if isinstance(answer, dict):
            for field in ("summary", "visible_evidence", "target_identity", "distractor_check",
                          "completion_evidence", "risk", "replan_condition"):
                if isinstance(answer.get(field), str) and len(answer[field]) > 600:
                    answer[field] = answer[field][:600]
        try:
            plan = validate_keyframe(answer)
        except ValueError as exc:
            rejected.append({"attempt": attempt + 1,
                             "reason": str(exc),
                             "target_identity": (answer.get("target_identity")
                                                 if isinstance(answer, dict) else None),
                             "target": (answer.get("target") if isinstance(answer, dict) else None)})
            continue
        if ((interaction_stage == "destination"
             or (interaction_stage is None and previous_plan is not None
                 and previous_plan.get("contact_mode") == "grasp"
                 and "contact_transition_complete" in (trigger_reasons or [])))
                and plan["contact_mode"] != "release"):
            rejected.append({"attempt": attempt + 1,
                             "reason": "completed grasp transition requires destination release planning",
                             "target_identity": plan["target_identity"], "target": plan["target"]})
            continue
        if (any(reason in {"empty_grasp_recovery", "grasp_lost_recovery"}
                for reason in (trigger_reasons or [])) and plan["contact_mode"] != "grasp"):
            rejected.append({"attempt": attempt + 1,
                             "reason": "grasp recovery requires contact_mode grasp",
                             "target_identity": plan["target_identity"], "target": plan["target"]})
            continue
        target = plan["target"]
        selected_proposal = None
        if target["view"] == "external":
            if required_source_proposal is not None:
                target["proposal_id"] = required_source_proposal
            selected_proposal = next(
                (item for item in proposals if item["id"] == target["proposal_id"]), None)
            if selected_proposal is None:
                rejected.append({"attempt": attempt + 1,
                                 "reason": "target did not return an available RGB-D proposal_id",
                                 "target_identity": plan["target_identity"], "target": target})
                continue
            target.update(u=selected_proposal["u"], v=selected_proposal["v"])
        elif target["proposal_id"] is not None:
            rejected.append({"attempt": attempt + 1,
                             "reason": "wrist targets require proposal_id null",
                             "target_identity": plan["target_identity"], "target": target})
            continue
        geometry = worker.request({"command": "grounded_geometry", "view": target["view"],
                                   "u": target["u"], "v": target["v"],
                                   "image_rotated_180": True,
                                   "contact_mode": plan["contact_mode"]})["geometry"]
        interior = (proposal_geometry.get(target.get("proposal_id"))
                    if plan["contact_mode"] == "release" else None)
        if (selected_proposal is not None
                and selected_proposal.get("candidate_kind") == "geometric_interior_probe"
                and isinstance(interior, list) and len(interior) == 3):
            geometry = {**geometry, "point_world": [float(value) for value in interior],
                        "normal_toward_camera_world": [0., 0., 1.],
                        "height_above_support_m": float(interior[2]),
                        "source": "VLM-selected RGB-D receptacle walls plus measured interior center"}
        try:
            validate_semantic_grounding(plan, geometry)
            crop = target_crop_png(policy_input["images"][target["view"]], target["u"], target["v"])
            crop_relative = f"inputs/{decision_index:05d}-plan{plan_revision:03d}-verify{attempt + 1:02d}.png"
            (directory / crop_relative).write_bytes(crop)
            model_inputs[f"target_crop_attempt_{attempt + 1}"] = {
                "path": crop_relative, "sha256": hashlib.sha256(crop).hexdigest()}
            verifications = []
            vote_count = (1 if cache_valid
                          and semantic_cache.get("verified_proposal_id") == target.get("proposal_id") else 3)
            for vote in range(vote_count):
                verification_images = {"target_crop": crop,
                                       "external_context": images["external_detail_reference_only"]}
                if reference_image is not None:
                    verification_images["target_reference"] = reference_image
                verification = vlm.request("local_vlm_target_verification", {
                    "task": policy_input["prompt"], "phase": plan["phase"],
                    "proposed_contact_mode": plan["contact_mode"],
                    "independent_vote": vote + 1,
                    "crop_center": {"view": target["view"], "u": target["u"], "v": target["v"]}},
                    system=VLM_VERIFY_SYSTEM, images=verification_images)
                if (not isinstance(verification, dict)
                        or set(verification) != {"accept", "observed_identity", "evidence"}
                        or type(verification["accept"]) is not bool
                        or any(not isinstance(verification[field], str) or not verification[field]
                               for field in ("observed_identity", "evidence"))):
                    raise ValueError("target crop verifier returned an invalid schema")
                verifications.append(verification)
            accepts = sum(vote["accept"] for vote in verifications)
            if accepts < (1 if vote_count == 1 else 2):
                reasons = "; ".join(vote["evidence"][:100] for vote in verifications if not vote["accept"])
                raise ValueError("target crop consensus rejected: " + reasons[:240])
            geometry = {**geometry, "target_verification_votes": verifications,
                        "target_verification_accepts": accepts,
                        "semantic_cache_reused": cache_valid,
                        "selected_rgbd_region": selected_proposal}
            if semantic_cache is not None:
                semantic_cache["verified_proposal_id"] = target.get("proposal_id")
            return plan, model_inputs, geometry
        except ValueError as exc:
            rejected.append({"attempt": attempt + 1, "reason": str(exc),
                             "target_identity": plan["target_identity"], "target": target})
            if cache_valid and source_stage and reference_image is not None and attempt == 0:
                # Proposal IDs are ranked geometric components, not persistent
                # object tracks.  Robot motion can reorder them.  A fresh crop
                # rejection invalidates only the cached ID: rematch the same
                # appearance reference against the current gallery, then use
                # the normal three-vote verification on the next attempt.
                refreshed_inventory_answer = vlm.request("local_vlm_task_blind_inventory", {
                    "proposal_ids": [proposal["id"] for proposal in proposals],
                    "note": "The robot task is deliberately withheld. Describe visible objects only."
                }, system=VLM_INVENTORY_SYSTEM,
                   images={"external_context": images["external_detail_reference_only"],
                           "external_region_gallery": images["external_region_gallery"]})
                refreshed_inventory = (refreshed_inventory_answer.get("regions")
                                       if isinstance(refreshed_inventory_answer, dict) else None)
                if (isinstance(refreshed_inventory, list)
                        and len(refreshed_inventory) == len(expected_ids)
                        and {item.get("proposal_id") for item in refreshed_inventory
                             if isinstance(item, dict)} == expected_ids
                        and all(set(item) == {"proposal_id", "category", "shape", "visible_text", "confidence"}
                                and all(isinstance(item[field], str) and item[field]
                                        for field in ("proposal_id", "category", "shape", "visible_text"))
                                and not isinstance(item["confidence"], bool)
                                and isinstance(item["confidence"], (int, float))
                                and 0 <= float(item["confidence"]) <= 1
                                for item in refreshed_inventory)):
                    inventory = refreshed_inventory
                    visual_state["task_blind_region_inventory"] = inventory
                    semantic_cache["inventory"] = inventory
                refreshed = vlm.request("local_vlm_reference_match", {
                    "target_noun_phrase": reference["target_noun_phrase"],
                    "available_proposal_ids": sorted(expected_ids),
                    "task_blind_region_inventory": inventory,
                    "reference_source": reference["source"]},
                    system=VLM_REFERENCE_MATCH_SYSTEM,
                    images={"target_reference": reference_image,
                            "current_region_gallery": images["external_region_gallery"]})
                if (isinstance(refreshed, dict)
                        and set(refreshed) == {"choice", "evidence", "confidence"}
                        and refreshed["choice"] in expected_ids
                        and isinstance(refreshed["evidence"], str) and refreshed["evidence"]
                        and not isinstance(refreshed["confidence"], bool)
                        and isinstance(refreshed["confidence"], (int, float))
                        and 0 <= float(refreshed["confidence"]) <= 1):
                    required_source_proposal = refreshed["choice"]
                    visual_state["source_appearance_match"] = {
                        **refreshed, "target_noun_phrase": reference["target_noun_phrase"],
                        "source": reference["source"]}
                    semantic_cache["source_appearance_match"] = refreshed
                    semantic_cache.pop("verified_proposal_id", None)
                    cache_valid = False
    raise ValueError("VLM failed RGB-D semantic grounding validation: " + rejected[-1]["reason"])


def ground_plan(worker, plan, policy_input, previous_gripper, candidate_scale, geometry=None,
                previous_controller_phase=None, previous_aperture=None):
    target = plan["target"]
    geometry = geometry or worker.request({"command": "grounded_geometry", "view": target["view"],
                                           "u": target["u"], "v": target["v"],
                                           "image_rotated_180": True,
                                           "contact_mode": plan["contact_mode"]})["geometry"]
    candidates = generate_grounded_action_chunks(
        plan, geometry, policy_input["state"], previous_gripper, candidate_scale,
        previous_controller_phase, previous_aperture)
    camera_packet = policy_input["images"][target["view"]]
    calibration = {"target_view": target["view"], "target_normalized_uv": [target["u"], target["v"]],
                   "camera_to_world": camera_packet["camera_to_world"],
                   "intrinsics": camera_packet["intrinsics"],
                   "vlm_image_transform": "rotate_180", "scene_depth_used": True,
                   "depth_policy": "metric RGB-D only after semantic pixel selection",
                   "geometry": geometry}
    return candidates, calibration


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
        plan_geometry = None
        plan_revision = 0
        plan_age = 0
        interaction_stage = "source"
        semantic_cache = {}
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
                prior = row["decisions"][-1] if row["decisions"] else None
                prior_candidate_geometry = {}
                if prior and prior.get("selected_chunk") in prior.get("candidate_chunks", {}):
                    prior_candidate_geometry = prior["candidate_chunks"][prior["selected_chunk"]].get("geometry", {})
                prior_controller_phase = prior_candidate_geometry.get("controller_phase")
                if prior_controller_phase == "post_grasp_lift":
                    current_aperture = abs(float(policy_input["state"][-2])
                                           - float(policy_input["state"][-1]))
                    if current_aperture > .008:
                        interaction_stage = "destination"
                        trigger_reasons.append("contact_transition_complete")
                    else:
                        interaction_stage = "source"
                        trigger_reasons.append("grasp_lost_recovery")
                elif prior_controller_phase == "contact_release":
                    current_aperture = abs(float(policy_input["state"][-2])
                                           - float(policy_input["state"][-1]))
                    # Opening under load takes more than one five-step chunk
                    # in LIBERO. Keep the verified destination plan and repeat
                    # the open command until the fingers have actually opened.
                    if current_aperture > .075:
                        interaction_stage = "verification"
                        trigger_reasons.append("contact_transition_complete")
                elif prior_controller_phase == "failed_grasp_reopen":
                    interaction_stage = "source"
                    semantic_cache.clear()
                    trigger_reasons.append("empty_grasp_recovery")
                elif prior_controller_phase in {"grasp_descent", "contact_close", "grasp_squeeze",
                                                  "release_descent"}:
                    # Final descent and finger contact form one atomic low-level
                    # transition.  The target is commonly occluded by the hand
                    # here, so dense semantic re-identification is unsafe; keep
                    # the already verified world geometry until contact resolves.
                    pass
                elif mode == "vlm-jev-dense" and interaction_stage == "source":
                    trigger_reasons.append("dense_schedule")
                elif plan is None:
                    trigger_reasons.append("initial_plan")
                elif (plan_age >= min(plan["plan_horizon_decisions"], args.vlm_max_plan_decisions)
                      and plan is not None and plan["contact_mode"] not in {"grasp", "release"}):
                    trigger_reasons.append("plan_horizon")
                if mode in {"vlm-jev-triggered", "vlm-chunk-no-jev"} and row["decisions"]:
                    prior_point = prior_candidate_geometry.get("point_world")
                    if prior_point is None:
                        prior_point = prior.get("calibration", {}).get("geometry", {}).get("point_world")
                    if isinstance(prior_point, list) and len(prior_point) == 3:
                        tcp = policy_input["state"][:3]
                        target_distance = math.sqrt(sum((float(prior_point[i]) - float(tcp[i])) ** 2
                                                        for i in range(3)))
                        if target_distance < .085 and plan["contact_mode"] not in {"grasp", "release"}:
                            trigger_reasons.append("grounded_target_reached")
                if (mode in {"vlm-jev-triggered", "vlm-chunk-no-jev"} and recent
                        and recent[-1].get("chunk_family") in {"retreat", "tangent_probe"}):
                    trigger_reasons.append("previous_probe_or_retreat")
                if (mode in {"vlm-jev-triggered", "vlm-chunk-no-jev"}
                        and prior_controller_phase not in {"contact_close", "grasp_squeeze"}
                        and current_visual_change is not None
                        and current_visual_change < args.visual_stagnation_threshold
                        and stalled(recent, args.stagnation_actions, args.stagnation_threshold)):
                    trigger_reasons.append("visual_and_proprioceptive_stagnation")
                vlm_called = bool(trigger_reasons)
                model_inputs = {}
                if vlm_called:
                    plan_revision += 1
                    plan, model_inputs, plan_geometry = plan_with_vlm(
                        vlm, worker, directory, index, plan_revision, policy_input, recent,
                        previous_plan=plan, trigger_reasons=trigger_reasons,
                        interaction_stage=interaction_stage, semantic_cache=semantic_cache)
                    plan_age = 0
                if plan is None:
                    raise AssertionError("The hybrid policy requires an initialized VLM plan")
                candidates, calibration = ground_plan(
                    worker, plan, policy_input, previous_gripper, args.candidate_scale,
                    plan_geometry, prior_controller_phase,
                    prior_candidate_geometry.get("gripper_aperture_m"))
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
                if (mode == "vlm-jev-triggered" and low_confidence and not vlm_called
                        and prior_controller_phase not in {"contact_close", "grasp_squeeze"}):
                    trigger_reasons.extend("jev_" + reason for reason in low_confidence)
                    plan_revision += 1
                    plan, model_inputs, plan_geometry = plan_with_vlm(
                        vlm, worker, directory, index, plan_revision, policy_input, recent,
                        previous_plan=plan, trigger_reasons=trigger_reasons,
                        interaction_stage=interaction_stage, semantic_cache=semantic_cache)
                    plan_age = 0
                    vlm_called = True
                    candidates, calibration = ground_plan(
                        worker, plan, policy_input, previous_gripper, args.candidate_scale,
                        plan_geometry, prior_controller_phase,
                        prior_candidate_geometry.get("gripper_aperture_m"))
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
                                calibration=calibration, interaction_stage=interaction_stage,
                                recent_visual_change=current_visual_change,
                               image_transform="rotate 180 degrees; normalized grid and measured TCP overlay; local VLM owns resize/tokenization")
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
            or not 0 <= args.jev_entropy_threshold <= 1 or not 1 <= args.vlm_max_plan_decisions <= 40
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
                "policy_contract": {"pi05": "official openpi LIBERO environment and direct finite 7D action chunks; Jev is never called",
                    "vlm-jev-triggered": "cached semantic pixel keyframe; calibrated RGB-D geometry creates exact Hx7 chunks for Jev",
                    "vlm-jev-dense": ("fresh semantic pixel keyframe before each non-atomic source-stage Jev judgment; "
                                      "verified geometry is locked through final contact and destination transport"),
                    "vlm-chunk-no-jev": "cached semantic pixel keyframe with deterministic nominal grounded Hx7 selection",
                    "hybrid_executor": ("task-agnostic semantic contact point, appearance-only public asset reference, "
                                        "metric deprojection and staged XYZ chunks; no scene object pose, segmentation, "
                                        "success state or task-named skill")},
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
    parser.add_argument("--vlm-max-plan-decisions", type=int, default=20)
    parser.add_argument("--stagnation-actions", type=int, default=3)
    parser.add_argument("--stagnation-threshold", type=float, default=.001)
    parser.add_argument("--visual-stagnation-threshold", type=float, default=.002,
                        help="Mean normalized two-camera RGB change below which a stalled robot triggers replanning")
    parser.add_argument("--request-retries", type=int, default=1)
    parser.add_argument("--continue-on-error", action="store_true")
