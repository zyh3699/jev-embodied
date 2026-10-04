"""Task-agnostic camera-frame keyframes and continuous LIBERO action chunks."""
from __future__ import annotations

import math


AXES = ("x", "y", "z")
DIRECTIONS = {"negative", "hold", "positive", "unknown"}
PHASES = {"approach", "align", "contact", "manipulate", "release", "recover", "verify", "uncertain"}
GRIPPER = {"open", "hold", "close", "unknown"}
MAGNITUDES = {"fine", "medium", "coarse"}


def validate_keyframe(value):
    """Validate the closed VLM keyframe schema without task-specific skills."""
    fields = {"phase", "summary", "visible_evidence", "motion_frame", "translation_camera",
              "rotation_camera", "gripper", "magnitude", "chunk_horizon", "plan_horizon_decisions",
              "completion_evidence", "risk", "replan_condition"}
    if not isinstance(value, dict) or set(value) != fields or value["phase"] not in PHASES:
        raise ValueError("Local VLM returned an invalid keyframe schema")
    for field in ("summary", "visible_evidence", "completion_evidence", "risk", "replan_condition"):
        if not isinstance(value[field], str) or not 1 <= len(value[field]) <= 600:
            raise ValueError(f"Local VLM returned invalid {field}")
    if value["motion_frame"] not in {"external", "wrist"}:
        raise ValueError("Local VLM returned an unknown camera frame")
    for field in ("translation_camera", "rotation_camera"):
        if (not isinstance(value[field], dict) or set(value[field]) != set(AXES)
                or any(direction not in DIRECTIONS for direction in value[field].values())):
            raise ValueError(f"Local VLM returned invalid {field}")
    if value["gripper"] not in GRIPPER or value["magnitude"] not in MAGNITUDES:
        raise ValueError("Local VLM returned invalid gripper or magnitude")
    if type(value["chunk_horizon"]) is not int or not 2 <= value["chunk_horizon"] <= 10:
        raise ValueError("Local VLM returned an invalid chunk horizon")
    if type(value["plan_horizon_decisions"]) is not int or not 2 <= value["plan_horizon_decisions"] <= 8:
        raise ValueError("Local VLM returned an invalid plan horizon")
    return value


def _matmul(matrix, vector):
    if (not isinstance(matrix, list) or len(matrix) != 4
            or any(not isinstance(row, list) or len(row) != 4 for row in matrix)):
        raise ValueError("Expected a finite 4x4 camera-to-world transform")
    values = [float(matrix[row][column]) for row in range(3) for column in range(3)]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Expected a finite 4x4 camera-to-world transform")
    return [sum(float(matrix[row][column]) * vector[column] for column in range(3)) for row in range(3)]


def camera_directions_to_world(directions, camera_to_world, *, image_rotated_180=True):
    """Map directions in the exact image shown to the VLM into world XYZ.

    LIBERO RGB is rotated 180 degrees before both local policies inspect it. That
    flips visible image X/Y relative to the calibrated raw camera while retaining
    camera depth Z. Magnitudes are normalized after rotation so this function
    carries direction only; the action generator owns control effort.
    """
    signs = {"negative": -1., "hold": 0., "positive": 1., "unknown": 0.}
    vector = [signs[directions[axis]] for axis in AXES]
    if image_rotated_180:
        vector[0], vector[1] = -vector[0], -vector[1]
    world = _matmul(camera_to_world, vector)
    largest = max(map(abs, world), default=0.)
    return [value / largest for value in world] if largest > 1e-12 else [0., 0., 0.]


def _bounded_action(translation, rotation, gripper):
    action = [*translation, *rotation, float(gripper)]
    if len(action) != 7 or any(not math.isfinite(value) or abs(value) > 1 for value in action):
        raise AssertionError("Generated an invalid bounded LIBERO action")
    return action


def _profile(action, horizon):
    weights = (1., .86, .72, .58, .46, .36, .28, .22, .18, .15)
    return [[value * weights[index] if channel < 6 else value
             for channel, value in enumerate(action)] for index in range(horizon)]


def generate_action_chunks(keyframe, camera_to_world, previous_gripper, maximum_scale=.5):
    """Generate generic numeric Hx7 trajectories; no task or object names enter."""
    validate_keyframe(keyframe)
    if not -1 <= previous_gripper <= 1 or not 0 < maximum_scale <= .5:
        raise ValueError("Invalid gripper command or candidate scale")
    translation_direction = camera_directions_to_world(
        keyframe["translation_camera"], camera_to_world)
    plane_direction = camera_directions_to_world(
        {**keyframe["translation_camera"], "z": "hold"}, camera_to_world)
    depth_direction = camera_directions_to_world(
        {"x": "hold", "y": "hold", "z": keyframe["translation_camera"]["z"]},
        camera_to_world)
    rotation_direction = camera_directions_to_world(
        keyframe["rotation_camera"], camera_to_world)
    base = {"fine": .10, "medium": .22, "coarse": .38}[keyframe["magnitude"]]
    effort = min(maximum_scale, base)
    rotation_effort = min(maximum_scale, effort * .65)
    gripper = {"open": -1., "close": 1., "hold": previous_gripper,
               "unknown": previous_gripper}[keyframe["gripper"]]
    horizon = keyframe["chunk_horizon"]

    def action(t_multiplier, r_multiplier, finger=gripper, *, direction=None, reverse=False, lift=False):
        sign = -1. if reverse else 1.
        direction = translation_direction if direction is None else direction
        translation = [sign * min(maximum_scale, effort * t_multiplier) * value
                       for value in direction]
        if lift:
            translation[2] = max(translation[2], min(maximum_scale, .12))
        rotation = [sign * min(maximum_scale, rotation_effort * r_multiplier) * value
                    for value in rotation_direction]
        return _bounded_action(translation, rotation, finger)

    proposals = [
        ("hold", "Hold pose for a fresh observation.", _bounded_action([0.]*3, [0.]*3, previous_gripper)),
        ("cautious", "Half-effort motion along the visual keyframe error.", action(.50, .50)),
        ("nominal", "Nominal simultaneous 6D motion toward the visual keyframe.", action(1., 1.)),
        ("assertive", "Higher-effort bounded motion along the same keyframe direction.", action(1.30, 1.20)),
        ("translation", "Use the full camera-derived translation while holding orientation.", action(1., 0.)),
        ("image_plane", "Use only visible image-plane X/Y translation; hold uncertain camera depth.",
         action(1., 0., direction=plane_direction)),
        ("camera_depth", "Use only the proposed camera-depth translation; hold image-plane motion.",
         action(.65, 0., direction=depth_direction)),
        ("rotation", "Rotate toward the keyframe while holding translation.", action(0., 1.)),
        ("recover", "Reverse the proposed motion and restore upward clearance.", action(.55, .35, reverse=True, lift=True)),
        ("gripper", "Hold pose and apply only the proposed gripper command.",
         _bounded_action([0.]*3, [0.]*3, gripper)),
    ]
    result = {}
    seen = set()
    for family, description, first in proposals:
        chunk = _profile(first, horizon)
        signature = tuple(tuple(round(value, 10) for value in row) for row in chunk)
        if signature in seen:
            continue
        seen.add(signature)
        identity = f"chunk_{len(result):02d}"
        result[identity] = {"family": family, "description": description, "actions": chunk,
                            "frame": keyframe["motion_frame"], "horizon": horizon}
    if len(result) < 3:
        raise ValueError("Visual keyframe did not produce enough distinct action chunks")
    return result


def nominal_chunk(candidates):
    for identity, candidate in candidates.items():
        if candidate["family"] == "nominal":
            return identity
    for identity, candidate in candidates.items():
        if candidate["family"] == "cautious":
            return identity
    raise ValueError("No executable nominal action chunk")
