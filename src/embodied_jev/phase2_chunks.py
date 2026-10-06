"""Task-agnostic camera-frame keyframes and continuous LIBERO action chunks."""
from __future__ import annotations

import math


AXES = ("x", "y", "z")
DIRECTIONS = {"negative", "hold", "positive", "unknown"}
PHASES = {"approach", "align", "contact", "manipulate", "release", "recover", "verify", "uncertain"}
GRIPPER = {"open", "hold", "close", "unknown"}
MAGNITUDES = {"fine", "medium", "coarse"}
CONTACT_MODES = {"push", "pull", "rotate", "grasp", "release", "none"}
GOAL_RELATIONS = {"none", "inside", "on", "at"}
MOTION_HINTS = {"auto", "normal_in", "normal_out", "image_left", "image_right",
                "image_up", "image_down", "image_up_left", "image_up_right",
                "image_down_left", "image_down_right", "clockwise", "counterclockwise"}


def validate_keyframe(value):
    """Validate a task-agnostic, pixel-grounded contact keyframe."""
    fields = {"phase", "summary", "visible_evidence", "target_identity", "distractor_check",
              "target", "contact_mode", "goal_relation", "motion_hint", "gripper",
              "magnitude", "chunk_horizon", "plan_horizon_decisions", "completion_evidence",
              "risk", "replan_condition"}
    if not isinstance(value, dict) or set(value) != fields or value["phase"] not in PHASES:
        raise ValueError("Local VLM returned an invalid keyframe schema")
    for field in ("summary", "visible_evidence", "target_identity", "distractor_check",
                  "completion_evidence", "risk", "replan_condition"):
        if not isinstance(value[field], str) or not 1 <= len(value[field]) <= 600:
            raise ValueError(f"Local VLM returned invalid {field}")
    target = value["target"]
    if (not isinstance(target, dict)
            or set(target) != {"view", "proposal_id", "u", "v", "confidence"}
            or target["view"] not in {"external", "wrist"}):
        raise ValueError("Local VLM returned an invalid visual target")
    if not (target["proposal_id"] is None or isinstance(target["proposal_id"], str)):
        raise ValueError("Local VLM returned an invalid proposal id")
    for field in ("u", "v", "confidence"):
        if (isinstance(target[field], bool) or not isinstance(target[field], (int, float))
                or not math.isfinite(float(target[field])) or not 0 <= float(target[field]) <= 1):
            raise ValueError(f"Local VLM returned invalid target {field}")
    if value["contact_mode"] not in CONTACT_MODES:
        raise ValueError("Local VLM returned an invalid contact mode")
    if value["goal_relation"] not in GOAL_RELATIONS:
        raise ValueError("Local VLM returned an invalid goal relation")
    if value["motion_hint"] not in MOTION_HINTS:
        raise ValueError("Local VLM returned an invalid motion hint")
    if value["contact_mode"] != "release" and value["goal_relation"] != "none":
        raise ValueError("Only release keyframes may specify a goal relation")
    if value["contact_mode"] == "release" and value["goal_relation"] == "none":
        raise ValueError("Release keyframes require an explicit goal relation")
    if value["gripper"] not in GRIPPER or value["magnitude"] not in MAGNITUDES:
        raise ValueError("Local VLM returned invalid gripper or magnitude")
    if type(value["chunk_horizon"]) is not int or not 2 <= value["chunk_horizon"] <= 10:
        raise ValueError("Local VLM returned an invalid chunk horizon")
    if type(value["plan_horizon_decisions"]) is not int or not 2 <= value["plan_horizon_decisions"] <= 40:
        raise ValueError("Local VLM returned an invalid plan horizon")
    return value


def generate_grounded_action_chunks(keyframe, geometry, proprioception, previous_gripper,
                                    maximum_scale=.5, previous_controller_phase=None,
                                    previous_aperture=None):
    """Generate generic Hx7 chunks from an RGB-D semantic contact point.

    The VLM supplies semantics and a normalized pixel only. The worker deprojects
    that pixel with a robot-mounted RGB-D camera; no task id, object pose, or
    success predicate enters this controller.
    """
    validate_keyframe(keyframe)
    if not -1 <= previous_gripper <= 1 or not 0 < maximum_scale <= .5:
        raise ValueError("Invalid gripper command or candidate scale")
    if (not isinstance(proprioception, (list, tuple)) or len(proprioception) < 3
            or not geometry.get("valid")):
        raise ValueError("Grounded chunks require valid TCP and RGB-D geometry")
    tcp = [float(value) for value in proprioception[:3]]
    gripper_aperture = (abs(float(proprioception[-2]) - float(proprioception[-1]))
                        if len(proprioception) >= 8 else 0.)
    target = [float(value) for value in geometry["point_world"]]
    normal = [float(value) for value in geometry["normal_toward_camera_world"]]
    if len(target) != 3 or len(normal) != 3 or any(not math.isfinite(x) for x in target + normal):
        raise ValueError("Grounded geometry must contain finite 3D vectors")
    norm = math.sqrt(sum(value * value for value in normal))
    if norm < 1e-8:
        raise ValueError("Grounded surface normal is degenerate")
    normal = [value / norm for value in normal]
    contact_mode = keyframe["contact_mode"]
    clearance = .065 if keyframe["magnitude"] != "fine" else .045
    approach_offset = (normal if contact_mode in {"push", "pull"} else [0., 0., 1.])
    precontact = [target[i] + clearance * approach_offset[i] for i in range(3)]
    distance_to_target = math.sqrt(sum((target[i] - tcp[i]) ** 2 for i in range(3)))
    horizontal_error = math.sqrt(sum((target[i] - tcp[i]) ** 2 for i in range(2)))
    vertical_gap = tcp[2] - target[2]
    distance_to_precontact = math.sqrt(sum((precontact[i] - tcp[i]) ** 2 for i in range(3)))
    # Once near the interaction surface, manipulate along its normal. Grasp and
    # release also need a generic contact transition; closing the fingers alone
    # is not evidence that a grasp has completed.
    contact_ready = distance_to_target < .028
    contact_established = (contact_mode == "push" or previous_gripper > .5)
    continued_contact = previous_controller_phase in {"surface_manipulation",
                                                       "rotational_manipulation"}
    manipulating = (contact_mode in {"push", "pull", "rotate"}
                    and contact_established and (contact_ready or continued_contact))
    controller_phase = (("rotational_manipulation" if contact_mode == "rotate" else
                         "surface_manipulation") if manipulating else
                        ("contact_close" if contact_mode in {"pull", "rotate"}
                         and contact_ready else "approach"))
    destination = (target if contact_mode == "release" or distance_to_precontact < .045
                   else precontact)
    approach = [destination[i] - tcp[i] for i in range(3)]
    learned_direction = geometry.get("manipulation_direction_world")
    if (not isinstance(learned_direction, list) or len(learned_direction) != 3
            or any(not math.isfinite(float(value)) for value in learned_direction)):
        learned_direction = None
    if learned_direction is not None:
        learned_norm = math.sqrt(sum(float(value) ** 2 for value in learned_direction))
        learned_direction = ([float(value) / learned_norm for value in learned_direction]
                             if learned_norm > 1e-8 else None)
    default_manipulation = ([-value for value in normal] if contact_mode == "push" else list(normal))
    manipulation_direction = learned_direction or default_manipulation
    motion = (manipulation_direction if manipulating and contact_mode != "rotate" else
              ([0., 0., 0.] if manipulating else
               ([target[i] - tcp[i] for i in range(3)]
                if controller_phase == "contact_close" else approach)))
    # LIBERO's robot0_eef_pos is the grip site between the fingers.  It should
    # meet the object's mid-height contact point directly; adding a fingertip
    # length here closes the fingers above the object.  The small positive gap
    # keeps the grip site just above the RGB-D center while allowing compliance.
    in_grasp_envelope = horizontal_error < .025 and -.005 < vertical_gap < .035
    # The MuJoCo grip site is between the fingers. On many geometries the palm
    # or fingertips make first contact while that site is still 2--4 cm above
    # the RGB-D object center. Requiring millimetre-perfect convergence causes
    # an endless downward command against a physically blocking object. Close
    # anywhere inside the same generic collision-aware grasp envelope.
    ready_to_close = horizontal_error < .025 and -.005 < vertical_gap < .040
    if contact_mode == "grasp" and previous_gripper > .5 and in_grasp_envelope:
        if (previous_controller_phase in {"contact_close", "grasp_squeeze"}
                and previous_aperture is not None
                and abs(gripper_aperture - float(previous_aperture)) > .003):
            controller_phase = "grasp_squeeze"
            motion = [0., 0., 0.]
        elif gripper_aperture > .008:
            controller_phase = "post_grasp_lift"
            motion = [0., 0., 1.]
        else:
            controller_phase = "failed_grasp_reopen"
            motion = [0., 0., 1.]
    elif contact_mode == "grasp" and ready_to_close:
        controller_phase = "contact_close"
    elif contact_mode == "grasp":
        if horizontal_error >= .020:
            controller_phase = "grasp_xy_transit"
            destination = [target[0], target[1], max(tcp[2], target[2] + .14)]
        else:
            controller_phase = "grasp_descent"
            destination = [target[0], target[1], target[2] + .006]
        motion = [destination[i] - tcp[i] for i in range(3)]
    elif contact_mode == "release":
        if distance_to_target < .020:
            controller_phase = "contact_release"
            motion = [0., 0., 0.]
        elif horizontal_error >= .050 and tcp[2] < target[2] + .12:
            controller_phase = "release_lift_clearance"
            destination = [tcp[0], tcp[1], target[2] + .14]
            motion = [destination[i] - tcp[i] for i in range(3)]
        elif horizontal_error >= .025:
            controller_phase = "release_xy_transit"
            destination = [target[0], target[1], max(tcp[2], target[2] + .12)]
            motion = [destination[i] - tcp[i] for i in range(3)]
        else:
            controller_phase = "release_descent"
            destination = target
            motion = [destination[i] - tcp[i] for i in range(3)]
    largest = max(map(abs, motion), default=0.)
    direction = [value / largest for value in motion] if largest > 1e-8 else [0., 0., 0.]
    base = {"fine": .10, "medium": .22, "coarse": .36}[keyframe["magnitude"]]
    if controller_phase == "post_grasp_lift":
        base = .06
    elif controller_phase == "failed_grasp_reopen":
        base = .22
    elif controller_phase == "grasp_squeeze":
        base = .03
    elif contact_mode == "release" and controller_phase == "approach":
        base = (.50 if distance_to_target > .14 else
                (.34 if distance_to_target > .06 else .18))
    elif not manipulating:
        base = .50 if distance_to_precontact > .14 else (.34 if distance_to_precontact > .06 else .18)
    effort = min(maximum_scale, base)
    finger = {"open": -1., "close": 1., "hold": previous_gripper,
              "unknown": previous_gripper}[keyframe["gripper"]]
    if contact_mode in {"push", "rotate"}:
        finger = 1.  # a compact closed tool is safer and more repeatable for pushing
    elif contact_mode == "pull":
        finger = 1. if controller_phase in {"contact_close", "surface_manipulation"} else -1.
    elif contact_mode == "grasp":
        finger = 1. if controller_phase in {"contact_close", "grasp_squeeze", "post_grasp_lift"} else -1.
    elif contact_mode == "release":
        finger = -1. if controller_phase == "contact_release" else previous_gripper
    horizon = keyframe["chunk_horizon"]

    def bounded(vector, multiplier=1., gripper=finger):
        translation = [max(-maximum_scale, min(maximum_scale, effort * multiplier * value))
                       for value in vector]
        return _bounded_action(translation, [0., 0., 0.], gripper)

    tangent_a = [normal[1], -normal[0], 0.]
    tangent_norm = math.sqrt(sum(value * value for value in tangent_a))
    tangent_a = ([value / tangent_norm for value in tangent_a]
                 if tangent_norm > 1e-8 else [1., 0., 0.])
    normal_motion = manipulation_direction
    if contact_mode == "none":
        controller_phase = "observe"
        proposals = [
            ("nominal", "Hold pose for a fresh observation.",
             _bounded_action([0.] * 3, [0.] * 3, previous_gripper)),
            ("observe_lift", "Add a small upward clearance for a less occluded observation.",
             _bounded_action([0., 0., .08], [0.] * 3, previous_gripper)),
            ("observe_parallax", "Add a small lateral parallax motion for a less occluded observation.",
             _bounded_action([.08, 0., 0.], [0.] * 3, previous_gripper)),
        ]
    elif controller_phase == "contact_release":
        proposals = [
            ("nominal", "Hold at the destination while opening the fingers.", bounded([0., 0., 0.], 1.)),
            ("release_lift", "Open and add slight upward clearance.", bounded([0., 0., 1.], .4)),
            ("release_retreat", "Open and retreat along the measured surface normal.", bounded(normal, .35)),
        ]
    elif controller_phase == "grasp_squeeze":
        proposals = [
            ("nominal", "Hold the TCP still while allowing the fingers to finish closing.",
             bounded([0., 0., 0.], 1.)),
            ("micro_lower", "Maintain closure with a tiny downward seating motion.",
             bounded([0., 0., -1.], .25)),
            ("micro_lift", "Maintain closure with a tiny upward tension motion.",
             bounded([0., 0., 1.], .25)),
        ]
    elif controller_phase == "rotational_manipulation":
        sign = -1. if keyframe["motion_hint"] == "clockwise" else 1.
        rotation_axis = normal
        inward = [-.10 * value for value in normal]
        rotation_effort = min(maximum_scale, .28)

        def rotational(axis, multiplier=1.):
            rotation = [max(-maximum_scale, min(maximum_scale,
                        sign * rotation_effort * multiplier * value)) for value in axis]
            return _bounded_action(inward, rotation, finger)

        tangent_b = [normal[2], 0., -normal[0]]
        tangent_b_norm = math.sqrt(sum(value * value for value in tangent_b))
        tangent_b = ([value / tangent_b_norm for value in tangent_b]
                     if tangent_b_norm > 1e-8 else [0., 1., 0.])
        proposals = [
            ("cautious", "Rotate cautiously about the measured surface-normal axis.",
             rotational(rotation_axis, .55)),
            ("nominal", "Rotate about the measured surface-normal axis while maintaining contact.",
             rotational(rotation_axis, 1.)),
            ("reverse_rotation", "Rotate in the opposite direction about the same axis.",
             _bounded_action(inward, [-value for value in rotational(rotation_axis, 1.)[3:6]], finger)),
            ("alternate_axis", "Rotate about an orthogonal local axis if the mechanism axis is ambiguous.",
             rotational(tangent_b, .75)),
        ]
    else:
        proposals = [
            ("cautious", "Three-quarter-effort progress toward the deprojected contact geometry.", bounded(direction, .75)),
            ("moderate", "Nine-tenths-effort progress toward the same grounded target.", bounded(direction, .9)),
            ("nominal", "Nominal progress toward the deprojected contact geometry.", bounded(direction, 1.)),
            ("assertive", "Higher-effort bounded progress toward the same grounded target.", bounded(direction, 1.3)),
        ]
    if manipulating and controller_phase != "rotational_manipulation":
        proposals.extend([
            ("surface_normal", "Manipulate along the locally measured surface normal.", bounded(normal_motion, 1.)),
            ("surface_normal_cautious", "Manipulate cautiously along the measured surface normal.", bounded(normal_motion, .55)),
            ("tangent_probe", "Small tangential probe if the contact normal is locally ambiguous.", bounded(tangent_a, .35)),
        ])
    elif controller_phase == "approach":
        proposals.extend([
            ("direct_contact", "Approach the measured contact point without the clearance offset.",
             bounded([value / max(max(map(abs, [target[i] - tcp[i] for i in range(3)])), 1e-8)
                      for value in [target[i] - tcp[i] for i in range(3)]], .65)),
            ("retreat", "Retreat from the surface along its measured outward normal.", bounded(normal, .45)),
        ])
        if contact_mode in {"pull", "rotate"}:
            rotation_effort = min(maximum_scale, .18)
            for axis_index, axis_name in enumerate(("roll", "pitch", "yaw")):
                for sign, label in ((1., "positive"), (-1., "negative")):
                    rotation = [0., 0., 0.]
                    rotation[axis_index] = sign * rotation_effort
                    proposals.append((f"orientation_probe_{axis_name}_{label}",
                                      f"Hold translation and probe {label} {axis_name} to resolve a side-contact pose.",
                                      _bounded_action([0., 0., 0.], rotation, finger)))
    if ((contact_mode == "grasp" and controller_phase == "contact_close")
            or (contact_mode == "release" and controller_phase == "contact_release")):
        proposals.append(("gripper", "Hold Cartesian pose and apply only the requested gripper command.",
                          _bounded_action([0.] * 3, [0.] * 3, finger)))
    result, seen = {}, set()
    for family, description, first in proposals:
        if controller_phase == "post_grasp_lift":
            settle_steps = min(2, horizon - 1)
            hold = _bounded_action([0.] * 3, [0.] * 3, finger)
            chunk = [hold] * settle_steps + _profile(first, horizon - settle_steps)
        else:
            chunk = _profile(first, horizon)
        signature = tuple(tuple(round(value, 10) for value in row) for row in chunk)
        if signature in seen:
            continue
        seen.add(signature)
        identity = f"chunk_{len(result):02d}"
        result[identity] = {"family": family, "description": description, "actions": chunk,
                            "frame": "world", "horizon": horizon,
                            "geometry": {"distance_to_target_m": distance_to_target,
                                         "horizontal_error_m": horizontal_error,
                                         "vertical_gap_m": vertical_gap,
                                         "distance_to_precontact_m": distance_to_precontact,
                                         "manipulating": manipulating,
                                         "controller_phase": controller_phase,
                                         "gripper_aperture_m": gripper_aperture,
                                         "depth_m": float(geometry["depth_m"]),
                                         "target_confidence": float(keyframe["target"]["confidence"])}}
    if len(result) < 3:
        raise ValueError("Grounded keyframe did not produce enough distinct action chunks")
    return result


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
