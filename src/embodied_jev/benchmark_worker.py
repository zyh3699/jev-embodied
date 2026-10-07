"""Standalone simulator worker. Run with the benchmark's Python, not this package's.

Only newline-delimited JSON crosses stdin/stdout. Simulator imports and logs are
isolated from the app and its incompatible MuJoCo dependency. No model runs here.
"""
from __future__ import annotations

import contextlib
import base64
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import sys


def libero_array_row_to_camera_row(row, height):
    """Map LIBERO's stored image row to robosuite's calibrated OpenCV row."""
    if height <= 0 or row < 0 or row >= height:
        raise ValueError("Image row is outside the camera frame")
    return height - 1 - row


def validate_action(action, size, bounded=True):
    import numpy as np
    value = np.asarray(action, dtype=float)
    if value.shape != (size,) or not np.isfinite(value).all() or bounded and (abs(value) > 1).any():
        suffix = " normalized action values in [-1, 1]" if bounded else " finite action values"
        raise ValueError(f"Expected {size}{suffix}")
    return value


def image_packet(array):
    """Keep transport explicit; policy profiles own orientation and resizing."""
    import numpy as np
    from PIL import Image
    array = np.asarray(array)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Camera must return RGB uint8 pixels")
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    data = buffer.getvalue()
    return {"encoding": "png", "data": base64.b64encode(data).decode("ascii"),
            "width": array.shape[1], "height": array.shape[0],
            "sha256": hashlib.sha256(data).hexdigest()}


def axis_angle(quaternion):
    """LIBERO/robosuite XYZW quaternion to shortest axis-angle vector."""
    import numpy as np
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError("Invalid proprioceptive quaternion")
    q = q / np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    length = np.linalg.norm(q[:3])
    return np.zeros(3) if length < 1e-8 else q[:3] * (2 * math.atan2(length, q[3]) / length)


def openpi_libero_axis_angle(quaternion):
    """Match openpi examples/libero/main.py rather than canonicalizing the sign."""
    import numpy as np
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("Invalid proprioceptive quaternion")
    w = float(np.clip(q[3], -1., 1.))
    denominator = math.sqrt(max(0., 1. - w * w))
    return np.zeros(3) if math.isclose(denominator, 0.) else q[:3] * (2. * math.acos(w) / denominator)


class MetaWorld:
    def __init__(self, case, horizon, observation_mode="privileged", control_mode="skills"):
        import gymnasium as gym
        import metaworld  # noqa: F401: registers official Gymnasium environments
        self.observation_mode = observation_mode
        self.control_mode = control_mode
        if control_mode not in {"skills", "hierarchical"} or control_mode == "hierarchical" and observation_mode != "privileged":
            raise ValueError("Meta-World hierarchy requires privileged observations")
        render = {"render_mode": "rgb_array", "camera_name": "corner2", "width": 256, "height": 256} if observation_mode == "vision" else {}
        self.env = gym.make("Meta-World/MT1", env_name=case["task"], seed=case["seed"], **render)
        self.case, self.horizon, self.steps = case, horizon, 0
        self.raw = None
        self.expert = None

    def observation(self):
        import numpy as np
        v = np.asarray(self.raw, dtype=float)
        if v.shape != (39,) or not np.isfinite(v).all():
            raise ValueError("Unsupported Meta-World observation schema; expected v3 39D state")
        if getattr(self, "observation_mode", "privileged") == "vision":
            return {"source": "rendered Meta-World RGB plus proprioception",
                    "task": self.case["task"], "tcp": v[:3].tolist(), "gripper_open_fraction": float(v[3])}
        result = {"source": "privileged Meta-World v3 state; no images",
                "task": self.case["task"], "units": "metres; world XYZ",
                "tcp": v[:3].tolist(), "gripper_open_fraction": float(v[3]),
                "object_slots": [{"position": v[a:a + 3].tolist(), "quaternion": v[a + 3:a + 7].tolist()}
                                 for a in (4, 11)],
                "previous_state": v[18:36].tolist(), "goal": v[36:39].tolist(),
                "note": "Object slots may be zero-padded. Goal is from the official goal-visible observation."}
        if getattr(self, "control_mode", "skills") == "hierarchical":
            env = self.env.unwrapped
            # Meta-World 3.1.1's drawer task still calls the removed
            # ``model.geom_name2id`` API, and the door task uses ``handle``
            # instead of the default ``objGeom``. Resolve both through the
            # supported named-data API so an observation never crashes.
            try:
                bilateral_contact = bool(env.touching_main_object)
            except (AttributeError, KeyError):
                geom_name = "handle" if self.case["task"].startswith("door-") else "objGeom"
                bilateral_contact = bool(env.touching_object(env.data.geom(geom_name).id))
            result["manipulation"] = {
                "hand_position_m": v[:3].tolist(),
                "finger_center_m": np.asarray(env.tcp_center, dtype=float).tolist(),
                "bilateral_contact": bilateral_contact,
                "translation_metres_per_unit": float(env.action_scale),
                "note": "Legacy tcp is the hand reference, not the finger center. Contact is measured on both pads; closed fingers alone do not prove a grasp.",
            }
        return result

    def policy_input(self):
        images = {"external": image_packet(self.env.render())} if self.observation_mode == "vision" else {}
        return {"prompt": self.case["task"].removesuffix("-v3").replace("-", " "),
                "state": self.raw[:4].tolist(), "images": images, "step": self.steps}

    def reset(self):
        self.raw, _ = self.env.reset(seed=self.case["seed"])
        dt = float(self.env.unwrapped.dt)
        return {"observation": self.observation(), "policy_input": self.policy_input(), "metadata": {
            "benchmark": "Meta-World", "version": importlib.metadata.version("metaworld"),
            "mujoco": importlib.metadata.version("mujoco"), "action_size": 4,
            "protocol": "individual MT1 goal-visible task; custom evaluation subset, not MT10/ML10",
            "success_source": "official info.success", "reset_seed": self.case["seed"],
            **({"hierarchy_observation": "hand-and-finger-center-plus-bilateral-contact-v1"}
               if self.control_mode == "hierarchical" else {}),
            "observation_mode": self.observation_mode, "camera_names": {"external": "corner2"} if self.observation_mode == "vision" else {},
            "action_spec": {"space": "metaworld_xyz", "size": 4, "normalized": True, "frame": "world",
                            "gripper": {"open": -1., "close": 1.}, "control_hz": 1 / dt}}}

    def step(self, action=None, scripted=False, capture=True):
        if scripted:
            from metaworld import policies
            # Use the installed official expert for plumbing validation only.
            name = "Sawyer" + "".join(p.capitalize() for p in self.case["task"].split("-")[:-1]) + "V3Policy"
            if self.expert is None:
                self.expert = getattr(policies, name)()
            # Some upstream experts return unbounded proportional effort;
            # normalize only this explicit comparator, never a model's action.
            import numpy as np
            action = np.clip(self.expert.get_action(self.raw), -1, 1)
        action = validate_action(action, 4)
        self.raw, _, terminated, truncated, info = self.env.step(action)
        self.steps += 1
        return {"observation": self.observation(), "success": bool(info["success"]),
                "terminated": bool(terminated), "truncated": bool(truncated or self.steps >= self.horizon),
                "action": action.tolist(),
                **({"policy_input": self.policy_input()} if capture or info["success"] else {})}

    def close(self):
        self.env.close()


class Libero:
    def __init__(self, case, horizon, observation_mode="privileged", policy_profile="default"):
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
        if policy_profile not in {"default", "openpi_libero"}:
            raise ValueError("Unknown LIBERO policy profile")
        self.case, self.steps, self.observation_mode, self.policy_profile = case, 0, observation_mode, policy_profile
        suite = benchmark.get_benchmark_dict()[case["suite"]](task_order_index=0)
        task_id = case["task_id"]
        if not 0 <= task_id < suite.get_num_tasks():
            raise ValueError("LIBERO task_id out of range")
        task = suite.get_task(task_id)
        self.language = task.language
        self.states = suite.get_task_init_states(task_id)
        if not 0 <= case["init_index"] < len(self.states):
            raise ValueError("LIBERO init_index out of range; never wrap initial-state indices")
        # Match openpi/examples/libero/main.py exactly at the environment boundary.
        # OffScreenRenderEnv owns the official OSC controller defaults; depth is
        # captured only for the grounded hybrid controller and never sent to pi0.5.
        self.env = OffScreenRenderEnv(bddl_file_name=suite.get_task_bddl_file_path(task_id),
                                      use_camera_obs=observation_mode == "vision",
                                      camera_names=["agentview", "robot0_eye_in_hand"],
                                      camera_heights=256, camera_widths=256,
                                      camera_depths=observation_mode == "vision", horizon=horizon)
        self.env.seed(case["seed"])
        self.raw = None

    def observation(self):
        import numpy as np
        if self.observation_mode == "vision":
            return {"source": "rendered LIBERO RGB plus proprioception", "task": self.language,
                    "tcp": np.asarray(self.raw["robot0_eef_pos"]).tolist(),
                    "gripper_qpos": np.asarray(self.raw["robot0_gripper_qpos"]).tolist()}
        # Explicit allowlist: evaluation predicates, reward and private simulator
        # attributes never become policy input. Object poses remain privileged.
        allowed = {k: np.asarray(v, dtype=float).tolist() for k, v in self.raw.items()
                   if k.endswith(("_pos", "_quat")) or k == "robot0_gripper_qpos"}
        return {"source": "privileged LIBERO named poses; no images", "task": self.language,
                "units": "metres; world frame; quaternions as emitted by LIBERO", "poses": allowed}

    def policy_input(self):
        import numpy as np
        from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix
        quaternion_transform = openpi_libero_axis_angle if self.policy_profile == "openpi_libero" else axis_angle
        state = np.concatenate([self.raw["robot0_eef_pos"], quaternion_transform(self.raw["robot0_eef_quat"]),
                                self.raw["robot0_gripper_qpos"]])
        images = {}
        if self.observation_mode == "vision":
            for view, key, camera in (("external", "agentview_image", "agentview"),
                                      ("wrist", "robot0_eye_in_hand_image", "robot0_eye_in_hand")):
                packet = image_packet(self.raw[key])
                packet["intrinsics"] = get_camera_intrinsic_matrix(
                    self.env.sim, camera, packet["height"], packet["width"]).tolist()
                packet["camera_to_world"] = get_camera_extrinsic_matrix(self.env.sim, camera).tolist()
                images[view] = packet
        return {"prompt": self.language, "state": state.tolist(), "images": images, "step": self.steps}

    def grounded_geometry(self, view, u, v, image_rotated_180=True, contact_mode="none",
                          support_profile="dynamic"):
        """Deproject a semantic VLM pixel and estimate its local surface normal."""
        import numpy as np
        from robosuite.utils.camera_utils import (get_camera_extrinsic_matrix,
                                                  get_camera_intrinsic_matrix,
                                                  get_real_depth_map)
        mapping = {"external": ("agentview_depth", "agentview"),
                   "wrist": ("robot0_eye_in_hand_depth", "robot0_eye_in_hand")}
        if view not in mapping or isinstance(u, bool) or isinstance(v, bool):
            raise ValueError("Invalid grounded-geometry camera or pixel")
        u, v = float(u), float(v)
        if not 0 <= u <= 1 or not 0 <= v <= 1:
            raise ValueError("Grounded-geometry pixels must be normalized")
        depth_key, camera = mapping[view]
        raw = np.asarray(self.raw[depth_key], dtype=float).squeeze()
        height, width = raw.shape
        model_column = int(round(u * (width - 1)))
        model_row = int(round(v * (height - 1)))
        if image_rotated_180:
            # RGB and depth are returned together in robosuite's OpenGL row
            # order. The policy rotates RGB by 180 degrees, so the metric depth
            # sample is at the opposite stored row and column. Calibration uses
            # OpenCV rows, however: OpenGL's vertical flip cancels the policy's
            # vertical flip, making the model row the calibrated camera row.
            sample_row = height - 1 - model_row
            sample_column = width - 1 - model_column
            camera_row = model_row
            camera_column = sample_column
        else:
            sample_row = model_row
            sample_column = model_column
            camera_row = libero_array_row_to_camera_row(sample_row, height)
            camera_column = sample_column
        metric = get_real_depth_map(self.env.sim, raw)
        intrinsic = get_camera_intrinsic_matrix(self.env.sim, camera, height, width)
        camera_to_world = get_camera_extrinsic_matrix(self.env.sim, camera)

        def point_at(depth_row, depth_column, calibrated_row, calibrated_column):
            depth_row = int(np.clip(depth_row, 0, height - 1))
            depth_column = int(np.clip(depth_column, 0, width - 1))
            calibrated_row = int(np.clip(calibrated_row, 0, height - 1))
            calibrated_column = int(np.clip(calibrated_column, 0, width - 1))
            radius = 2
            patch = metric[max(0, depth_row - radius):min(height, depth_row + radius + 1),
                           max(0, depth_column - radius):min(width, depth_column + radius + 1)]
            finite = patch[np.isfinite(patch) & (patch > 0)]
            if not finite.size:
                raise ValueError("No finite depth around the visual target")
            z = float(np.median(finite))
            camera_point = np.array([(calibrated_column - intrinsic[0, 2]) * z / intrinsic[0, 0],
                                     (calibrated_row - intrinsic[1, 2]) * z / intrinsic[1, 1], z, 1.])
            return (camera_to_world @ camera_point)[:3], z

        center, depth = point_at(sample_row, sample_column, camera_row, camera_column)
        semantic_center = center.copy()
        left, _ = point_at(sample_row, sample_column + 3, camera_row, camera_column - 3)
        right, _ = point_at(sample_row, sample_column - 3, camera_row, camera_column + 3)
        up, _ = point_at(sample_row + 3, sample_column, camera_row - 3, camera_column)
        down, _ = point_at(sample_row - 3, sample_column, camera_row + 3, camera_column)
        normal = np.cross(right - left, down - up)
        length = float(np.linalg.norm(normal))
        valid = bool(np.isfinite(center).all() and np.isfinite(normal).all() and length > 1e-8)
        if valid:
            normal /= length
            camera_position = camera_to_world[:3, 3]
            if float(np.dot(normal, camera_position - center)) < 0:
                normal = -normal
        semantic_normal = normal.copy() if valid else np.array([0., 0., 0.])
        # Estimate the nearest broad horizontal support below the semantic seed.
        # Peaks in world-Z density correspond to table, cabinet and shelf tops;
        # selecting the highest broad peak below the target avoids a fixed table
        # height while excluding the much smaller object's own top surface.
        raw_rows, raw_columns = np.indices((height, width))
        calibrated_rows = height - 1 - raw_rows
        camera_cloud = np.stack([
            (raw_columns - intrinsic[0, 2]) * metric / intrinsic[0, 0],
            (calibrated_rows - intrinsic[1, 2]) * metric / intrinsic[1, 1],
            metric, np.ones_like(metric)], axis=-1)
        world_cloud = camera_cloud @ camera_to_world.T
        workspace = (np.isfinite(world_cloud[..., 2])
                     & (np.abs(world_cloud[..., 0]) < .75)
                     & (np.abs(world_cloud[..., 1]) < .90)
                     & (world_cloud[..., 2] > -.15) & (world_cloud[..., 2] < 1.20))
        below = world_cloud[..., 2][workspace & (world_cloud[..., 2] < semantic_center[2] - .008)]
        if support_profile not in {"dynamic", "libero_tabletop"}:
            raise ValueError("Invalid support profile")
        support_height = 0. if support_profile == "libero_tabletop" else None
        if support_height is None and below.size:
            quantized = np.round(below / .005).astype(int)
            bins, counts = np.unique(quantized, return_counts=True)
            broad = bins[counts >= max(40, int(.001 * below.size))]
            if broad.size:
                support_height = float(broad.max() * .005)
        if support_height is None:
            support_height = float(np.percentile(below, 5)) if below.size else 0.
        grasp_refined = False
        push_refined = False
        object_center_world = None
        object_half_extent_world = None
        component_size = 0
        if contact_mode in {"grasp", "push"}:
            # The VLM owns semantic identity, while RGB-D owns precise contact.
            # Grow a depth-continuous component around the semantic seed and use
            # its upper world-space band for a top-down grasp point. This reads
            # neither simulator segmentation nor privileged object poses.
            radius = 36
            r0, r1 = max(0, sample_row - radius), min(height, sample_row + radius + 1)
            c0, c1 = max(0, sample_column - radius), min(width, sample_column + radius + 1)
            local = metric[r0:r1, c0:c1]
            seed = (sample_row - r0, sample_column - c0)
            visited = np.zeros(local.shape, dtype=bool)
            stack = [seed]
            pixels = []
            while stack:
                rr, cc = stack.pop()
                if visited[rr, cc]:
                    continue
                visited[rr, cc] = True
                value = local[rr, cc]
                if not np.isfinite(value) or value <= 0 or abs(float(value) - depth) > .12:
                    continue
                pixels.append((rr + r0, cc + c0, float(value)))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < local.shape[0] and 0 <= nc < local.shape[1] and not visited[nr, nc]:
                        neighbor = local[nr, nc]
                        if np.isfinite(neighbor) and abs(float(neighbor) - float(value)) <= .018:
                            stack.append((nr, nc))
            component_size = len(pixels)
            if component_size >= 12:
                rr = np.asarray([item[0] for item in pixels], dtype=float)
                cc = np.asarray([item[1] for item in pixels], dtype=float)
                zz = np.asarray([item[2] for item in pixels], dtype=float)
                calibrated_rr = height - 1 - rr
                camera_component = np.stack([
                    (cc - intrinsic[0, 2]) * zz / intrinsic[0, 0],
                    (calibrated_rr - intrinsic[1, 2]) * zz / intrinsic[1, 1],
                    zz, np.ones_like(zz)], axis=-1)
                world_component = camera_component @ camera_to_world.T
                # A depth-continuous flood can leak from an object's lower
                # silhouette onto the table.  Split it again after projection:
                # retain pixels physically above the support plane, then keep
                # only the image-connected island containing the semantic seed.
                minimum_object_height = .003 if contact_mode == "push" else .010
                above = world_component[:, 2] > support_height + minimum_object_height
                lookup = {(int(row), int(column)): index
                          for index, (row, column) in enumerate(zip(rr, cc)) if above[index]}
                seed_key = (sample_row, sample_column)
                object_indices = []
                if seed_key in lookup:
                    pending, object_seen = [seed_key], set()
                    while pending:
                        key = pending.pop()
                        if key in object_seen or key not in lookup:
                            continue
                        object_seen.add(key)
                        object_indices.append(lookup[key])
                        row, column = key
                        pending.extend(((row - 1, column), (row + 1, column),
                                        (row, column - 1), (row, column + 1)))
                object_points = world_component[object_indices, :3]
                component_size = len(object_points)
                cutoff = (float(np.percentile(object_points[:, 2], 70))
                          if component_size >= 12 else float("inf"))
                upper = object_points[object_points[:, 2] >= cutoff]
                if len(upper) >= 3:
                    object_top = float(np.percentile(upper[:, 2], 65))
                    lower_bounds = np.percentile(object_points, 5, axis=0)
                    upper_bounds = np.percentile(object_points, 95, axis=0)
                    object_center_world = np.median(object_points, axis=0)
                    object_half_extent_world = .5 * (upper_bounds - lower_bounds)
                    # OSC controls the EEF grip site between the fingers, not
                    # the fingertips. Keep semantic-seed XY and place the grip
                    # site near the object's vertical center, rather than on
                    # the top surface where the fingers would close in air.
                    # The semantic pixel is deliberately allowed to land on a
                    # distinctive part of the label.  It is therefore a poor
                    # XY grasp target (and can sit over a centimetre away from
                    # the cylinder axis).  The high world-Z band is the object
                    # cap for top-down tabletop views; its robust XY median is
                    # invariant to which part of the label the VLM selected.
                    # This uses only RGB-D geometry, never simulator object
                    # poses or segmentation.
                    center = semantic_center.copy()
                    if contact_mode == "grasp":
                        center[:2] = np.median(upper[:, :2], axis=0)
                    elif contact_mode == "push":
                        center[:2] = object_center_world[:2]
                    center[2] = (support_height + .50 * (object_top - support_height)
                                 if support_height is not None else
                                 (float(semantic_center[2]) + object_top) / 2)
                    normal = np.array([0., 0., 1.])
                    valid = bool(np.isfinite(center).all())
                    grasp_refined = valid and contact_mode == "grasp"
                    push_refined = valid and contact_mode == "push"
        return {"valid": valid, "view": view, "pixel_rotated_normalized": [u, v],
                "raw_pixel_row_column": [sample_row, sample_column],
                "camera_pixel_row_column": [camera_row, camera_column], "depth_m": depth,
                "point_world": center.tolist(),
                "semantic_seed_point_world": semantic_center.tolist(),
                "semantic_seed_normal_toward_camera_world": semantic_normal.tolist(),
                "normal_toward_camera_world": normal.tolist() if valid else [0., 0., 0.],
                "dominant_support_height_m": support_height,
                "height_above_support_m": (float(center[2]) - support_height
                                             if support_height is not None else None),
                "grasp_geometry_refined": grasp_refined,
                "push_geometry_refined": push_refined,
                "object_center_world": (object_center_world.tolist()
                                         if object_center_world is not None else None),
                "object_half_extent_world": (object_half_extent_world.tolist()
                                              if object_half_extent_world is not None else None),
                "semantic_component_pixels": component_size,
                "source": ("semantic RGB-D seed plus depth-connected object-center refinement"
                           if grasp_refined else
                           "semantic trailing-side seed plus depth-connected object mid-height refinement"
                           if push_refined else
                           "metric RGB-D deprojection and local depth-plane normal")}

    def grounded_proposals(self, view="external", image_rotated_180=True):
        """Return RGB-D surface components without simulator segmentation."""
        import numpy as np
        from robosuite.utils.camera_utils import (get_camera_extrinsic_matrix,
                                                  get_camera_intrinsic_matrix,
                                                  get_real_depth_map)
        if view != "external" or not image_rotated_180:
            raise ValueError("Invalid proposal camera")
        raw = np.asarray(self.raw["agentview_depth"], dtype=float).squeeze()
        height, width = raw.shape
        metric = get_real_depth_map(self.env.sim, raw)
        intrinsic = get_camera_intrinsic_matrix(self.env.sim, "agentview", height, width)
        camera_to_world = get_camera_extrinsic_matrix(self.env.sim, "agentview")
        raw_rows, columns = np.indices((height, width))
        camera_rows = height - 1 - raw_rows
        camera_points = np.stack([
            (columns - intrinsic[0, 2]) * metric / intrinsic[0, 0],
            (camera_rows - intrinsic[1, 2]) * metric / intrinsic[1, 1],
            metric, np.ones_like(metric)], axis=-1)
        world = camera_points @ camera_to_world.T
        # Minimal assisted-grounding assumption: LIBERO tabletop object tasks
        # use the calibrated z=0 support and a bounded central workspace. This
        # endpoint is only used by the opt-in appearance-reference fallback;
        # the default direct-pixel controller remains support-height agnostic.
        # Keeping robot/background surfaces outside this proposal set is what
        # makes a task-blind appearance match meaningful.
        support_height = 0.
        mask = ((world[..., 2] > support_height + .012) & (world[..., 2] < .30)
                & (np.abs(world[..., 0]) < .40) & (np.abs(world[..., 1]) < .40)
                & np.isfinite(metric) & (metric > 0))
        visited = np.zeros(mask.shape, dtype=bool)
        components = []
        for start_row, start_column in zip(*np.nonzero(mask)):
            if visited[start_row, start_column]:
                continue
            visited[start_row, start_column] = True
            stack = [(int(start_row), int(start_column))]
            pixels = []
            while stack:
                row, column = stack.pop()
                pixels.append((row, column))
                depth = float(metric[row, column])
                for nr, nc in ((row - 1, column), (row + 1, column),
                               (row, column - 1), (row, column + 1)):
                    if (0 <= nr < height and 0 <= nc < width and mask[nr, nc]
                            and not visited[nr, nc]
                            and abs(float(metric[nr, nc]) - depth) <= .025):
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            if len(pixels) >= 8:
                components.append(pixels)

        # Larger coherent surfaces are the most useful semantic candidates.
        # Keep a bounded list so point labels never obscure the source image.
        components.sort(key=len, reverse=True)
        proposals = []
        for pixels in components[:24]:
            rr = np.asarray([pixel[0] for pixel in pixels], dtype=int)
            cc = np.asarray([pixel[1] for pixel in pixels], dtype=int)
            model_rr = height - 1 - rr
            model_cc = width - 1 - cc
            center_row, center_column = np.median(model_rr), np.median(model_cc)
            index = int(np.argmin((model_rr - center_row) ** 2 + (model_cc - center_column) ** 2))
            u = float(model_cc[index] / (width - 1))
            v = float(model_rr[index] / (height - 1))
            proposals.append({"id": "region_%02d" % len(proposals), "view": view,
                              "u": u, "v": v,
                              "bbox_uv": [float(model_cc.min() / (width - 1)),
                                          float(model_rr.min() / (height - 1)),
                                          float(model_cc.max() / (width - 1)),
                                          float(model_rr.max() / (height - 1))],
                              "pixels": len(pixels), "candidate_kind": "visible_surface",
                              "height_m": round(float(world[rr[index], cc[index], 2]
                                                      - support_height), 4)})
        # A large U-shaped depth component has useful wall pixels but its median
        # surface point is a poor drop target. Add an upper-middle interior probe
        # for large components. This is geometric and carries no object identity.
        for proposal in list(proposals):
            left, top, right, bottom = proposal["bbox_uv"]
            if (proposal["pixels"] < 1000 or right - left < .16 or bottom - top < .16):
                continue
            component = components[int(proposal["id"].split("_")[-1])]
            component_rows = np.asarray([pixel[0] for pixel in component], dtype=int)
            component_columns = np.asarray([pixel[1] for pixel in component], dtype=int)
            component_world = world[component_rows, component_columns, :3]
            x_bounds = np.percentile(component_world[:, 0], [2, 98])
            y_bounds = np.percentile(component_world[:, 1], [2, 98])
            wall_top = float(np.percentile(component_world[:, 2], 95))
            component_floor = float(np.percentile(component_world[:, 2], 3))
            interior_world = [float(x_bounds.mean()), float(y_bounds.mean()),
                              float(min(component_floor + .25,
                                        max(component_floor + .08, wall_top + .025)))]
            u = (left + right) / 2
            v = top + .22 * (bottom - top)
            try:
                geometry = self.grounded_geometry(view, u, v, image_rotated_180, "none")
            except ValueError:
                continue
            proposals.append({"id": "region_%02d" % len(proposals), "view": view,
                              "u": u, "v": v,
                              "bbox_uv": [u - .045, v - .035, u + .045, v + .035],
                              "pixels": None, "candidate_kind": "geometric_interior_probe",
                              "_interior_point_world": interior_world,
                              "height_m": round(float(geometry["height_above_support_m"]), 4)})
        return {"proposals": proposals, "support_height_m": support_height,
                "source": "metric-depth-continuous surface components; no segmentation or object pose"}

    def appearance_reference(self):
        """Resolve a static public asset texture from the language noun phrase.

        This is an appearance memory only: it never reads scene bodies, poses,
        segmentation, contacts, predicates, or success state.
        """
        import numpy as np
        from PIL import Image
        import libero.libero as libero_package
        root = Path(libero_package.__file__).resolve().parent / "assets" / "stable_hope_objects"
        language = " ".join(self.language.lower().replace("_", " ").split())
        matches = []
        if root.is_dir():
            for directory in root.iterdir():
                texture = directory / "texture_map.png"
                phrase = directory.name.lower().replace("_", " ")
                if texture.is_file() and phrase in language:
                    matches.append((len(phrase), phrase, texture))
        if not matches:
            return {"available": False, "source": "no matching public static appearance asset"}
        _, phrase, texture = max(matches)
        pixels = np.asarray(Image.open(texture).convert("RGB"), dtype=np.uint8)
        return {"available": True, "target_noun_phrase": phrase,
                "image": image_packet(pixels),
                "source": "public LIBERO static asset texture; appearance only; no scene state"}

    def reset(self):
        self.env.reset()
        self.raw = self.env.set_init_state(self.states[self.case["init_index"]])
        return {"observation": self.observation(), "policy_input": self.policy_input(), "metadata": {
            "benchmark": "LIBERO", "suite": self.case["suite"], "action_size": 7,
            "robosuite": importlib.metadata.version("robosuite"),
            "init_index": self.case["init_index"], "reset_seed": self.case["seed"],
            "controller": "OSC_POSE, 20 Hz, normalized XYZ + axis-angle + gripper",
            "protocol": "custom rollout using official initial state and success predicate",
            "observation_mode": self.observation_mode,
            "camera_names": {"external": "agentview", "wrist": "robot0_eye_in_hand"} if self.observation_mode == "vision" else {},
            "camera_orientation": "raw MuJoCo output; policy profile specifies transforms",
            "camera_calibration": "known intrinsics and camera-to-world extrinsics; no scene depth or object pose",
            "policy_profile": self.policy_profile,
            "action_spec": {"space": "libero_osc_pose", "size": 7, "normalized": True, "frame": "world",
                            "gripper": {"open": -1., "close": 1.}, "control_hz": 20.},
            "success_source": "official env.check_success()"}}

    def step(self, action=None, scripted=False, capture=True, allow_unbounded=False):
        if scripted:
            raise ValueError("No LIBERO scripted baseline is supplied")
        action = validate_action(action, 7, bounded=not allow_unbounded)
        self.raw, _, done, _ = self.env.step(action)
        self.steps += 1
        success = bool(self.env.check_success())
        return {"observation": self.observation(), "success": success,
                "terminated": False, "truncated": bool(done), "action": action.tolist(),
                **({"policy_input": self.policy_input()} if capture or success else {})}

    def close(self):
        self.env.close()


def main():
    backend = None
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                with contextlib.redirect_stdout(sys.stderr):
                    if request["command"] == "reset":
                        if backend is not None:
                            backend.close()
                        cls = {"metaworld": MetaWorld, "libero": Libero}[request["backend"]]
                        mode = request.get("observation_mode", "privileged")
                        if mode not in {"privileged", "vision"}:
                            raise ValueError("Worker supports privileged or vision observations")
                        control = request.get("control_mode", "skills")
                        if control != "skills" and request["backend"] != "metaworld":
                            raise ValueError("External hierarchy is currently Meta-World only")
                        extra = ({"control_mode": control} if request["backend"] == "metaworld" else
                                 {"policy_profile": request.get("policy_profile", "default")})
                        backend = cls(request["case"], request["horizon"], mode, **extra)
                        result = backend.reset()
                    elif request["command"] == "step" and backend is not None:
                        extra = ({"allow_unbounded": request.get("allow_unbounded", False)}
                                 if isinstance(backend, Libero) else {})
                        result = backend.step(request.get("action"), request.get("scripted", False),
                                              request.get("capture", True), **extra)
                    elif request["command"] == "grounded_geometry" and isinstance(backend, Libero):
                        result = {"geometry": backend.grounded_geometry(
                            request.get("view"), request.get("u"), request.get("v"),
                            request.get("image_rotated_180", True), request.get("contact_mode", "none"),
                            request.get("support_profile", "dynamic"))}
                    elif request["command"] == "grounded_proposals" and isinstance(backend, Libero):
                        result = backend.grounded_proposals(
                            request.get("view", "external"), request.get("image_rotated_180", True))
                    elif request["command"] == "appearance_reference" and isinstance(backend, Libero):
                        result = backend.appearance_reference()
                    elif request["command"] == "close":
                        break
                    else:
                        raise ValueError("Invalid worker command or missing reset")
                payload = {"ok": True, **result}
            except Exception as exc:
                import traceback
                traceback.print_exc(file=sys.stderr)
                payload = {"ok": False, "error": type(exc).__name__, "message": str(exc)}
            print(json.dumps(payload, allow_nan=False), flush=True)
    finally:
        if backend is not None:
            with contextlib.redirect_stdout(sys.stderr):
                backend.close()


if __name__ == "__main__":
    main()
