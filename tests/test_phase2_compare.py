import unittest
from types import SimpleNamespace

from embodied_jev.benchmark_worker import (libero_array_row_to_camera_row,
    openpi_libero_axis_angle, validate_action)
from embodied_jev.phase2_chunks import (generate_grounded_action_chunks, nominal_chunk,
    validate_keyframe)
from embodied_jev.phase2_compare import (MODES, confidence_metrics, confidence_triggers,
    select_with_jev, stalled, validate_action_chunk)


class Phase2ContractTests(unittest.TestCase):
    def keyframe(self):
        return {"phase": "align", "summary": "hand near handle", "visible_evidence": "handle visible",
                "target_identity": "drawer handle", "distractor_check": "not the cabinet edge",
                "target": {"view": "external", "proposal_id": "region_02",
                           "u": .35, "v": .62, "confidence": .9},
                "contact_mode": "push", "goal_relation": "none", "motion_hint": "auto",
                "gripper": "close", "magnitude": "medium", "chunk_horizon": 5,
                "plan_horizon_decisions": 2, "completion_evidence": "hand aligned to handle",
                "risk": "depth is uncertain", "replan_condition": "handle leaves view"}

    def test_keyframe_generates_bounded_grounded_action_chunks(self):
        geometry = {"valid": True, "point_world": [.1, .2, .9],
                    "normal_toward_camera_world": [0., -1., 0.], "depth_m": .7}
        candidates = generate_grounded_action_chunks(
            self.keyframe(), geometry, [0., 0., 1.1] + [0.] * 5, -1., .5)
        self.assertGreaterEqual(len(candidates), 5)
        self.assertEqual(candidates[nominal_chunk(candidates)]["family"], "nominal")
        for candidate in candidates.values():
            self.assertEqual(len(candidate["actions"]), 5)
            self.assertTrue(all(len(action) == 7 and max(map(abs, action)) <= 1
                                for action in candidate["actions"]))
        nominal = candidates[nominal_chunk(candidates)]["actions"][0]
        self.assertGreater(sum(value != 0 for value in nominal[:3]), 1)
        self.assertEqual(nominal[-1], 1.)

    def test_grounded_grasp_closes_then_lifts_without_reopening(self):
        keyframe = {**self.keyframe(), "contact_mode": "grasp", "gripper": "open"}
        geometry = {"valid": True, "point_world": [0., 0., .02],
                    "normal_toward_camera_world": [0., 0., 1.], "depth_m": .2}
        close = generate_grounded_action_chunks(
            keyframe, geometry, [0., 0., .028, 0., 0., 0., .04, -.04], -1., .5)
        close_nominal = close[nominal_chunk(close)]
        self.assertEqual(close_nominal["geometry"]["controller_phase"], "contact_close")
        self.assertEqual(close_nominal["actions"][0][-1], 1.)
        lift = generate_grounded_action_chunks(
            keyframe, geometry, [0., 0., .028, 0., 0., 0., .02, -.02], 1., .5)
        lift_nominal = lift[nominal_chunk(lift)]
        self.assertEqual(lift_nominal["geometry"]["controller_phase"], "post_grasp_lift")
        self.assertEqual(lift_nominal["actions"][0][2], 0.)
        self.assertTrue(any(action[2] > 0. for action in lift_nominal["actions"]))
        self.assertTrue(all(action[-1] == 1. for action in lift_nominal["actions"]))

    def test_empty_grasp_reopens_instead_of_claiming_lift(self):
        keyframe = {**self.keyframe(), "contact_mode": "grasp", "gripper": "open"}
        geometry = {"valid": True, "point_world": [0., 0., .02],
                    "normal_toward_camera_world": [0., 0., 1.], "depth_m": .2}
        empty = generate_grounded_action_chunks(
            keyframe, geometry, [0., 0., .028, 0., 0., 0., .0005, -.0005], 1., .5)
        result = empty[nominal_chunk(empty)]
        self.assertEqual(result["geometry"]["controller_phase"], "failed_grasp_reopen")
        self.assertGreater(result["actions"][0][2], 0.)
        self.assertEqual(result["actions"][0][-1], -1.)

    def test_grasp_waits_for_aperture_to_stabilize(self):
        keyframe = {**self.keyframe(), "contact_mode": "grasp", "gripper": "open"}
        geometry = {"valid": True, "point_world": [0., 0., .02],
                    "normal_toward_camera_world": [0., 0., 1.], "depth_m": .2}
        squeeze = generate_grounded_action_chunks(
            keyframe, geometry, [0., 0., .028, 0., 0., 0., .025, -.025], 1., .5,
            previous_controller_phase="contact_close", previous_aperture=.08)
        result = squeeze[nominal_chunk(squeeze)]
        self.assertEqual(result["geometry"]["controller_phase"], "grasp_squeeze")
        self.assertTrue(all(action[:3] == [0., 0., 0.] for action in result["actions"]))
        self.assertTrue(all(action[-1] == 1. for action in result["actions"]))

    def test_grasp_uses_xy_then_vertical_hierarchy(self):
        keyframe = {**self.keyframe(), "contact_mode": "grasp", "gripper": "open"}
        geometry = {"valid": True, "point_world": [0., 0., .02],
                    "normal_toward_camera_world": [0., 0., 1.], "depth_m": .2}
        xy = generate_grounded_action_chunks(
            keyframe, geometry, [.10, .10, .20, 0., 0., 0., .04, -.04], -1., .5)
        xy_nominal = xy[nominal_chunk(xy)]
        self.assertEqual(xy_nominal["geometry"]["controller_phase"], "grasp_xy_transit")
        self.assertEqual(xy_nominal["actions"][0][2], 0.)
        descent = generate_grounded_action_chunks(
            keyframe, geometry, [.005, .005, .20, 0., 0., 0., .04, -.04], -1., .5)
        descent_nominal = descent[nominal_chunk(descent)]
        self.assertEqual(descent_nominal["geometry"]["controller_phase"], "grasp_descent")
        self.assertLess(descent_nominal["actions"][0][2], 0.)

    def test_release_keeps_hold_until_contact_radius(self):
        keyframe = {**self.keyframe(), "phase": "release", "contact_mode": "release",
                    "goal_relation": "on", "gripper": "hold"}
        geometry = {"valid": True, "point_world": [0., 0., .10],
                    "normal_toward_camera_world": [0., 0., 1.], "depth_m": .5}
        approach = generate_grounded_action_chunks(
            keyframe, geometry, [0., 0., .175, 0., 0., 0., .02, -.02], 1., .5)
        approach_nominal = approach[nominal_chunk(approach)]
        self.assertEqual(approach_nominal["geometry"]["controller_phase"], "release_descent")
        self.assertEqual(approach_nominal["actions"][0][-1], 1.)
        release = generate_grounded_action_chunks(
            keyframe, geometry, [0., 0., .115, 0., 0., 0., .02, -.02], 1., .5)
        release_nominal = release[nominal_chunk(release)]
        self.assertEqual(release_nominal["geometry"]["controller_phase"], "contact_release")
        self.assertEqual(release_nominal["actions"][0][-1], -1.)

    def test_release_uses_lift_translate_descend_hierarchy(self):
        keyframe = {**self.keyframe(), "phase": "release", "contact_mode": "release",
                    "goal_relation": "on", "gripper": "hold"}
        geometry = {"valid": True, "point_world": [0., 0., .10],
                    "normal_toward_camera_world": [0., 0., 1.], "depth_m": .5}
        lift = generate_grounded_action_chunks(
            keyframe, geometry, [.20, .20, .15, 0., 0., 0., .02, -.02], 1., .5)
        lift_nominal = lift[nominal_chunk(lift)]
        self.assertEqual(lift_nominal["geometry"]["controller_phase"], "release_lift_clearance")
        self.assertEqual(lift_nominal["actions"][0][:2], [0., 0.])
        self.assertGreater(lift_nominal["actions"][0][2], 0.)
        translate = generate_grounded_action_chunks(
            keyframe, geometry, [.20, .20, .30, 0., 0., 0., .02, -.02], 1., .5)
        translate_nominal = translate[nominal_chunk(translate)]
        self.assertEqual(translate_nominal["geometry"]["controller_phase"], "release_xy_transit")
        self.assertTrue(all(value < 0. for value in translate_nominal["actions"][0][:2]))
        descend = generate_grounded_action_chunks(
            keyframe, geometry, [.01, .01, .30, 0., 0., 0., .02, -.02], 1., .5)
        descend_nominal = descend[nominal_chunk(descend)]
        self.assertEqual(descend_nominal["geometry"]["controller_phase"], "release_descent")
        self.assertLess(descend_nominal["actions"][0][2], 0.)

    def test_pi05_chunk_matches_official_unclipped_actions(self):
        with self.assertRaises(ValueError):
            validate_action_chunk([[0, 0, 0, 0, 0, 0, float("nan")]])
        self.assertEqual(validate_action_chunk([[0, 0, 0, 0, 0, 0, 1.01]])[0][-1], 1.01)
        with self.assertRaises(ValueError):
            validate_action([0, 0, 0, 0, 0, 0, 1.01], 7)
        self.assertEqual(validate_action([0, 0, 0, 0, 0, 0, 1.01], 7, bounded=False)[-1], 1.01)

    def test_vlm_schema_is_closed(self):
        value = self.keyframe()
        self.assertIs(validate_keyframe(value), value)
        with self.assertRaises(ValueError):
            validate_keyframe({**value, "success": True})
        with self.assertRaises(ValueError):
            validate_keyframe({**value, "target": {**value["target"], "u": 1.1}})
        with self.assertRaises(ValueError):
            validate_keyframe({**value, "goal_relation": "inside"})
        with self.assertRaises(ValueError):
            validate_keyframe({**value, "contact_mode": "release", "goal_relation": "none"})
        self.assertEqual(validate_keyframe({**value, "motion_hint": "image_up_left"})["motion_hint"],
                         "image_up_left")

    def test_push_uses_task_general_image_direction_when_grounded(self):
        geometry = {"valid": True, "point_world": [0., 0., .10],
                    "normal_toward_camera_world": [0., 0., 1.], "depth_m": .5,
                    "manipulation_direction_world": [1., 0., 0.]}
        candidates = generate_grounded_action_chunks(
            {**self.keyframe(), "motion_hint": "image_right"}, geometry,
            [0., 0., .11, 0., 0., 0., .04, -.04], -1., .5)
        action = candidates[nominal_chunk(candidates)]["actions"][0]
        self.assertGreater(action[0], 0.)
        self.assertAlmostEqual(action[1], 0.)
        self.assertAlmostEqual(action[2], 0.)

    def test_surface_manipulation_persists_after_leaving_contact_origin(self):
        geometry = {"valid": True, "point_world": [0., 0., .10],
                    "normal_toward_camera_world": [0., 0., 1.], "depth_m": .5,
                    "manipulation_direction_world": [1., 0., 0.]}
        candidates = generate_grounded_action_chunks(
            {**self.keyframe(), "motion_hint": "image_right"}, geometry,
            [.12, 0., .10, 0., 0., 0., .04, -.04], 1., .5,
            previous_controller_phase="surface_manipulation")
        nominal = candidates[nominal_chunk(candidates)]
        self.assertEqual(nominal["geometry"]["controller_phase"], "surface_manipulation")
        self.assertGreater(nominal["actions"][0][0], 0.)

    def test_rotate_generates_contact_preserving_rotation_chunks(self):
        geometry = {"valid": True, "point_world": [0., 0., .10],
                    "normal_toward_camera_world": [0., -1., 0.], "depth_m": .5}
        contact = generate_grounded_action_chunks(
            {**self.keyframe(), "contact_mode": "rotate", "motion_hint": "clockwise"},
            geometry, [0., 0., .11, 0., 0., 0., .04, -.04], -1., .5)
        self.assertEqual(contact[nominal_chunk(contact)]["geometry"]["controller_phase"],
                         "contact_close")
        candidates = generate_grounded_action_chunks(
            {**self.keyframe(), "contact_mode": "rotate", "motion_hint": "clockwise"},
            geometry, [0., 0., .11, 0., 0., 0., .04, -.04], 1., .5,
            previous_controller_phase="contact_close")
        nominal = candidates[nominal_chunk(candidates)]
        self.assertEqual(nominal["geometry"]["controller_phase"], "rotational_manipulation")
        self.assertTrue(any(abs(value) > 0 for value in nominal["actions"][0][3:6]))

    def test_none_contact_mode_holds_for_observation(self):
        geometry = {"valid": True, "point_world": [0., 0., .10],
                    "normal_toward_camera_world": [0., 0., 1.], "depth_m": .5}
        candidates = generate_grounded_action_chunks(
            {**self.keyframe(), "contact_mode": "none", "gripper": "hold"},
            geometry, [0., 0., .30, 0., 0., 0., .04, -.04], -1., .5)
        nominal = candidates[nominal_chunk(candidates)]
        self.assertEqual(nominal["geometry"]["controller_phase"], "observe")
        self.assertEqual(nominal["actions"][0][:6], [0.] * 6)

    def test_side_contact_approach_offers_orientation_probes(self):
        geometry = {"valid": True, "point_world": [0., 0., .10],
                    "normal_toward_camera_world": [0., -1., 0.], "depth_m": .5}
        candidates = generate_grounded_action_chunks(
            {**self.keyframe(), "contact_mode": "pull", "motion_hint": "normal_out",
             "gripper": "open"}, geometry,
            [.20, .20, .30, 0., 0., 0., .04, -.04], -1., .5)
        probes = [candidate for candidate in candidates.values()
                  if candidate["family"].startswith("orientation_probe_")]
        self.assertEqual(len(probes), 6)
        self.assertTrue(all(any(abs(value) > 0 for value in candidate["actions"][0][3:6])
                            for candidate in probes))

    def test_grasp_closes_when_contact_blocks_exact_center(self):
        geometry = {"valid": True, "point_world": [0., 0., .10],
                    "normal_toward_camera_world": [0., 0., 1.], "depth_m": .5}
        candidates = generate_grounded_action_chunks(
            {**self.keyframe(), "contact_mode": "grasp", "gripper": "open"}, geometry,
            [.020, 0., .132, 0., 0., 0., .04, -.04], -1., .5)
        nominal = candidates[nominal_chunk(candidates)]
        self.assertEqual(nominal["geometry"]["controller_phase"], "contact_close")
        self.assertEqual(nominal["actions"][0][-1], 1.)

    def test_trigger_metrics_are_explicit_and_deterministic(self):
        metrics = confidence_metrics({"a": .34, "b": .33, "c": .33}, "a")
        args = SimpleNamespace(jev_confidence_threshold=.35, jev_margin_threshold=.08,
                               jev_entropy_threshold=.9)
        self.assertEqual(confidence_triggers(metrics, args),
                         ["low_selected_probability", "low_max_probability",
                          "low_top_two_margin", "high_normalized_entropy"])
        self.assertTrue(stalled([{"state_delta_l2": .0001}] * 3, 3, .001))
        self.assertFalse(stalled([{"state_delta_l2": .01}] * 3, 3, .001))

    def test_jev_selects_one_numeric_action_chunk(self):
        class Client:
            def request(self, stage, state, *, questions):
                self.stage, self.state, self.questions = stage, state, questions
                options = questions["action_chunk"]["criteria"]
                probabilities = {name: 0. for name in options}
                probabilities["chunk_01"] = 1.
                return {"action_chunk": {"choice": "chunk_01", "probabilities": probabilities}}

        client = Client()
        chunks = {
            "chunk_00": {"family": "hold", "description": "wait", "actions": [[0.] * 6 + [-1.]]},
            "chunk_01": {"family": "nominal", "description": "move", "actions": [[.1, .1, 0., 0., 0., 0., -1.]]},
            "chunk_02": {"family": "recover", "description": "recover", "actions": [[-.1, -.1, .1, 0., 0., 0., -1.]]},
        }
        selected, probabilities, state = select_with_jev(
            client, "close drawer", [0.] * 8, {"phase": "approach"}, chunks, [], .01)
        self.assertEqual(selected, "chunk_01")
        self.assertEqual(probabilities[selected], 1.)
        self.assertEqual(client.stage, "jev_action_chunk_selection")
        self.assertEqual(state["candidate_chunk_count"], 3)
        self.assertEqual(state["recent_candidate_effects"], [])

    def test_phase2_has_direct_triggered_and_dense_modes(self):
        self.assertEqual(MODES, ("pi05", "vlm-jev-triggered", "vlm-jev-dense", "vlm-chunk-no-jev"))

    def test_openpi_quaternion_profile_does_not_flip_sign(self):
        positive = openpi_libero_axis_angle([0., 0., .5, .8660254038])
        negative = openpi_libero_axis_angle([0., 0., -.5, -.8660254038])
        self.assertAlmostEqual(positive[2], 1.0471975512, places=6)
        self.assertGreater(abs(negative[2]), 5.)

    def test_libero_depth_rows_use_camera_calibration_convention(self):
        self.assertEqual(libero_array_row_to_camera_row(0, 256), 255)
        self.assertEqual(libero_array_row_to_camera_row(107, 256), 148)
        with self.assertRaises(ValueError):
            libero_array_row_to_camera_row(256, 256)


if __name__ == "__main__":
    unittest.main()
