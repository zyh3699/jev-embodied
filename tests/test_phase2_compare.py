import unittest
from types import SimpleNamespace

from embodied_jev.benchmark_worker import openpi_libero_axis_angle, validate_action
from embodied_jev.phase2_chunks import (camera_directions_to_world, generate_action_chunks,
    nominal_chunk, validate_keyframe)
from embodied_jev.phase2_compare import (MODES, confidence_metrics, confidence_triggers,
    select_with_jev, stalled, validate_action_chunk)


class Phase2ContractTests(unittest.TestCase):
    def keyframe(self):
        return {"phase": "align", "summary": "hand near handle", "visible_evidence": "handle visible",
                "motion_frame": "external",
                "translation_camera": {"x": "negative", "y": "positive", "z": "negative"},
                "rotation_camera": {"x": "hold", "y": "positive", "z": "negative"},
                "gripper": "close", "magnitude": "medium", "chunk_horizon": 5,
                "plan_horizon_decisions": 3, "completion_evidence": "hand aligned to handle",
                "risk": "depth is uncertain", "replan_condition": "handle leaves view"}

    def test_keyframe_generates_bounded_concurrent_action_chunks(self):
        transform = [[1., 0., 0., 0.], [0., 1., 0., 0.],
                     [0., 0., 1., 0.], [0., 0., 0., 1.]]
        candidates = generate_action_chunks(self.keyframe(), transform, -1., .5)
        self.assertGreaterEqual(len(candidates), 6)
        self.assertEqual(candidates[nominal_chunk(candidates)]["family"], "nominal")
        for candidate in candidates.values():
            self.assertEqual(len(candidate["actions"]), 5)
            self.assertTrue(all(len(action) == 7 and max(map(abs, action)) <= 1
                                for action in candidate["actions"]))
        nominal = candidates[nominal_chunk(candidates)]["actions"][0]
        self.assertGreater(sum(value != 0 for value in nominal[:6]), 1)
        self.assertEqual(nominal[-1], 1.)
        self.assertEqual(camera_directions_to_world(
            {"x": "positive", "y": "hold", "z": "hold"}, transform), [-1., 0., 0.])

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
            validate_keyframe({**value, "motion_frame": "world"})

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

    def test_phase2_has_direct_triggered_and_dense_modes(self):
        self.assertEqual(MODES, ("pi05", "vlm-jev-triggered", "vlm-jev-dense", "vlm-chunk-no-jev"))

    def test_openpi_quaternion_profile_does_not_flip_sign(self):
        positive = openpi_libero_axis_angle([0., 0., .5, .8660254038])
        negative = openpi_libero_axis_angle([0., 0., -.5, -.8660254038])
        self.assertAlmostEqual(positive[2], 1.0471975512, places=6)
        self.assertGreater(abs(negative[2]), 5.)


if __name__ == "__main__":
    unittest.main()
