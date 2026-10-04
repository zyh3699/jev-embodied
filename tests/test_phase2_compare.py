import unittest
from types import SimpleNamespace

from embodied_jev.benchmark_worker import openpi_libero_axis_angle, validate_action
from embodied_jev.phase2_compare import (MODES, action_candidates, confidence_metrics,
    confidence_triggers, stalled, validate_action_chunk, validate_vlm_analysis)


class Phase2ContractTests(unittest.TestCase):
    def test_candidate_set_is_21_unique_bounded_actions(self):
        candidates = action_candidates(.5, -1.)
        self.assertEqual(len(candidates), 21)
        actions = [tuple(item["action"]) for item in candidates.values()]
        self.assertEqual(len(set(actions)), 21)
        self.assertTrue(all(len(action) == 7 and max(map(abs, action)) <= 1 for action in actions))

    def test_pi05_chunk_matches_official_unclipped_actions(self):
        with self.assertRaises(ValueError):
            validate_action_chunk([[0, 0, 0, 0, 0, 0, float("nan")]])
        self.assertEqual(validate_action_chunk([[0, 0, 0, 0, 0, 0, 1.01]])[0][-1], 1.01)
        with self.assertRaises(ValueError):
            validate_action([0, 0, 0, 0, 0, 0, 1.01], 7)
        self.assertEqual(validate_action([0, 0, 0, 0, 0, 0, 1.01], 7, bounded=False)[-1], 1.01)

    def test_vlm_schema_is_closed(self):
        candidates = action_candidates(.5, -1.)
        value = {"phase": "align", "summary": "hand near handle", "visible_evidence": "handle visible",
                 "translation": {"x": "unknown", "y": "positive", "z": "hold"},
                 "rotation": {"rx": "hold", "ry": "unknown", "rz": "negative"},
                 "gripper": "open", "risk": "depth is uncertain",
                 "candidate_actions": ["hold", "translate_y_positive_small", "gripper_open"],
                 "plan_horizon_decisions": 3, "replan_condition": "handle leaves view"}
        self.assertIs(validate_vlm_analysis(value, candidates), value)
        with self.assertRaises(ValueError):
            validate_vlm_analysis({**value, "success": True}, candidates)
        with self.assertRaises(ValueError):
            validate_vlm_analysis({**value, "candidate_actions": ["hold", "invented", "gripper_open"]}, candidates)
        with self.assertRaises(ValueError):
            validate_vlm_analysis({**value, "plan_horizon_decisions": 1}, candidates)

    def test_trigger_metrics_are_explicit_and_deterministic(self):
        metrics = confidence_metrics({"a": .34, "b": .33, "c": .33}, "a")
        args = SimpleNamespace(jev_confidence_threshold=.35, jev_margin_threshold=.08,
                               jev_entropy_threshold=.9)
        self.assertEqual(confidence_triggers(metrics, args),
                         ["low_selected_probability", "low_max_probability",
                          "low_top_two_margin", "high_normalized_entropy"])
        self.assertTrue(stalled([{"state_delta_l2": .0001}] * 3, 3, .001))
        self.assertFalse(stalled([{"state_delta_l2": .01}] * 3, 3, .001))

    def test_phase2_has_direct_triggered_and_dense_modes(self):
        self.assertEqual(MODES, ("pi05", "vlm-jev-triggered", "vlm-jev-dense"))

    def test_openpi_quaternion_profile_does_not_flip_sign(self):
        positive = openpi_libero_axis_angle([0., 0., .5, .8660254038])
        negative = openpi_libero_axis_angle([0., 0., -.5, -.8660254038])
        self.assertAlmostEqual(positive[2], 1.0471975512, places=6)
        self.assertGreater(abs(negative[2]), 5.)


if __name__ == "__main__":
    unittest.main()
