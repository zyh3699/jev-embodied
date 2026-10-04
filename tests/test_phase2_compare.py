import unittest

from embodied_jev.benchmark_worker import openpi_libero_axis_angle, validate_action
from embodied_jev.phase2_compare import action_candidates, validate_action_chunk, validate_vlm_analysis


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
        value = {"phase": "align", "summary": "hand near handle", "visible_evidence": "handle visible",
                 "translation": {"x": "unknown", "y": "positive", "z": "hold"},
                 "rotation": {"rx": "hold", "ry": "unknown", "rz": "negative"},
                 "gripper": "open", "risk": "depth is uncertain"}
        self.assertIs(validate_vlm_analysis(value), value)
        with self.assertRaises(ValueError):
            validate_vlm_analysis({**value, "success": True})

    def test_openpi_quaternion_profile_does_not_flip_sign(self):
        positive = openpi_libero_axis_angle([0., 0., .5, .8660254038])
        negative = openpi_libero_axis_angle([0., 0., -.5, -.8660254038])
        self.assertAlmostEqual(positive[2], 1.0471975512, places=6)
        self.assertGreater(abs(negative[2]), 5.)


if __name__ == "__main__":
    unittest.main()
