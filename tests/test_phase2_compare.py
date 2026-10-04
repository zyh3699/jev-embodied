import unittest
from types import SimpleNamespace

from embodied_jev.benchmark_worker import openpi_libero_axis_angle, validate_action
from embodied_jev.phase2_compare import (MODES, confidence_metrics, confidence_triggers,
    expand_skill, select_with_jev, skill_catalog, stalled, validate_action_chunk,
    validate_vlm_analysis)


class Phase2ContractTests(unittest.TestCase):
    def test_macro_skills_expand_to_bounded_concurrent_actions(self):
        candidates = skill_catalog()
        self.assertEqual(len(candidates), 10)
        plan = {"translation": {"x": "negative", "y": "positive", "z": "negative"},
                "rotation": {"rx": "hold", "ry": "positive", "rz": "negative"},
                "gripper": "close"}
        for name in candidates:
            actions = expand_skill(name, plan, .5, -1., 5)
            self.assertTrue(1 <= len(actions) <= 5)
            self.assertTrue(all(len(action) == 7 and max(map(abs, action)) <= 1 for action in actions))
        approach = expand_skill("approach_coarse", plan, .5, -1., 5)[0]
        self.assertGreater(sum(value != 0 for value in approach[:6]), 1)
        self.assertEqual(expand_skill("manipulate_firm", plan, .5, -1., 3)[0][-1], 1.)
        self.assertGreater(expand_skill("retract_recover", plan, .5, -1., 5)[0][2], 0)

    def test_pi05_chunk_matches_official_unclipped_actions(self):
        with self.assertRaises(ValueError):
            validate_action_chunk([[0, 0, 0, 0, 0, 0, float("nan")]])
        self.assertEqual(validate_action_chunk([[0, 0, 0, 0, 0, 0, 1.01]])[0][-1], 1.01)
        with self.assertRaises(ValueError):
            validate_action([0, 0, 0, 0, 0, 0, 1.01], 7)
        self.assertEqual(validate_action([0, 0, 0, 0, 0, 0, 1.01], 7, bounded=False)[-1], 1.01)

    def test_vlm_schema_is_closed(self):
        candidates = skill_catalog()
        value = {"phase": "align", "summary": "hand near handle", "visible_evidence": "handle visible",
                 "translation": {"x": "unknown", "y": "positive", "z": "hold"},
                 "rotation": {"rx": "hold", "ry": "unknown", "rz": "negative"},
                 "gripper": "open", "risk": "depth is uncertain",
                 "candidate_skills": ["observe_hold", "approach_fine", "align_pose"],
                 "plan_horizon_decisions": 3, "replan_condition": "handle leaves view"}
        self.assertIs(validate_vlm_analysis(value, candidates), value)
        with self.assertRaises(ValueError):
            validate_vlm_analysis({**value, "success": True}, candidates)
        with self.assertRaises(ValueError):
            validate_vlm_analysis({**value, "candidate_skills": ["observe_hold", "invented", "align_pose"]}, candidates)
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

    def test_jev_selects_one_expanded_skill(self):
        class Client:
            def request(self, stage, state, *, questions):
                self.stage, self.state, self.questions = stage, state, questions
                options = questions["skill"]["criteria"]
                probabilities = {name: 0. for name in options}
                probabilities["approach_fine"] = 1.
                return {"skill": {"choice": "approach_fine", "probabilities": probabilities}}

        client = Client()
        skills = {
            "observe_hold": {"description": "wait", "actions": [[0.] * 6 + [-1.]]},
            "approach_fine": {"description": "move", "actions": [[.1, .1, 0., 0., 0., 0., -1.]]},
            "retract_recover": {"description": "recover", "actions": [[-.1, -.1, .1, 0., 0., 0., -1.]]},
        }
        selected, probabilities, state = select_with_jev(
            client, "close drawer", [0.] * 8, {"phase": "approach"}, skills, [])
        self.assertEqual(selected, "approach_fine")
        self.assertEqual(probabilities[selected], 1.)
        self.assertEqual(client.stage, "jev_skill_selection")
        self.assertEqual(state["candidate_skill_count"], 3)

    def test_phase2_has_direct_triggered_and_dense_modes(self):
        self.assertEqual(MODES, ("pi05", "vlm-jev-triggered", "vlm-jev-dense"))

    def test_openpi_quaternion_profile_does_not_flip_sign(self):
        positive = openpi_libero_axis_angle([0., 0., .5, .8660254038])
        negative = openpi_libero_axis_angle([0., 0., -.5, -.8660254038])
        self.assertAlmostEqual(positive[2], 1.0471975512, places=6)
        self.assertGreater(abs(negative[2]), 5.)


if __name__ == "__main__":
    unittest.main()
