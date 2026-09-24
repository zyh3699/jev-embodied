"""Sensor rights, geometry, selected-action fidelity, budgets and paired starts."""
import base64
import hashlib
import json
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from embodied_jev.benchmark_worker import image_packet
from embodied_jev.libero_compare import verify_pair
from embodied_jev.libero_policy import (ModelClient, RequestBudget, annotated_image, backproject,
    local_state, motor_action, motor_questions, orientation_error, quaternion_matrix,
    resolve_plan, rotation_matrix, usage_summary)
from embodied_jev.libero_worker import VisionEnvironment
from embodied_jev.policies import DecisionPolicy


def camera():
    depth = np.full((128, 128), .5, dtype="<f4").tobytes()
    return {**image_packet(np.zeros((128, 128, 3), dtype=np.uint8)),
            "depth_encoding": "float32-le-metres", "depth": base64.b64encode(depth).decode(),
            "depth_sha256": hashlib.sha256(depth).hexdigest(),
            "intrinsics": [[100, 0, 64], [0, 100, 64], [0, 0, 1]], "camera_to_world": np.eye(4).tolist()}


def observation():
    return {"task": "put the bowl on the plate", "tcp": [.1, .1, .6],
            "quaternion_xyzw": [0, 0, 0, 1], "gripper_qpos": [.04, -.04], "step": 0}


def answer():
    return {"stage": "approach", "intent": "Approach above the visible bowl", "visual_evidence": "Bowl at center",
            "target": {"kind": "pixel", "camera": "external", "pixel": [64, 64], "offset_m": [0, 0, .1]},
            "rotation_delta": [0, 0, 0], "gripper": "open", "max_motor_steps": 6}


def test_depth_projection_is_calibrated_and_uses_selected_pixel():
    packet = camera()
    xyz, source = backproject(packet, [84, 44])
    np.testing.assert_allclose(xyz, [.1, -.1, .5])
    assert source["depth_m"] == .5
    packet["camera_to_world"][0][3] = .3
    np.testing.assert_allclose(backproject(packet, [84, 44])[0], [.4, -.1, .5])
    packet["depth"] = base64.b64encode(bytes(128 * 128 * 4)).decode()
    with pytest.raises(ValueError, match="hash"):
        backproject(packet, [64, 64])


@pytest.mark.parametrize("pixel", [[-1, 20], [128, 20], [20, float("nan")]])
def test_invalid_pixels_cannot_be_clipped_into_another_target(pixel):
    with pytest.raises(ValueError):
        backproject(camera(), pixel)


def test_waypoint_resolves_measured_depth_with_explicit_offset():
    plan = resolve_plan(answer(), observation(), {"external": camera()})
    np.testing.assert_allclose(plan["target_xyz"], [0, 0, .6])
    assert plan["grounding"]["observation_step"] == 0
    assert plan["grounding"]["surface_xyz_m"] == [0, 0, .5]
    assert annotated_image(camera(), observation()["tcp"]).startswith(b"\x89PNG")


def test_selected_wrong_sign_is_executed_without_rule_correction():
    plan = resolve_plan(answer(), observation(), {"external": camera()})
    state = local_state(observation(), plan, -1, [])
    assert state["position_error_m"][0] < 0
    choices = dict(x="positive", y="negative", z="hold", rx="hold", ry="hold", rz="hold", gripper="close")
    action = motor_action(choices, state)
    assert action[0] > 0 and action[1] < 0 and action[2] == 0 and action[-1] == 1
    assert max(map(abs, action[:6])) <= .5
    questions = motor_questions(state)
    assert "current=" in questions["x"]["instructions"] and "target=" in questions["x"]["instructions"]
    assert "signed" in questions["x"]["instructions"] and "0.005" in questions["x"]["instructions"]


def test_rotation_has_world_frame_sign_and_quaternion_sign_invariance():
    current = quaternion_matrix([0, 0, 0, -1])
    np.testing.assert_allclose(current, np.eye(3))
    np.testing.assert_allclose(orientation_error(rotation_matrix([0, 0, .2]), current), [0, 0, np.sin(.2)])


def test_visual_observation_cannot_leak_hidden_objects_or_success():
    env = object.__new__(VisionEnvironment)
    env.language, env.steps = "put the bowl on the plate", 4
    env.raw = {"robot0_eef_pos": np.array([.1, .2, .3]), "robot0_eef_quat": np.array([0, 0, 0, 1]),
               "robot0_gripper_qpos": np.array([.04, -.04]), "bowl_pos": object(), "success": object(), "goal_pos": object()}
    state = env.observation()
    assert set(state) == {"task", "step", "tcp", "quaternion_xyzw", "gripper_qpos", "frame"}


def test_pair_validation_requires_actual_settled_state_and_engine_match():
    meta = {key: "same" for key in ("initial_state_sha256", "settled_state_sha256", "versions", "libero_revision", "case",
                "camera_size", "camera_transform", "controller", "action_output_min", "action_output_max")}
    verify_pair(meta, dict(meta))
    with pytest.raises(ValueError, match="settled_state"):
        verify_pair(meta, {**meta, "settled_state_sha256": "different"})


def test_api_meter_records_both_layers_and_real_response_models(monkeypatch):
    response = {"model": "gpt-6-astra-2026-09-03", "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(answer())}}]}
    monkeypatch.setattr(DecisionPolicy, "_post", lambda *a, **k: httpx.Response(200, json=response, request=httpx.Request("POST", "https://example.com")))
    budget = RequestBudget(2, 60, 2)
    client = ModelClient("chat", {"url": "https://example.com", "model": "gpt-6-astra", "key": "fake-test"}, budget)
    client.request("vision_plan", {}, system="test")
    client.request("local_control", {}, system="test")
    metrics = usage_summary(budget.calls)
    assert metrics["requests"] == 2 and metrics["usage"]["input_reported_tokens"] == 200
    assert metrics["estimated_usd"] == pytest.approx(.004)
    assert [call["stage"] for call in budget.calls] == ["vision_plan", "local_control"]
    with pytest.raises(RuntimeError, match="request_budget"):
        client.request("local_control", {}, system="test")
    assert len(budget.calls) == 2


def test_timeout_is_counted_with_unknown_cost(monkeypatch):
    def fail(*a, **kw):
        raise httpx.ReadTimeout("read timed out")
    monkeypatch.setattr(DecisionPolicy, "_post", fail)
    budget = RequestBudget(2, 60, 2)
    client = ModelClient("chat", {"url": "https://example.com", "model": "gpt-6-astra", "key": "fake-test"}, budget)
    with pytest.raises(httpx.ReadTimeout):
        client.request("vision_plan", {}, system="test")
    assert len(budget.calls) == 1
    assert usage_summary(budget.calls)["estimated_usd"] is None
    assert budget.calls[0]["error_type"] == "ReadTimeout"


def test_pre_action_timeout_can_be_retried_and_both_attempts_are_metered(monkeypatch):
    calls = 0
    def post(*a, **kw):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("transient timeout")
        return httpx.Response(200, request=httpx.Request("POST", "https://example.com"), json={
            "model": "gpt-6-astra", "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(answer())}}]})
    monkeypatch.setattr(DecisionPolicy, "_post", post)
    budget = RequestBudget(3, 60, 2)
    client = ModelClient("chat", {"url": "https://example.com", "model": "gpt-6-astra", "key": "fake-test"},
                         budget, request_retries=1)
    assert client.request("vision_plan", {}, system="test")["stage"] == "approach"
    assert calls == 2 and len(budget.calls) == 2
    assert budget.calls[0]["error_type"] == "ReadTimeout"
    assert usage_summary(budget.calls)["estimated_usd"] is None


def test_transient_gateway_status_can_be_retried_before_action(monkeypatch):
    calls = 0
    def post(*a, **kw):
        nonlocal calls
        calls += 1
        request = httpx.Request("POST", "https://example.com")
        if calls == 1:
            return httpx.Response(529, request=request, json={"error": "overloaded"})
        return httpx.Response(200, request=request, json={"model": "gpt-6-astra",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(answer())}}]})
    monkeypatch.setattr(DecisionPolicy, "_post", post)
    budget = RequestBudget(3, 60, 2)
    client = ModelClient("chat", {"url": "https://example.com", "model": "gpt-6-astra", "key": "fake-test"},
                         budget, request_retries=1)
    assert client.request("vision_plan", {}, system="test")["stage"] == "approach"
    assert calls == 2 and [call["http_status"] for call in budget.calls] == [529, 200]


def test_expired_response_is_never_returned_as_executable_plan(monkeypatch):
    budget = RequestBudget(2, 60, 2)
    def late(*a, **kw):
        budget.deadline = 0
        return httpx.Response(200, json={"model": "gpt-6-astra", "choices": [
            {"finish_reason": "stop", "message": {"content": "{}"}}]}, request=httpx.Request("POST", "https://example.com"))
    monkeypatch.setattr(DecisionPolicy, "_post", late)
    client = ModelClient("chat", {"url": "https://example.com", "model": "gpt-6-astra", "key": "fake-test"}, budget)
    with pytest.raises(RuntimeError, match="time_budget"):
        client.request("vision_plan", {}, system="test")
    assert len(budget.calls) == 1


def test_wall_time_budget_allows_more_requests_but_keeps_time_and_cost_guards():
    budget = RequestBudget(None, 1200, 20)
    budget.calls = [{"estimated_usd": .01} for _ in range(100)]
    budget.check()
    budget.deadline = 0
    with pytest.raises(RuntimeError, match="time_budget"):
        budget.check()
    budget = RequestBudget(None, 1200, 20)
    budget.calls = [{"estimated_usd": 20}]
    with pytest.raises(RuntimeError, match="cost_budget"):
        budget.check()


def test_request_cut_off_at_episode_deadline_is_time_budget_not_network_failure(monkeypatch):
    budget = RequestBudget(None, 1200, 20)
    def expire(*a, **kw):
        budget.deadline = 0
        raise httpx.ReadTimeout("deadline reached")
    monkeypatch.setattr(DecisionPolicy, "_post", expire)
    client = ModelClient("chat", {"url": "https://example.com", "model": "gpt-6-astra", "key": "fake-test"}, budget)
    with pytest.raises(RuntimeError, match="time_budget"):
        client.request("vision_plan", {}, system="test")
    assert len(budget.calls) == 1
    assert usage_summary(budget.calls)["estimated_usd"] is None


def test_visual_plan_retries_rejected_geometry_with_public_feedback(monkeypatch):
    invalid = {**answer(), "target": {"kind": "world", "xyz_m": [2, 0, .5]}}
    valid = {**answer(), "target": {"kind": "relative", "delta_m": [0, 0, .1]}}
    replies, requests = [invalid, valid], []
    def post(*args, **kwargs):
        requests.append(kwargs["json"])
        reply = replies.pop(0)
        return httpx.Response(200, request=httpx.Request("POST", "https://example.com"), json={
            "model": "gpt-6-astra", "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(reply)}}]})
    monkeypatch.setattr(DecisionPolicy, "_post", post)
    budget = RequestBudget(3, 60, 2)
    client = ModelClient("chat", {"url": "https://example.com", "model": "gpt-6-astra", "key": "fake-test"}, budget)
    plan, _ = client.plan(observation(), {"external": camera()}, None, [], validation_retries=1)
    np.testing.assert_allclose(plan["target_xyz"], [.1, .1, .7])
    second_state = json.loads(requests[1]["messages"][1]["content"][0]["text"])
    assert second_state["rejected_plans"][0]["error"] == "Visual waypoint is outside the declared world workspace"
    assert "Do not repeat" in second_state["correction"]
    assert len(budget.calls) == 2
