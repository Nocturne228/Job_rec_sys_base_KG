import base64

import pytest
import torch

from src.generation import LLMSimulator, fallback_advice, validate_advice
from src.metrics import EventStore
from src.metrics.ab_test import mann_whitney_u, welch_t_test, z_test_proportions
from src.metrics.online_metrics import ActionType, OnlineMetricsCollector
from src.security import issue_token, verify_token
from src.utils.crypto import decrypt_personal_info, encrypt_personal_info
from src.utils.training import sample_unobserved_negatives


def test_aes_gcm_round_trip_randomness_and_tamper_detection():
    first = encrypt_personal_info("alice@example.com", "correct horse battery staple")
    second = encrypt_personal_info("alice@example.com", "correct horse battery staple")
    assert first != second
    assert (
        decrypt_personal_info(first, "correct horse battery staple")
        == "alice@example.com"
    )
    raw = bytearray(base64.urlsafe_b64decode(first))
    raw[-1] ^= 1
    with pytest.raises(Exception):
        decrypt_personal_info(
            base64.urlsafe_b64encode(raw).decode(), "correct horse battery staple"
        )


def test_signed_tokens_reject_expiry_and_tampering(monkeypatch):
    monkeypatch.setenv("JOBREC_TOKEN_SECRET", "unit-test-secret")
    token = issue_token("u1", "admin")
    assert verify_token(token)["role"] == "admin"
    with pytest.raises(Exception):
        verify_token(token[:-1] + ("a" if token[-1] != "a" else "b"))
    with pytest.raises(Exception):
        verify_token(issue_token("u1", expires_seconds=-1))


def test_event_store_persists_feedback_across_instances(tmp_path):
    path = tmp_path / "events.sqlite3"
    first = EventStore(str(path))
    first.record("impression", "u1", "j1", "v1")
    first.record("impression", "u1", "j2", "v1")
    first.record_feedback("u1", "j1", "v1", True)
    first.record_feedback("u1", "j2", "v1", False)
    with pytest.raises(ValueError, match="matching impression"):
        first.record_feedback("u1", "unseen", "v1", True)
    second = EventStore(str(path))
    stats = second.effectiveness()
    assert stats["n_total"] == 2
    assert stats["effectiveness"] == 0.5


def test_negative_sampler_never_selects_observed_item():
    train = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    negatives, retained = sample_unobserved_negatives(train, torch.tensor([0, 1]))
    assert retained.tolist() == [0]
    assert negatives.tolist() == [1]
    assert train[0, negatives[0]] == 0


def test_statistical_tests_handle_unequal_variance_ties_and_pooled_null():
    assert welch_t_test([1, 2, 3, 4], [10, 11, 13, 18])["p_value"] < 0.05
    tied = mann_whitney_u([1, 1, 2, 2], [1, 2, 2, 3])
    assert 0 <= tied["p_value"] <= 1
    null = z_test_proportions(0.1, 1000, 0.1, 1000)
    assert null["p_value"] > 0.4


def test_online_metrics_count_events_and_detect_sample_ratio_mismatch():
    collector = OnlineMetricsCollector()
    for index in range(50):
        collector.record(f"a{index}", "j", ActionType.IMPRESSION, group="A")
    for index in range(2):
        collector.record(f"b{index}", "j", ActionType.IMPRESSION, group="B")
    collector.record("a0", "j", ActionType.IMPRESSION, group="A")
    assert collector.group_metrics("A")["impressions"] == 51
    assert collector.sample_ratio_mismatch()["mismatch"] is True


def test_generation_fallback_is_schema_valid_and_nonempty():
    advice = fallback_advice(
        {"python": {"user_level": "beginner", "required_level": "advanced"}}
    )
    validated = validate_advice(advice.model_dump_json())
    assert validated.learning_path
    assert validated.learning_path[0].skill_id == "python"


def test_llm_simulator_only_parses_bulleted_gap_rows():
    prompt = """## Skill Gap Analysis
The user has the following skill gaps:
- python: Current=None, Required=advanced
Skill Coverage: 25.0%
## Task
Provide JSON output containing: summary
"""
    response = LLMSimulator(seed=7).generate(prompt)
    advice = validate_advice(response["response"])
    assert advice.critical_skills == ["python"]
