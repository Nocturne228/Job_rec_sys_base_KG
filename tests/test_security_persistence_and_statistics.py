import base64
import sqlite3
from contextlib import closing

import pytest
import torch

from src.data import EncryptedProfileStore, PrivateProfile
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


def test_private_profile_store_encrypts_four_fields_and_binds_ciphertext(tmp_path):
    path = tmp_path / "private-profiles.sqlite3"
    store = EncryptedProfileStore(str(path), "profile-store-test-master-key")
    profile = PrivateProfile(
        name="Alice Chen",
        phone="13800138000",
        email="alice@example.com",
        address="1 Example Road, Shanghai",
    )
    store.upsert("user_001", profile)

    with closing(sqlite3.connect(path)) as db:
        row = db.execute("""
            SELECT name_ciphertext, phone_ciphertext,
                   email_ciphertext, address_ciphertext
            FROM private_profiles WHERE user_id='user_001'
            """).fetchone()
        assert row is not None
        assert all(
            value and value not in profile.model_dump().values() for value in row
        )
        db.execute(
            """
            UPDATE private_profiles
            SET name_ciphertext=?, email_ciphertext=?
            WHERE user_id='user_001'
            """,
            (row[2], row[0]),
        )
        db.commit()

    database_bytes = path.read_bytes()
    for plaintext in profile.model_dump().values():
        assert plaintext.encode() not in database_bytes

    # Context-bound AAD rejects moving a valid ciphertext to another field.
    with pytest.raises(Exception):
        store.get("user_001")

    clean = EncryptedProfileStore(
        str(tmp_path / "clean-private-profiles.sqlite3"),
        "profile-store-test-master-key",
    )
    clean.upsert("user_001", profile)
    assert clean.get("user_001") == profile
    assert clean.delete("user_001") is True
    assert clean.get("user_001") is None


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
    first_impression = first.record_impression("u1", "j1", "v1")
    second_impression = first.record_impression("u1", "j2", "v1")
    first.record_feedback(first_impression, "u1", "j1", "v1", True)
    first.record_feedback(second_impression, "u1", "j2", "v1", False)
    with pytest.raises(ValueError, match="exact matching impression"):
        first.record_feedback(first_impression, "u1", "unseen", "v1", True)
    with pytest.raises(ValueError, match="already recorded"):
        first.record_feedback(first_impression, "u1", "j1", "v1", True)
    second = EventStore(str(path))
    stats = second.effectiveness()
    assert stats["n_total"] == 2
    assert stats["effectiveness"] == 0.5
    feedback_events = [
        event for event in second.list_events() if event["event_type"] == "feedback"
    ]
    assert {event["impression_id"] for event in feedback_events} == {
        first_impression,
        second_impression,
    }


def test_event_store_migrates_legacy_schema_before_exact_feedback(tmp_path):
    path = tmp_path / "legacy-events.sqlite3"
    with closing(sqlite3.connect(path)) as db:
        with db:
            db.execute("""
                CREATE TABLE events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    payload TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """)
            cursor = db.execute("""
                INSERT INTO events(event_type,user_id,job_id,model_version)
                VALUES('impression','u1','j1','v1')
                """)
            impression_id = int(cursor.lastrowid)
            db.execute("""
                INSERT INTO events(event_type,user_id,job_id,model_version,payload)
                VALUES('feedback','u1','j1','v1','{"satisfied": false}')
                """)

    store = EventStore(str(path))
    store.record_feedback(impression_id, "u1", "j1", "v1", True)
    # A legacy feedback row has no exact exposure relationship and is retained
    # for inspection, but it must not affect the post-migration statistic.
    assert store.effectiveness()["n_total"] == 1
    assert store.effectiveness()["n_satisfied"] == 1
    assert "impression_id" in store.list_events()[0]


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
