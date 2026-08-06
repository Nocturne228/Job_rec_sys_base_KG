from pathlib import Path

from fastapi.testclient import TestClient

from scripts.build_model_bundle import build
from src.api.routes import app
from src.models import ModelBundle


def test_bundle_publish_has_content_identity(tmp_path: Path):
    path = tmp_path / "bundle.json"
    first = build(str(path), seed=42, epochs=1)
    second = build(str(path), seed=42, epochs=1)
    loaded = ModelBundle.load(path)
    assert loaded.model_version == first.model_version
    assert second.model_version == first.model_version
    assert second.checkpoint_sha256 == first.checkpoint_sha256
    assert loaded.serving_sha256[:16] in loaded.model_version
    assert Path(loaded.checkpoint_path).name.startswith("lightgcn-")
    assert loaded.ranking_model["kind"] == "standardized_logistic"
    assert not list(tmp_path.glob("*.tmp"))


def test_api_tells_known_and_cold_story_and_links_feedback(tmp_path, monkeypatch):
    monkeypatch.setenv("JOBREC_EVENT_DB", str(tmp_path / "events.sqlite3"))
    with TestClient(app) as client:
        known = client.post("/api/recommend", json={"user_id": "user_001"})
        assert known.status_code == 200
        known_rows = known.json()
        assert known_rows[0]["retrieval_mode"] == "known_hybrid"
        assert known_rows[0]["recall_sources"]
        assert known_rows[0]["generation_mode"] == "deterministic_fallback"
        assert any(row["contributions"]["lightgcn"] != 0 for row in known_rows)
        seen = set(client.app.state.pipeline.bundle.train_items_by_user["user_001"])
        assert seen.isdisjoint(row["job_id"] for row in known_rows)
        for row in known_rows:
            assert abs(row["score"] - sum(row["contributions"].values())) < 1e-5

        cold = client.post(
            "/api/recommend", json={"resume_text": "Python pandas machine learning"}
        )
        assert cold.status_code == 200
        cold_rows = cold.json()
        assert cold_rows[0]["retrieval_mode"] == "cold_start_text_skill"

        report = client.post(
            "/api/competency", json={"user_id": "user_001", "job_id": "job_001"}
        )
        assert report.status_code == 200
        assert report.json()["evidence_source"] == "typed_in_memory_graph"

        first = known_rows[0]
        payload = {
            "impression_id": first["impression_id"],
            "subject_id": first["subject_id"],
            "job_id": first["job_id"],
            "satisfied": True,
        }
        feedback = client.post("/api/feedback", json=payload)
        assert feedback.status_code == 200
        assert feedback.json()["satisfaction_rate"] == 1.0
        cold_first = cold_rows[0]
        behavior = {
            "impression_id": cold_first["impression_id"],
            "subject_id": cold_first["subject_id"],
            "job_id": cold_first["job_id"],
            "clicked": True,
            "dwell_seconds": 28.0,
        }
        assert client.post("/api/feedback", json=behavior).status_code == 200
        assert client.post("/api/feedback", json=payload).status_code == 409
