from fastapi.testclient import TestClient

from src.api.routes import app


def _token(client: TestClient, username: str) -> str:
    response = client.post(
        "/api/token", json={"username": username, "password": "jobrec-demo"}
    )
    assert response.status_code == 200
    return response.json()["access_token"]


def test_authenticated_api_known_cold_start_competency_and_feedback(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("JOBREC_EVENT_DB", str(tmp_path / "events.sqlite3"))
    monkeypatch.setenv("JOBREC_TOKEN_SECRET", "api-contract-test-secret")
    monkeypatch.setenv("JOBREC_USE_PRETRAINED_SBERT", "0")
    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").json()["status"] == "ready"
        assert (
            client.post("/api/recommend", json={"user_id": "user_001"}).status_code
            == 401
        )

        user_token = _token(client, "user_001")
        headers = {"Authorization": f"Bearer {user_token}"}
        known = client.post(
            "/api/recommend", json={"user_id": "user_001"}, headers=headers
        )
        assert known.status_code == 200
        assert known.json()[0]["retrieval_mode"].startswith("hybrid_lightgcn")
        assert any(abs(row["contributions"]["lightgcn"]) > 0 for row in known.json())

        cold_token = _token(client, "new_user")
        cold = client.post(
            "/api/recommend",
            json={"resume_text": "Python pandas machine learning"},
            headers={"Authorization": f"Bearer {cold_token}"},
        )
        assert cold.status_code == 200
        assert cold.json()[0]["retrieval_mode"] == "cold_start_semantic_skill"

        competency = client.post(
            "/api/competency",
            json={
                "resume_text": "Python",
                "job_id": "job_001",
            },
            headers=headers,
        )
        assert competency.status_code == 200
        assert competency.json()["learning_paths"]
        assert competency.json()["evidence_source"] == "typed_in_memory_graph"

        class StubGraphStore:
            def competency_evidence(self, user_id: str, job_id: str) -> dict:
                return {
                    "paths": [
                        {
                            "skills": ["python", "pytorch"],
                            "evidence": [{"source": "stub_graph"}],
                        }
                    ]
                }

        client.app.state.pipeline["graph_store"] = StubGraphStore()
        client.app.state.pipeline["graph_source"] = "stub_graph"
        known_competency = client.post(
            "/api/competency",
            json={"user_id": "user_001", "job_id": "job_001"},
            headers=headers,
        )
        assert known_competency.status_code == 200
        assert known_competency.json()["evidence_source"] == "stub_graph"
        assert known_competency.json()["graph_paths"][0]["evidence"][0]["source"] == (
            "stub_graph"
        )

        feedback = client.post(
            "/api/feedback",
            json={
                "user_id": "user_001",
                "job_id": known.json()[0]["job_id"],
                "satisfied": True,
            },
            headers=headers,
        )
        assert feedback.status_code == 200
        assert feedback.json()["effectiveness"] == 1.0
        unexposed = client.post(
            "/api/feedback",
            json={"user_id": "user_001", "job_id": "unseen", "satisfied": True},
            headers=headers,
        )
        assert unexposed.status_code == 409

        admin = _token(client, "admin")
        report = client.get(
            "/api/effectiveness",
            headers={"Authorization": f"Bearer {admin}"},
        )
        assert report.status_code == 200
        assert report.json()["pass_threshold"] is True
