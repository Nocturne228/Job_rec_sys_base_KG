from fastapi.testclient import TestClient

from src.api.routes import app


def _token(client: TestClient, username: str) -> str:
    password = {
        "admin": "jobrec-admin-demo",
        "recruiter": "jobrec-recruiter-demo",
    }.get(username, "jobrec-demo")
    response = client.post(
        "/api/token", json={"username": username, "password": password}
    )
    assert response.status_code == 200
    return response.json()["access_token"]


def test_authenticated_api_known_cold_start_competency_and_feedback(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("JOBREC_EVENT_DB", str(tmp_path / "events.sqlite3"))
    monkeypatch.setenv("JOBREC_PROFILE_DB", str(tmp_path / "profiles.sqlite3"))
    monkeypatch.setenv("JOBREC_PROFILE_MASTER_KEY", "api-profile-test-master-key")
    monkeypatch.setenv("JOBREC_TOKEN_SECRET", "api-contract-test-secret")
    monkeypatch.setenv("JOBREC_USE_PRETRAINED_SBERT", "0")
    with TestClient(app) as client:
        assert client.get("/openapi.json").json()["info"]["version"] == "2.2"
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").json()["status"] == "ready"
        assert (
            client.post("/api/recommend", json={"user_id": "user_001"}).status_code
            == 401
        )
        assert (
            client.post(
                "/api/token",
                json={"username": "admin", "password": "jobrec-demo"},
            ).status_code
            == 401
        )
        assert (
            client.post(
                "/api/token",
                json={"username": "recruiter", "password": "jobrec-demo"},
            ).status_code
            == 401
        )

        user_token = _token(client, "user_001")
        headers = {"Authorization": f"Bearer {user_token}"}
        profile_payload = {
            "user_id": "user_001",
            "name": "Alice Chen",
            "phone": "13800138000",
            "email": "alice@example.com",
            "address": "1 Example Road, Shanghai",
        }
        stored_profile = client.post(
            "/api/profile", json=profile_payload, headers=headers
        )
        assert stored_profile.status_code == 200
        assert stored_profile.json()["status"] == "stored_encrypted"
        own_profile = client.get("/api/profile/user_001", headers=headers)
        assert own_profile.status_code == 200
        assert own_profile.json() == profile_payload

        other_token = _token(client, "user_002")
        assert (
            client.get(
                "/api/profile/user_001",
                headers={"Authorization": f"Bearer {other_token}"},
            ).status_code
            == 403
        )
        recruiter_token = _token(client, "recruiter")
        assert (
            client.get(
                "/api/profile/user_001",
                headers={"Authorization": f"Bearer {recruiter_token}"},
            ).status_code
            == 403
        )

        known = client.post(
            "/api/recommend", json={"user_id": "user_001"}, headers=headers
        )
        assert known.status_code == 200
        assert known.json()[0]["retrieval_mode"].startswith("hybrid_lightgcn")
        assert any(abs(row["contributions"]["lightgcn"]) > 0 for row in known.json())
        assert len({row["request_id"] for row in known.json()}) == 1
        assert len({row["impression_id"] for row in known.json()}) == len(known.json())
        seen = set(client.app.state.pipeline["bundle"].train_items_by_user["user_001"])
        assert seen.isdisjoint(row["job_id"] for row in known.json())

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

        missing_impression = client.post(
            "/api/feedback",
            json={
                "user_id": "user_001",
                "job_id": known.json()[0]["job_id"],
                "satisfied": True,
            },
            headers=headers,
        )
        assert missing_impression.status_code == 422
        feedback = client.post(
            "/api/feedback",
            json={
                "impression_id": known.json()[0]["impression_id"],
                "user_id": "user_001",
                "job_id": known.json()[0]["job_id"],
                "satisfied": True,
            },
            headers=headers,
        )
        assert feedback.status_code == 200
        assert feedback.json()["impression_id"] == known.json()[0]["impression_id"]
        assert feedback.json()["effectiveness"] == 1.0
        duplicate = client.post(
            "/api/feedback",
            json={
                "impression_id": known.json()[0]["impression_id"],
                "user_id": "user_001",
                "job_id": known.json()[0]["job_id"],
                "satisfied": False,
            },
            headers=headers,
        )
        assert duplicate.status_code == 409
        unexposed = client.post(
            "/api/feedback",
            json={
                "impression_id": 999_999,
                "user_id": "user_001",
                "job_id": "unseen",
                "satisfied": True,
            },
            headers=headers,
        )
        assert unexposed.status_code == 409

        admin = _token(client, "admin")
        admin_headers = {"Authorization": f"Bearer {admin}"}
        assert (
            client.get("/api/profile/user_001", headers=admin_headers).status_code
            == 200
        )
        report = client.get(
            "/api/effectiveness",
            headers=admin_headers,
        )
        assert report.status_code == 200
        assert report.json()["pass_threshold"] is True
        deleted_profile = client.delete("/api/profile/user_001", headers=headers)
        assert deleted_profile.status_code == 200
        assert client.get("/api/profile/user_001", headers=headers).status_code == 404
