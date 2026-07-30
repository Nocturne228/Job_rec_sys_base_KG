import json

import pytest

from scripts.fetch_usajobs import normalize_items
from src.cli import main
from src.data.external import prepare_external_snapshot


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_external_snapshot_validates_references_and_pseudonymizes_users(tmp_path):
    jobs_path = tmp_path / "jobs.jsonl"
    interactions_path = tmp_path / "interactions.jsonl"
    output = tmp_path / "normalized"
    _write_jsonl(
        jobs_path,
        [
            {
                "job_id": "source:j1",
                "title": "Data Engineer",
                "company": "Public Agency",
                "description": "Build Python data pipelines",
                "required_skills": {"python": "intermediate"},
            }
        ],
    )
    _write_jsonl(
        interactions_path,
        [
            {
                "user_id": "raw-user-42",
                "job_id": "source:j1",
                "interaction_type": "click",
                "timestamp": "2026-07-30T10:00:00Z",
            }
        ],
    )

    with pytest.raises(ValueError, match="JOBREC_IMPORT_PSEUDONYM_KEY"):
        prepare_external_snapshot(
            jobs_path,
            output,
            source_name="licensed-test-source",
            source_url="https://example.invalid/source",
            license_name="test-only",
            license_url="https://example.invalid/license",
            retrieved_at="2026-07-30",
            interactions_path=interactions_path,
        )

    manifest = prepare_external_snapshot(
        jobs_path,
        output,
        source_name="licensed-test-source",
        source_url="https://example.invalid/source",
        license_name="test-only",
        license_url="https://example.invalid/license",
        retrieved_at="2026-07-30",
        interactions_path=interactions_path,
        pseudonymization_secret="unit-test-secret",
    )

    normalized = json.loads((output / "interactions.jsonl").read_text(encoding="utf-8"))
    assert normalized["user_id"].startswith("user_")
    assert "raw-user-42" not in (output / "interactions.jsonl").read_text()
    assert manifest.jobs == 1
    assert manifest.interactions == 1
    assert manifest.distinct_users == 1
    assert set(manifest.input_sha256) == {"jobs", "interactions"}

    bad_interactions = tmp_path / "bad-interactions.jsonl"
    _write_jsonl(
        bad_interactions,
        [
            {
                "user_id": "u1",
                "job_id": "missing-job",
                "interaction_type": "view",
                "timestamp": "2026-07-30T10:00:00Z",
            }
        ],
    )
    with pytest.raises(ValueError, match="unknown jobs"):
        prepare_external_snapshot(
            jobs_path,
            output,
            source_name="test",
            source_url="https://example.invalid",
            license_name="test",
            license_url="https://example.invalid/license",
            retrieved_at="2026-07-30",
            interactions_path=bad_interactions,
            pseudonymization_secret="unit-test-secret",
        )


def test_usajobs_normalization_and_read_only_default_cli(capsys):
    payload = {
        "SearchResult": {
            "SearchResultItems": [
                {
                    "MatchedObjectDescriptor": {
                        "PositionID": "ABC-123",
                        "PositionTitle": "Software Engineer",
                        "OrganizationName": "Example Agency",
                        "PositionURI": "https://www.usajobs.gov/job/123",
                        "UserArea": {
                            "Details": {
                                "JobSummary": "Build public services.",
                                "MajorDuties": ["Design APIs", "Review systems"],
                                "QualificationSummary": "Python experience",
                            }
                        },
                    }
                }
            ]
        }
    }
    jobs = list(normalize_items(payload))
    assert jobs[0]["job_id"] == "usajobs:ABC-123"
    assert "Python experience" in jobs[0]["description"]
    assert jobs[0]["required_skills"] == {}

    assert main([]) == 0
    assert "build-bundle" in capsys.readouterr().out
