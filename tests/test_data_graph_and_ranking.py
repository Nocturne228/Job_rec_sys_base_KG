import numpy as np

from src.data import (
    GraphEntities,
    GraphLoader,
    JobPosting,
    Skill,
    SkillLevel,
    SkillRelation,
    User,
    compatibility_score,
    generate_mock_data,
)
from src.ranking import LinearFusionRanker, RankingFeatures
from src.recall import SBERTRecall


def test_generated_behavior_is_compatibility_conditioned():
    data = generate_mock_data(60, 80, seed=7)
    users = {user.id: user for user in data.users}
    jobs = {job.id: job for job in data.jobs}
    observed = {
        (interaction.user_id, interaction.job_id) for interaction in data.interactions
    }
    positive = [
        compatibility_score(users[user_id], jobs[job_id])
        for user_id, job_id in observed
    ]
    negative = [
        compatibility_score(user, job)
        for user in data.users
        for job in data.jobs
        if (user.id, job.id) not in observed
    ]
    assert np.mean(positive) > np.mean(negative) + 0.08
    for job in data.jobs:
        assert all(
            next(skill.name for skill in data.skills if skill.id == skill_id)
            in job.description
            for skill_id in job.required_skills
        )


def test_typed_graph_path_contains_edge_provenance():
    entities = GraphEntities(
        users=[
            User(
                id="u1",
                name="Candidate",
                skills={"python": SkillLevel.INTERMEDIATE},
            )
        ],
        jobs=[
            JobPosting(
                id="j1",
                title="Data Engineer",
                company="Example",
                description="Requires NumPy",
                required_skills={"numpy": SkillLevel.INTERMEDIATE},
            )
        ],
        skills=[
            Skill(id="python", name="Python", category="programming"),
            Skill(id="numpy", name="NumPy", category="data"),
        ],
        applications=[],
        interactions=[],
        skill_relations=[
            SkillRelation(
                source_skill_id="python",
                target_skill_id="numpy",
                confidence=0.95,
                source="test_curriculum",
            )
        ],
    )
    paths = GraphLoader(entities).find_paths_for_skills(
        {"python": "intermediate"}, "j1"
    )
    assert paths[0]["skills"] == ["python", "numpy"]
    assert paths[0]["evidence"][0]["relation_type"] == "PREREQUISITE_OF"
    assert paths[0]["evidence"][0]["source"] == "test_curriculum"


def test_query_normalization_is_stateless_and_explanations_are_additive():
    ranker = LinearFusionRanker(normalization_mode="query")
    first = [
        RankingFeatures(lightgcn_score=0.2, sbert_score=0.9, skill_coverage=0.5),
        RankingFeatures(lightgcn_score=0.8, sbert_score=0.1, skill_coverage=0.7),
    ]
    unrelated = [
        RankingFeatures(lightgcn_score=-10, sbert_score=100, skill_coverage=0),
        RankingFeatures(lightgcn_score=20, sbert_score=-2, skill_coverage=1),
    ]
    before = ranker.rank_with_explanations(first)
    ranker.rank(unrelated)
    after = ranker.rank_with_explanations(first)
    assert before == after
    assert ranker.feature_stats == {}
    for _, score, contributions in before:
        assert score == sum(contributions.values())


def test_offline_semantic_baseline_preserves_lexical_overlap():
    recall = SBERTRecall(use_faiss=False, use_pretrained=False)
    recall.add_job("python", "Python pandas machine learning")
    recall.add_job("frontend", "JavaScript React CSS")
    results = recall.recommend_for_text("Python machine learning", k=2)
    assert results[0][0] == "python"
    assert recall.get_embedding_stats()["encoder"] == "feature_hashing"
