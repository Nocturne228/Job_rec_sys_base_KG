from src.data import (
    GraphEntities,
    GraphLoader,
    JobPosting,
    Skill,
    SkillLevel,
    SkillRelation,
    User,
)
from src.ranking import (
    FeedRankingFeatures,
    LinearFusionRanker,
    PointwiseRanker,
    RankingFeatures,
    SkillCoverageCalculator,
)
from src.recall import TextRecall, merge_recall_routes


def test_query_ranking_is_stateless_and_explanations_add_up():
    ranker = LinearFusionRanker()
    rows = [
        RankingFeatures(lightgcn=0.2, text=0.9, skill=0.5),
        RankingFeatures(lightgcn=0.8, text=0.1, skill=0.7),
    ]
    before = ranker.rank_with_explanations(rows)
    ranker.rank([RankingFeatures(lightgcn=-10, text=100, skill=0)])
    assert before == ranker.rank_with_explanations(rows)
    for _, score, contributions in before:
        assert score == sum(contributions.values())


def test_skill_coverage_respects_proficiency_and_text_supports_cold_start():
    coverage = SkillCoverageCalculator().calculate_coverage(
        {"python": "beginner"}, {"python": "advanced"}
    )
    assert coverage["coverage_score"] == 0.0
    recall = TextRecall()
    recall.add_job("python", "Python pandas machine learning")
    recall.add_job("frontend", "JavaScript React CSS")
    assert recall.recommend_for_text("Python machine learning", k=1)[0][0] == "python"


def test_learning_path_uses_typed_edge_with_provenance():
    entities = GraphEntities(
        users=[
            User(id="u1", name="Candidate", skills={"python": SkillLevel.INTERMEDIATE})
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


def test_multi_route_recall_and_learned_ranker_keep_auditable_evidence():
    candidates = merge_recall_routes(
        {
            "collaborative": {"j1": 0.9, "j2": 0.1},
            "semantic": {"j2": 0.8, "j1": 0.2},
        },
        ["j1", "j2"],
        per_route_k=1,
    )
    assert {row.job_id for row in candidates} == {"j1", "j2"}
    assert {source for row in candidates for source in row.sources} == {
        "collaborative",
        "semantic",
    }

    ranker = PointwiseRanker.fit(
        [
            FeedRankingFeatures(text=0.1, skill=0.1),
            FeedRankingFeatures(text=0.9, skill=0.8),
            FeedRankingFeatures(text=0.2, skill=0.3),
            FeedRankingFeatures(text=0.8, skill=0.9),
        ],
        [0, 1, 0, 1],
    )
    for _, score, contributions in ranker.rank_with_explanations(
        [FeedRankingFeatures(text=0.7, skill=0.8)]
    ):
        assert abs(score - sum(contributions.values())) < 1e-9
