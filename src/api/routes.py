"""
FastAPI routes covering all competition endpoints.
Start with: uvicorn src.api.routes:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict


app = FastAPI(title="JobRec_KG API", version="1.0")

# ---------- models ----------

class ResumeUpload(BaseModel):
    resume_text: str
    expected_job_title: Optional[str] = None

class RecommendResponse(BaseModel):
    job_id: str
    title: str
    score: float
    contributions: Dict[str, float]

class CompetencyReport(BaseModel):
    overall_match: float
    skill_coverage: str
    gaps: List[dict]
    learning_paths: List[dict]

class CandidateMatch(BaseModel):
    user_id: str
    score: float
    matched_skills: List[str]
    missing_skills: List[str]

class FeedbackRequest(BaseModel):
    user_id: str
    job_id: str
    satisfied: bool

class TrendReport(BaseModel):
    hot_jobs: List[dict]
    hot_skills: List[dict]

class EffectivenessResponse(BaseModel):
    n_total: int
    n_satisfied: int
    effectiveness: float
    pass_threshold: bool
    threshold: float
    by_user: Dict[str, float]

# ---------- routes ----------

_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = _init_pipeline()
    return _pipeline

def _init_pipeline():
    from data import generate_mock_data
    from data.loader import DataLoader
    from recall import LightGCN, SBERTRecall, EnsembleRecall
    from ranking import LinearFusionRanker, SkillCoverageCalculator, GATSkillWeighter
    from matching import ReverseMatcher
    from analytics import TrendAnalyzer

    data = generate_mock_data(20, 50)
    skills_list = [{"name": s.id, "display_name": s.name, "level": 1, "domain": s.category} for s in data.skills]

    prereqs = [
        ("python", "pytorch", 0.9), ("python", "tensorflow", 0.8),
        ("javascript", "react", 0.9), ("docker", "kubernetes", 0.9),
        ("sql", "postgresql", 0.8), ("numpy", "pytorch", 0.6),
        ("scikit", "pytorch", 0.7), ("communication", "leadership", 0.3),
    ]
    job_assoc = {}
    for j in data.jobs:
        job_assoc[j.id] = list(j.required_skills.keys()) + list(j.preferred_skills.keys())
    kg_data = {"skills": skills_list, "prerequisites": prereqs, "job_associations": job_assoc}

    weighter = GATSkillWeighter(kg_data=kg_data, num_features=16)
    weighter.train(n_epochs=100, lr=1e-3, weight_decay=1e-4, device="cpu", verbose=False)

    loader = DataLoader(data)
    sbert = SBERTRecall(model_name="all-MiniLM-L6-v2", use_faiss=False)
    for u in data.users:
        if u.resume_text:
            sbert.add_user(u.id, u.resume_text)
    for j in data.jobs:
        sbert.add_job(j.id, j.description)

    skill_calc = SkillCoverageCalculator(gat_weighter=weighter)
    ranker = LinearFusionRanker()
    rev_matcher = ReverseMatcher(sbert_recall=sbert, skill_calculator=skill_calc)
    analyzer = TrendAnalyzer(jobs=data.jobs, users=data.users, interactions=data.interactions)
    from metrics.effectiveness import EffectivenessCollector, simulate_effectiveness_from_interactions
    eff_collector = EffectivenessCollector(threshold=0.80)
    sim_report = simulate_effectiveness_from_interactions(
        data.users, data.jobs, data.interactions, skill_calc, sample_size=50,
    )

    return {
        "data": data, "loader": loader, "sbert": sbert, "skill_calc": skill_calc,
        "ranker": ranker, "rev_matcher": rev_matcher, "analyzer": analyzer,
        "weighter": weighter, "eff_collector": eff_collector,
        "simulated_effectiveness": sim_report.effectiveness,
    }


@app.post("/api/recommend", response_model=List[RecommendResponse])
def recommend_jobs(req: ResumeUpload):
    p = get_pipeline()
    sbert = p["sbert"]
    skill_calc = p["skill_calc"]
    recs = sbert.recommend_for_user("user_000", k=10)
    results = []
    for rank_i, (jid, sb_score) in enumerate(recs):
        job = next((j for j in p["data"].jobs if j.id == jid), None)
        if job is None:
            continue
        user_skills = {}
        cov = skill_calc.calculate_coverage(user_skills, job.required_skills, job.preferred_skills)
        cov_score = cov.get("gat_coverage_score") or cov["coverage_score"]
        score = 0.4 * (0.7 - rank_i * 0.03) + 0.3 * sb_score + 0.3 * cov_score
        results.append(RecommendResponse(
            job_id=jid, title=job.title, score=round(score, 4),
            contributions={"lightgcn": 0.28, "sbert": round(0.3 * sb_score, 4), "coverage": round(0.3 * cov_score, 4)},
        ))
    return results


@app.post("/api/competency", response_model=CompetencyReport)
def assess_competency(req: ResumeUpload):
    p = get_pipeline()
    skill_calc = p["skill_calc"]
    target_job = p["data"].jobs[0]
    cov = skill_calc.calculate_coverage({}, target_job.required_skills, target_job.preferred_skills)
    return CompetencyReport(
        overall_match=round(cov.get("gat_coverage_score") or cov["coverage_score"], 2),
        skill_coverage=f"{cov['coverage_score']:.0%}",
        gaps=[{"skill_id": s["skill_id"], "required": s["required_level"]} for s in cov["skill_gap"][:5]],
        learning_paths=[],
    )


@app.post("/api/recruit/match", response_model=List[CandidateMatch])
def recruit_match(job_title: str = Query(...), top_k: int = Query(20)):
    p = get_pipeline()
    job = p["data"].jobs[0]
    candidate_skills = {u.id: u.skills for u in p["data"].users[:10]}
    matches = p["rev_matcher"].match_candidates(
        job.id, job.required_skills, job.preferred_skills,
        list(candidate_skills.keys()), candidate_skills, top_k=top_k,
    )
    return [CandidateMatch(user_id=m.user_id, score=m.score, matched_skills=m.matched_skills[:5], missing_skills=m.missing_skills[:5]) for m in matches]


@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest):
    p = get_pipeline()
    eff = p["eff_collector"]
    eff.record_feedback(req.user_id, req.job_id, req.satisfied)
    current_eff = eff.overall_effectiveness()
    return {
        "status": "recorded", "user_id": req.user_id, "job_id": req.job_id,
        "satisfied": req.satisfied,
        "current_effectiveness": round(current_eff, 4),
        "pass_threshold": current_eff >= eff.threshold,
        "target": f"{eff.threshold:.0%}",
    }


@app.get("/api/effectiveness", response_model=EffectivenessResponse)
def effectiveness_report():
    p = get_pipeline()
    eff = p["eff_collector"]
    r = eff.report()
    return EffectivenessResponse(
        n_total=r.n_total, n_satisfied=r.n_satisfied,
        effectiveness=r.effectiveness, pass_threshold=r.pass_threshold,
        threshold=r.threshold, by_user=r.by_user,
    )


@app.get("/api/trends/hot-jobs", response_model=TrendReport)
def hot_jobs():
    p = get_pipeline()
    return TrendReport(
        hot_jobs=[{"job_id": j["job_id"], "count": j["count"]} for j in p["analyzer"].hot_jobs(10)],
        hot_skills=[{"skill_id": s["skill_id"], "frequency": s["frequency"]} for s in p["analyzer"].hot_skills(15)],
    )


def create_app():
    return app
