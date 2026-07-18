"""Production-shaped FastAPI surface backed by an offline model bundle."""

from __future__ import annotations

import hmac
import json
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import torch
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, model_validator

from src.data import DataLoader, GraphLoader, generate_mock_data
from src.generation import (
    LLMSimulator,
    OpenAICompatibleLLM,
    fallback_advice,
    validate_advice,
)
from src.metrics import EventStore
from src.models import ModelBundle, StaticSkillWeighter
from src.ranking import LinearFusionRanker, RankingFeatures, SkillCoverageCalculator
from src.recall import LightGCN, SBERTRecall
from src.recall.lightgcn import prepare_adj_matrix
from src.security import issue_token, require_roles


class TokenRequest(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=200)


class ResumeUpload(BaseModel):
    resume_text: str = Field(default="", max_length=20_000)
    user_id: Optional[str] = None
    job_id: Optional[str] = None
    expected_job_title: Optional[str] = None

    @model_validator(mode="after")
    def require_identity_or_resume(self) -> "ResumeUpload":
        if not self.user_id and not self.resume_text.strip():
            raise ValueError("Provide user_id or resume_text")
        return self


class RecommendResponse(BaseModel):
    job_id: str
    title: str
    score: float
    contributions: Dict[str, float]
    retrieval_mode: str
    model_version: str


class CompetencyReport(BaseModel):
    job_id: str
    overall_match: float
    skill_coverage: str
    gaps: List[dict]
    learning_paths: List[dict]
    graph_paths: List[dict]
    advice_summary: str
    evidence_source: str


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


def _coverage_value(result: dict) -> float:
    weighted = result.get("gat_coverage_score")
    return float(weighted if weighted is not None else result["coverage_score"])


def _load_pipeline() -> dict:
    from src.analytics import TrendAnalyzer
    from src.data.graph_store import (
        InMemorySkillGraph,
        Neo4jSkillGraph,
        SkillGraphStore,
    )
    from src.matching import ReverseMatcher

    bundle_path = os.environ.get("JOBREC_BUNDLE_PATH", "models/jobrec_bundle.json")
    bundle = ModelBundle.load(bundle_path)
    data = generate_mock_data(20, 50, seed=bundle.data_seed)
    loader = DataLoader(data, random_seed=bundle.data_seed)
    if (
        loader.user_id_to_idx != bundle.user_id_to_idx
        or loader.job_id_to_idx != bundle.job_id_to_idx
    ):
        raise RuntimeError(
            "Model-bundle ID mappings do not match the generated dataset"
        )

    lightgcn = LightGCN.load(bundle.lightgcn_checkpoint)
    adjacency = prepare_adj_matrix(loader.get_sparse_graph())
    lightgcn.eval()
    with torch.no_grad():
        user_embeddings, item_embeddings = lightgcn(adjacency)

    use_pretrained = os.environ.get("JOBREC_USE_PRETRAINED_SBERT", "0") == "1"
    sbert = SBERTRecall(
        model_name="all-MiniLM-L6-v2",
        use_faiss=True,
        use_pretrained=use_pretrained,
    )
    for user in data.users:
        sbert.add_user(user.id, user.resume_text or "")
    for job in data.jobs:
        sbert.add_job(job.id, job.description)

    weighter = StaticSkillWeighter(bundle.skill_weights)
    skill_calc = SkillCoverageCalculator(gat_weighter=weighter)
    ranker = LinearFusionRanker(
        weights=bundle.ranking_weights, normalization_mode="query"
    )
    graph_loader = GraphLoader(data)
    graph_store: SkillGraphStore
    if os.environ.get("JOBREC_GRAPH_BACKEND") == "neo4j":
        graph_store = Neo4jSkillGraph(
            os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            os.environ.get("NEO4J_USER", "neo4j"),
            os.environ.get("NEO4J_PASSWORD", ""),
        )
        graph_source = "neo4j"
    else:
        graph_store = InMemorySkillGraph(data)
        graph_source = "typed_in_memory_graph"

    llm = (
        OpenAICompatibleLLM()
        if os.environ.get("JOBREC_LLM_ENDPOINT")
        else LLMSimulator(seed=42)
    )
    event_store = EventStore(
        os.environ.get("JOBREC_EVENT_DB", "data/jobrec_events.sqlite3")
    )
    analyzer = TrendAnalyzer(
        jobs=data.jobs, users=data.users, interactions=data.interactions
    )
    reverse = ReverseMatcher(sbert_recall=sbert, skill_calculator=skill_calc)
    return {
        "bundle": bundle,
        "data": data,
        "loader": loader,
        "lightgcn": lightgcn,
        "user_embeddings": user_embeddings,
        "item_embeddings": item_embeddings,
        "sbert": sbert,
        "skill_calc": skill_calc,
        "ranker": ranker,
        "graph_loader": graph_loader,
        "graph_store": graph_store,
        "graph_source": graph_source,
        "llm": llm,
        "event_store": event_store,
        "analyzer": analyzer,
        "reverse": reverse,
        "started_at": time.time(),
        "requests": 0,
        "total_latency_ms": 0.0,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = _load_pipeline()
    yield
    graph = app.state.pipeline.get("graph_store")
    if hasattr(graph, "close"):
        graph.close()


app = FastAPI(title="JobRec-KG API", version="2.0", lifespan=lifespan)


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    if hasattr(request.app.state, "pipeline"):
        pipeline = request.app.state.pipeline
        pipeline["requests"] += 1
        pipeline["total_latency_ms"] += elapsed
    response.headers["X-Process-Time-Ms"] = f"{elapsed:.2f}"
    return response


def get_pipeline(request: Request) -> dict:
    if not hasattr(request.app.state, "pipeline"):
        raise HTTPException(status_code=503, detail="Model bundle is not ready")
    return request.app.state.pipeline


def _extract_skills(resume_text: str, data) -> Dict[str, str]:
    normalized = resume_text.casefold()
    return {
        skill.id: "beginner"
        for skill in data.skills
        if any(
            re.search(rf"(?<!\w){re.escape(label.casefold())}(?!\w)", normalized)
            for label in (skill.id, skill.name)
        )
    }


def _candidate_jobs(data, title: Optional[str]):
    if not title or not title.strip():
        return data.jobs
    query = title.casefold().strip()
    matches = [job for job in data.jobs if query in job.title.casefold()]
    if not matches:
        raise HTTPException(status_code=404, detail=f"No job matches title: {title}")
    return matches


def _target_job(data, job_id: Optional[str], title: Optional[str]):
    if job_id:
        match = next((job for job in data.jobs if job.id == job_id), None)
        if match is None:
            raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")
        return match
    if not title:
        raise HTTPException(
            status_code=422,
            detail="Competency assessment requires job_id or exact title",
        )
    matches = [
        job for job in data.jobs if job.title.casefold() == title.casefold().strip()
    ]
    if len(matches) != 1:
        raise HTTPException(
            status_code=409, detail="Title is ambiguous; select a job_id"
        )
    return matches[0]


@app.post("/api/token")
def token(req: TokenRequest):
    expected = os.environ.get("JOBREC_DEMO_PASSWORD", "jobrec-demo")
    if not hmac.compare_digest(req.password, expected):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    role = (
        "admin"
        if req.username == "admin"
        else "recruiter" if req.username == "recruiter" else "user"
    )
    return {
        "access_token": issue_token(req.username, role),
        "token_type": "bearer",
        "role": role,
    }


@app.get("/health/live")
def live():
    return {"status": "ok"}


@app.get("/health/ready")
def ready(request: Request):
    if not hasattr(request.app.state, "pipeline"):
        raise HTTPException(status_code=503, detail="not ready")
    pipeline = request.app.state.pipeline
    return {
        "status": "ready",
        "model_version": pipeline["bundle"].model_version,
        "graph": pipeline["graph_store"].healthcheck(),
    }


@app.get("/api/model")
def model_info(
    request: Request, _=Depends(require_roles("user", "recruiter", "admin"))
):
    pipeline = get_pipeline(request)
    bundle = pipeline["bundle"]
    return {
        "model_version": bundle.model_version,
        "schema_version": bundle.schema_version,
        "created_at": bundle.created_at,
        "graph_source": pipeline["graph_source"],
    }


@app.post("/api/recommend", response_model=List[RecommendResponse])
def recommend_jobs(
    req: ResumeUpload, request: Request, claims=Depends(require_roles("user", "admin"))
):
    p = get_pipeline(request)
    user_map = {user.id: user for user in p["data"].users}
    known = bool(req.user_id and req.user_id in p["bundle"].user_id_to_idx)
    if req.user_id and req.user_id != claims["sub"] and claims["role"] != "admin":
        raise HTTPException(
            status_code=403, detail="Cannot request another user's recommendations"
        )
    resume = req.resume_text.strip() or (
        user_map[req.user_id].resume_text if known else ""
    )
    user_skills = _extract_skills(resume, p["data"])
    if known and not req.resume_text.strip():
        user_skills = {
            key: str(value.value if hasattr(value, "value") else value)
            for key, value in user_map[req.user_id].skills.items()
        }
    jobs = _candidate_jobs(p["data"], req.expected_job_title)
    job_ids = [job.id for job in jobs]
    semantic = dict(
        p["sbert"].recommend_for_text(resume, k=len(job_ids), job_ids=job_ids)
    )
    lg_scores: Dict[str, float] = {}
    mode = "cold_start_semantic_skill"
    if known:
        mode = "hybrid_lightgcn_semantic_skill"
        user_idx = p["bundle"].user_id_to_idx[req.user_id]
        raw = (p["user_embeddings"][user_idx] @ p["item_embeddings"].T).detach().cpu()
        seen = set(p["bundle"].train_items_by_user.get(req.user_id, []))
        lg_scores = {
            job_id: float(raw[p["bundle"].job_id_to_idx[job_id]])
            for job_id in job_ids
            if job_id not in seen
        }

    features = []
    coverage_by_job = {}
    for job in jobs:
        coverage = p["skill_calc"].calculate_coverage(
            user_skills, job.required_skills, job.preferred_skills
        )
        coverage_by_job[job.id] = coverage
        features.append(
            RankingFeatures(
                lightgcn_score=lg_scores.get(job.id, 0.0),
                sbert_score=semantic.get(job.id, 0.0),
                skill_coverage=_coverage_value(coverage),
            )
        )
    ranked = p["ranker"].rank_with_explanations(features)[:10]
    results: List[RecommendResponse] = []
    event_user = req.user_id or claims["sub"]
    for index, score, contribution in ranked:
        job = jobs[index]
        p["event_store"].record(
            "impression",
            event_user,
            job.id,
            p["bundle"].model_version,
            {"rank": len(results) + 1, "retrieval_mode": mode},
        )
        results.append(
            RecommendResponse(
                job_id=job.id,
                title=job.title,
                score=round(score, 4),
                contributions={
                    "lightgcn": round(contribution["lightgcn_score"], 4),
                    "sbert": round(contribution["sbert_score"], 4),
                    "coverage": round(contribution["skill_coverage"], 4),
                },
                retrieval_mode=mode,
                model_version=p["bundle"].model_version,
            )
        )
    return results


@app.post("/api/competency", response_model=CompetencyReport)
def assess_competency(
    req: ResumeUpload, request: Request, _=Depends(require_roles("user", "admin"))
):
    p = get_pipeline(request)
    job = _target_job(p["data"], req.job_id, req.expected_job_title)
    user_map = {user.id: user for user in p["data"].users}
    if req.resume_text.strip():
        user_skills = _extract_skills(req.resume_text, p["data"])
    elif req.user_id in user_map:
        user_skills = {
            key: str(value.value if hasattr(value, "value") else value)
            for key, value in user_map[req.user_id].skills.items()
        }
    else:
        user_skills = {}
    coverage = p["skill_calc"].calculate_coverage(
        user_skills, job.required_skills, job.preferred_skills
    )
    gap_dict = {
        item["skill_id"]: {
            "user_level": item.get("user_level"),
            "required_level": item["required_level"],
        }
        for item in coverage["skill_gap"]
    }
    if req.user_id in user_map and not req.resume_text.strip():
        graph_evidence = p["graph_store"].competency_evidence(req.user_id, job.id)
        graph_paths = graph_evidence.get("paths", [])
        evidence_source = p["graph_source"]
    else:
        graph_paths = p["graph_loader"].find_paths_for_skills(user_skills, job.id)
        evidence_source = "typed_in_memory_graph"
    fallback = fallback_advice(gap_dict)
    prompt = json.dumps(
        {
            "job_id": job.id,
            "skill_gaps": gap_dict,
            "graph_paths": graph_paths,
            "instruction": "Return the required career-advice JSON schema.",
        }
    )
    try:
        advice = validate_advice(p["llm"].generate(prompt, temperature=0.2)["response"])
    except Exception:
        advice = fallback
    return CompetencyReport(
        job_id=job.id,
        overall_match=round(_coverage_value(coverage), 4),
        skill_coverage=f"{coverage['coverage_score']:.0%}",
        gaps=[{"skill_id": key, **value} for key, value in gap_dict.items()],
        learning_paths=[step.model_dump() for step in advice.learning_path],
        graph_paths=graph_paths,
        advice_summary=advice.summary,
        evidence_source=evidence_source,
    )


@app.post("/api/recruit/match", response_model=List[CandidateMatch])
def recruit_match(
    request: Request,
    job_id: str = Query(...),
    top_k: int = Query(20, ge=1, le=100),
    _=Depends(require_roles("recruiter", "admin")),
):
    p = get_pipeline(request)
    job = _target_job(p["data"], job_id, None)
    candidate_skills = {user.id: user.skills for user in p["data"].users}
    matches = p["reverse"].match_candidates(
        job.id,
        job.required_skills,
        job.preferred_skills,
        list(candidate_skills),
        candidate_skills,
        top_k=top_k,
    )
    return [
        CandidateMatch(
            user_id=item.user_id,
            score=item.score,
            matched_skills=item.matched_skills[:5],
            missing_skills=item.missing_skills[:5],
        )
        for item in matches
    ]


@app.post("/api/feedback")
def submit_feedback(
    req: FeedbackRequest,
    request: Request,
    claims=Depends(require_roles("user", "admin")),
):
    p = get_pipeline(request)
    if req.user_id != claims["sub"] and claims["role"] != "admin":
        raise HTTPException(
            status_code=403, detail="Cannot submit feedback for another user"
        )
    try:
        p["event_store"].record_feedback(
            req.user_id,
            req.job_id,
            p["bundle"].model_version,
            req.satisfied,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    stats = p["event_store"].effectiveness()
    return {
        "status": "recorded",
        **stats,
        "pass_threshold": stats["effectiveness"] >= 0.8,
    }


@app.get("/api/effectiveness", response_model=EffectivenessResponse)
def effectiveness_report(request: Request, _=Depends(require_roles("admin"))):
    stats = get_pipeline(request)["event_store"].effectiveness()
    return EffectivenessResponse(
        **stats, pass_threshold=stats["effectiveness"] >= 0.8, threshold=0.8
    )


@app.get("/api/trends/hot-jobs", response_model=TrendReport)
def hot_jobs(request: Request, _=Depends(require_roles("user", "recruiter", "admin"))):
    analyzer = get_pipeline(request)["analyzer"]
    return TrendReport(
        hot_jobs=analyzer.hot_jobs(10), hot_skills=analyzer.hot_skills(15)
    )


@app.get("/api/metrics")
def service_metrics(request: Request, _=Depends(require_roles("admin"))):
    p = get_pipeline(request)
    return {
        "requests": p["requests"],
        "mean_latency_ms": p["total_latency_ms"] / max(p["requests"], 1),
        "uptime_seconds": time.time() - p["started_at"],
        "model_version": p["bundle"].model_version,
    }


@app.get("/demo", response_class=HTMLResponse)
def demo_page():
    return """<!doctype html><html><head><meta charset='utf-8'><title>JobRec-KG Demo</title>
<style>body{font:16px system-ui;max-width:960px;margin:40px auto;color:#17324d}textarea,input{width:100%;padding:10px;margin:6px 0}button{padding:10px 18px;background:#2f6b9a;color:white;border:0;border-radius:5px}pre{background:#eef3f7;padding:16px;white-space:pre-wrap}</style></head>
<body><h1>JobRec-KG Evidence Demo</h1><p>This page calls the same authenticated API used by clients.</p>
<input id='user' value='user_001'><input id='password' type='password' value='jobrec-demo'>
<textarea id='resume'>Python SQL Docker machine learning</textarea><button onclick='run()'>Recommend</button><pre id='out'></pre>
<script>async function run(){let u=document.querySelector('#user').value,p=document.querySelector('#password').value;
let t=await fetch('/api/token',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({username:u,password:p})}).then(r=>r.json());
let x=await fetch('/api/recommend',{method:'POST',headers:{'content-type':'application/json','authorization':'Bearer '+t.access_token},body:JSON.stringify({user_id:u,resume_text:document.querySelector('#resume').value})}).then(r=>r.json());
document.querySelector('#out').textContent=JSON.stringify(x,null,2)}</script></body></html>"""


def create_app() -> FastAPI:
    return app
