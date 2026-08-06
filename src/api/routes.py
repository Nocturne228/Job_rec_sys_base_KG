"""面试演示所需的最小 FastAPI 服务。"""

from __future__ import annotations

import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, model_validator

from src.data import DataLoader, GraphEntities, GraphLoader, generate_mock_data
from src.generation import (
    DeterministicProfileExpander,
    ExpansionResult,
    OpenAICompatibleProfileExpander,
    ResilientProfileExpander,
)
from src.metrics import EventStore
from src.models import ModelBundle, data_fingerprint
from src.ranking import (
    DiversityReranker,
    FeatureBuilder,
    PointwiseRanker,
    SkillCoverageCalculator,
)
from src.recall import LightGCN, TextRecall, merge_recall_routes
from src.recall.lightgcn import prepare_adj_matrix


class RecommendationRequest(BaseModel):
    user_id: str | None = None
    resume_text: str = Field(default="", max_length=20_000)
    title_query: str | None = Field(default=None, max_length=100)
    top_k: int = Field(default=10, ge=1, le=20)
    use_interest_expansion: bool = True

    @model_validator(mode="after")
    def require_identity_or_resume(self) -> "RecommendationRequest":
        if not self.user_id and not self.resume_text.strip():
            raise ValueError("provide user_id or resume_text")
        return self


class Recommendation(BaseModel):
    request_id: str
    impression_id: int
    subject_id: str
    job_id: str
    title: str
    score: float
    contributions: dict[str, float]
    recall_sources: list[str]
    retrieval_mode: str
    generation_mode: str
    expanded_interests: list[str]
    model_version: str


class CompetencyRequest(BaseModel):
    job_id: str
    user_id: str | None = None
    resume_text: str = Field(default="", max_length=20_000)

    @model_validator(mode="after")
    def require_identity_or_resume(self) -> "CompetencyRequest":
        if not self.user_id and not self.resume_text.strip():
            raise ValueError("provide user_id or resume_text")
        return self


class CompetencyReport(BaseModel):
    job_id: str
    overall_match: float
    gaps: list[dict[str, Any]]
    learning_paths: list[dict[str, Any]]
    evidence_source: str


class FeedbackRequest(BaseModel):
    impression_id: int = Field(gt=0)
    subject_id: str
    job_id: str
    clicked: bool = False
    dwell_seconds: float = Field(default=0.0, ge=0.0, le=86_400.0)
    saved: bool = False
    applied: bool = False
    satisfied: bool | None = None

    @model_validator(mode="after")
    def require_observable_outcome(self) -> "FeedbackRequest":
        if not any(
            (
                self.clicked,
                self.dwell_seconds > 0,
                self.saved,
                self.applied,
                self.satisfied is not None,
            )
        ):
            raise ValueError("provide at least one observable feedback outcome")
        return self


class FeedbackResponse(BaseModel):
    status: str
    feedback_count: int
    satisfied_count: int
    satisfaction_rate: float


@dataclass
class Pipeline:
    bundle: ModelBundle
    data: GraphEntities
    loader: DataLoader
    user_embeddings: torch.Tensor
    item_embeddings: torch.Tensor
    text: TextRecall
    skills: SkillCoverageCalculator
    features: FeatureBuilder
    ranker: PointwiseRanker
    reranker: DiversityReranker
    expander: ResilientProfileExpander
    graph: GraphLoader
    events: EventStore


def _load_pipeline() -> Pipeline:
    bundle_path = os.environ.get("JOBREC_BUNDLE_PATH", "models/jobrec_bundle.json")
    bundle = ModelBundle.load(bundle_path)
    data = generate_mock_data(20, 50, seed=bundle.data_seed)
    if data_fingerprint(data) != bundle.data_sha256:
        raise RuntimeError("bundle data fingerprint does not match generated data")
    loader = DataLoader(data, random_seed=bundle.data_seed)
    if (
        loader.user_id_to_idx != bundle.user_id_to_idx
        or loader.job_id_to_idx != bundle.job_id_to_idx
    ):
        raise RuntimeError("bundle ID mappings do not match generated data")
    model = LightGCN.load(bundle.checkpoint_path)
    if model.n_users != loader.n_users or model.n_items != loader.n_jobs:
        raise RuntimeError("checkpoint dimensions do not match bundle mappings")
    model.eval()
    with torch.no_grad():
        users, items = model(prepare_adj_matrix(loader.get_sparse_graph()))
    text = TextRecall(n_features=int(bundle.text_config["n_features"]))
    for job in data.jobs:
        text.add_job(job.id, job.description)
    features = FeatureBuilder(data, loader, text, users, items)
    llm_endpoint = os.environ.get("JOBREC_LLM_ENDPOINT")
    llm_model = os.environ.get("JOBREC_LLM_MODEL")
    llm_key = os.environ.get("JOBREC_LLM_API_KEY")
    primary = (
        OpenAICompatibleProfileExpander(llm_endpoint, llm_model, llm_key)
        if llm_endpoint and llm_model and llm_key
        else None
    )
    return Pipeline(
        bundle=bundle,
        data=data,
        loader=loader,
        user_embeddings=users,
        item_embeddings=items,
        text=text,
        skills=SkillCoverageCalculator(),
        features=features,
        ranker=PointwiseRanker.from_config(bundle.ranking_model),
        reranker=DiversityReranker(**bundle.reranking_config),
        expander=ResilientProfileExpander(
            DeterministicProfileExpander(data.skills), primary
        ),
        graph=GraphLoader(data),
        events=EventStore(
            os.environ.get("JOBREC_EVENT_DB", "data/jobrec_events.sqlite3")
        ),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = _load_pipeline()
    yield


app = FastAPI(title="JobRec-Feed Interview Demo", version="4.0", lifespan=lifespan)


def get_pipeline(request: Request) -> Pipeline:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="model bundle is not ready")
    return pipeline


def _extract_skills(text: str, data: GraphEntities) -> dict[str, str]:
    normalized = text.casefold()
    return {
        skill.id: "beginner"
        for skill in data.skills
        if any(
            re.search(rf"(?<!\w){re.escape(label.casefold())}(?!\w)", normalized)
            for label in (skill.id, skill.name)
        )
    }


def _profile(
    pipeline: Pipeline, user_id: str | None, resume_text: str
) -> tuple[str, dict[str, str], str]:
    users = {user.id: user for user in pipeline.data.users}
    known = bool(user_id and user_id in pipeline.bundle.user_id_to_idx)
    resume = resume_text.strip() or (
        (users[user_id].resume_text or "") if known and user_id is not None else ""
    )
    if known and not resume_text.strip() and user_id is not None:
        skills = {
            skill_id: str(getattr(level, "value", level))
            for skill_id, level in users[user_id].skills.items()
        }
    else:
        skills = _extract_skills(resume, pipeline.data)
    return resume, skills, "known_hybrid" if known else "cold_start_text_skill"


def _recent_job_texts(pipeline: Pipeline, user_id: str, limit: int = 5) -> list[str]:
    seen = set(pipeline.bundle.train_items_by_user.get(user_id, []))
    jobs = {job.id: job for job in pipeline.data.jobs}
    recent = sorted(
        (
            interaction
            for interaction in pipeline.data.interactions
            if interaction.user_id == user_id and interaction.job_id in seen
        ),
        key=lambda row: row.timestamp,
        reverse=True,
    )[:limit]
    return [jobs[row.job_id].description for row in recent]


@app.get("/health/live")
def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
def ready(request: Request) -> dict[str, str]:
    pipeline = get_pipeline(request)
    return {"status": "ready", "model_version": pipeline.bundle.model_version}


@app.get("/api/model")
def model_info(request: Request) -> dict[str, Any]:
    bundle = get_pipeline(request).bundle
    return {
        "model_version": bundle.model_version,
        "schema_version": bundle.schema_version,
        "created_at": bundle.created_at,
        "data_sha256": bundle.data_sha256,
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "serving_sha256": bundle.serving_sha256,
        "training_config": bundle.training_config,
        "ranking_kind": bundle.ranking_model["kind"],
    }


@app.post("/api/recommend", response_model=list[Recommendation])
def recommend(body: RecommendationRequest, request: Request) -> list[Recommendation]:
    pipeline = get_pipeline(request)
    resume, user_skills, mode = _profile(pipeline, body.user_id, body.resume_text)
    jobs = pipeline.data.jobs
    if body.title_query:
        query = body.title_query.casefold().strip()
        jobs = [job for job in jobs if query in job.title.casefold()]
        if not jobs:
            raise HTTPException(status_code=404, detail="no job matches title_query")
    known = mode == "known_hybrid"
    if known and body.user_id is not None:
        seen = set(pipeline.bundle.train_items_by_user.get(body.user_id, []))
        jobs = [job for job in jobs if job.id not in seen]
    recent_job_texts: list[str] = []
    if known and body.user_id is not None:
        recent_job_texts = _recent_job_texts(pipeline, body.user_id)
    if body.use_interest_expansion:
        expansion = pipeline.expander.expand(resume, recent_job_texts)
        ranking_text = expansion.profile.augmented_text(resume)
    else:
        expansion = pipeline.expander.expand("", [])
        ranking_text = resume
        expansion = ExpansionResult(expansion.profile, "disabled")
    feature_by_job = pipeline.features.build(
        user_id=body.user_id,
        resume_text=ranking_text,
        user_skills=user_skills,
        jobs=jobs,
        known_user=known,
    )
    candidates = merge_recall_routes(
        pipeline.features.recall_routes(feature_by_job, known),
        [job.id for job in jobs],
        per_route_k=max(10, min(30, body.top_k * 3)),
    )
    candidate_by_id = {candidate.job_id: candidate for candidate in candidates}
    jobs_by_id = {job.id: job for job in jobs}
    candidate_jobs = [jobs_by_id[candidate.job_id] for candidate in candidates]
    candidate_features = [feature_by_job[job.id] for job in candidate_jobs]
    subject_id = body.user_id or f"cold-{uuid4().hex[:8]}"
    request_id = uuid4().hex
    response: list[Recommendation] = []
    ranked = pipeline.ranker.rank_with_explanations(candidate_features)
    for index, score, contributions in pipeline.reranker.rerank(
        ranked, candidate_jobs, body.top_k
    ):
        job = candidate_jobs[index]
        recall_sources = candidate_by_id[job.id].sources
        impression_id = pipeline.events.record_impression(
            subject_id,
            job.id,
            pipeline.bundle.model_version,
            {
                "request_id": request_id,
                "rank": len(response) + 1,
                "mode": mode,
                "recall_sources": recall_sources,
                "generation_mode": expansion.mode,
            },
        )
        response.append(
            Recommendation(
                request_id=request_id,
                impression_id=impression_id,
                subject_id=subject_id,
                job_id=job.id,
                title=job.title,
                score=round(score, 6),
                contributions={
                    key: round(value, 6) for key, value in contributions.items()
                },
                recall_sources=recall_sources,
                retrieval_mode=mode,
                generation_mode=expansion.mode,
                expanded_interests=expansion.profile.interests,
                model_version=pipeline.bundle.model_version,
            )
        )
    return response


@app.post("/api/competency", response_model=CompetencyReport)
def competency(body: CompetencyRequest, request: Request) -> CompetencyReport:
    pipeline = get_pipeline(request)
    job = next((item for item in pipeline.data.jobs if item.id == body.job_id), None)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job_id")
    _, user_skills, _ = _profile(pipeline, body.user_id, body.resume_text)
    coverage = pipeline.skills.calculate_coverage(
        user_skills, job.required_skills, job.preferred_skills
    )
    paths = pipeline.graph.find_paths_for_skills(user_skills, job.id)
    path_targets = {item["skills"][-1] for item in paths}
    for gap in coverage["skill_gap"]:
        if gap["skill_id"] not in path_targets:
            paths.append(
                {
                    "skills": [gap["skill_id"]],
                    "evidence": [],
                    "note": "direct learning target; no prerequisite path in demo graph",
                }
            )
    return CompetencyReport(
        job_id=job.id,
        overall_match=round(float(coverage["coverage_score"]), 6),
        gaps=coverage["skill_gap"],
        learning_paths=paths[:8],
        evidence_source="typed_in_memory_graph",
    )


@app.post("/api/feedback", response_model=FeedbackResponse)
def feedback(body: FeedbackRequest, request: Request) -> FeedbackResponse:
    pipeline = get_pipeline(request)
    try:
        pipeline.events.record_feedback(
            body.impression_id,
            body.subject_id,
            body.job_id,
            pipeline.bundle.model_version,
            body.satisfied,
            clicked=body.clicked,
            dwell_seconds=body.dwell_seconds,
            saved=body.saved,
            applied=body.applied,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    stats = pipeline.events.effectiveness()
    return FeedbackResponse(
        status="recorded",
        feedback_count=stats["n_total"],
        satisfied_count=stats["n_satisfied"],
        satisfaction_rate=stats["effectiveness"],
    )


@app.get("/demo", response_class=HTMLResponse)
def demo() -> str:
    return """<!doctype html><meta charset='utf-8'><title>JobRec-Feed</title>
<style>body{font:16px system-ui;max-width:900px;margin:40px auto}textarea,input{width:100%;padding:8px;margin:5px}button{padding:10px}pre{background:#f4f5f7;padding:16px}</style>
<h1>JobRec-Feed 面试演示</h1><p>所有数据均为固定种子半合成数据。</p>
<input id='uid' value='user_001'><textarea id='resume'>Python SQL Docker</textarea>
<button onclick='run()'>推荐</button><pre id='out'></pre>
<script>async function run(){const body={user_id:uid.value,resume_text:resume.value};const r=await fetch('/api/recommend',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)});out.textContent=JSON.stringify(await r.json(),null,2)}</script>"""
