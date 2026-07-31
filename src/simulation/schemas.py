"""合成用户评估的显式数据契约。"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class Persona(BaseModel):
    """最小化、非识别性的合成求职者画像。"""

    persona_id: str = Field(min_length=1)
    education: Optional[str] = None
    experience_years: float = Field(default=0.0, ge=0.0)
    skills: Dict[str, str] = Field(default_factory=dict)
    target_titles: List[str] = Field(default_factory=list)
    excluded_titles: List[str] = Field(default_factory=list)
    preferences: List[str] = Field(default_factory=list)


class PersonaJobJudgment(BaseModel):
    """LLM 或确定性基线返回的岗位潜在匹配判断。

    ``effective`` 由代码按固定规则计算，避免模型直接迎合目标通过率。
    """

    hard_constraints_pass: bool
    skill_fit: int = Field(ge=1, le=5)
    role_interest_fit: int = Field(ge=1, le=5)
    growth_fit: int = Field(ge=1, le=5)
    willingness_to_consider: int = Field(ge=1, le=5)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: List[str] = Field(default_factory=list, max_length=5)
    effective: bool = False
    latent_relevance: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def derive_outcomes(self) -> "PersonaJobJudgment":
        self.effective = (
            self.hard_constraints_pass
            and self.skill_fit >= 4
            and self.willingness_to_consider >= 4
        )
        if not self.hard_constraints_pass:
            self.latent_relevance = 0.0
        else:
            weighted = (
                0.45 * self.skill_fit
                + 0.25 * self.role_interest_fit
                + 0.15 * self.growth_fit
                + 0.15 * self.willingness_to_consider
            )
            self.latent_relevance = float((weighted - 1.0) / 4.0)
        return self


class JudgmentEnvelope(BaseModel):
    judgment: PersonaJobJudgment
    source: Literal["llm", "deterministic", "deterministic_fallback"]
    prompt_version: str
    model: Optional[str] = None
    error: Optional[str] = None


class BehaviorOutcome(BaseModel):
    rank: int = Field(ge=1)
    examination_probability: float = Field(ge=0.0, le=1.0)
    attraction_probability: float = Field(ge=0.0, le=1.0)
    marginal_click_probability: float = Field(ge=0.0, le=1.0)
    examined: bool
    clicked: bool
    saved: bool
    applied: bool
    feedback_given: bool
    satisfied: bool


class SimulationRecord(BaseModel):
    persona_id: str
    job_id: str
    rank: int
    judgment: JudgmentEnvelope
    behavior: BehaviorOutcome


class SimulationSummary(BaseModel):
    schema_version: int = 1
    protocol: str
    prompt_version: str
    judge_mode: str
    top_k: int = Field(ge=1)
    personas: int = Field(ge=0)
    impressions: int = Field(ge=0)
    effective_judgments: int = Field(ge=0)
    proxy_effectiveness_at_k: float = Field(ge=0.0, le=1.0)
    simulated_ctr_at_k: float = Field(ge=0.0, le=1.0)
    simulated_save_rate_at_k: float = Field(ge=0.0, le=1.0)
    simulated_apply_rate_at_k: float = Field(ge=0.0, le=1.0)
    simulated_feedback_count: int = Field(ge=0)
    simulated_feedback_effectiveness: Optional[float] = Field(
        default=None, ge=0.0, le=1.0
    )
    source_counts: Dict[str, int] = Field(default_factory=dict)
    records: List[SimulationRecord] = Field(default_factory=list)


class SimulationArtifact(BaseModel):
    generated_at: str
    evidence_label: Literal["已验证", "已实现", "环境受限"]
    run_metadata: Dict[str, Any]
    metric_definitions: Dict[str, str]
    literature: List[Dict[str, str]]
    summary: SimulationSummary
