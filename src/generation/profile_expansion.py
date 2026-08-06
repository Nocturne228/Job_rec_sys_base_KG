"""将简历和近期岗位转为可验证兴趣标签，并安全降级。"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from typing import Protocol, Sequence

from pydantic import BaseModel, Field, model_validator

from src.data import Skill


class InterestEvidence(BaseModel):
    term: str = Field(min_length=1, max_length=80)
    source_text: str = Field(min_length=1, max_length=240)


class InterestProfile(BaseModel):
    interests: list[str] = Field(default_factory=list, max_length=12)
    negative_preferences: list[str] = Field(default_factory=list, max_length=8)
    expanded_queries: list[str] = Field(default_factory=list, max_length=12)
    evidence: list[InterestEvidence] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def require_evidence_for_terms(self) -> "InterestProfile":
        supported = {row.term.casefold() for row in self.evidence}
        claimed = {term.casefold() for term in self.interests + self.expanded_queries}
        if not claimed.issubset(supported):
            raise ValueError("every generated interest/query needs input evidence")
        return self

    def augmented_text(self, original: str) -> str:
        additions = list(dict.fromkeys(self.interests + self.expanded_queries))
        return " ".join([original.strip(), *additions]).strip()


@dataclass(frozen=True)
class ExpansionResult:
    profile: InterestProfile
    mode: str


class ProfileExpander(Protocol):
    def expand(
        self, resume_text: str, recent_job_texts: Sequence[str]
    ) -> InterestProfile: ...


class DeterministicProfileExpander:
    """只复用输入中实际出现的技能词，作为离线兜底而非生成模型。"""

    def __init__(self, skills: Sequence[Skill]) -> None:
        self.skills = list(skills)

    def expand(
        self, resume_text: str, recent_job_texts: Sequence[str]
    ) -> InterestProfile:
        sources = [resume_text, *recent_job_texts]
        evidence: list[InterestEvidence] = []
        interests: list[str] = []
        for skill in self.skills:
            labels = (skill.id, skill.name)
            source = next(
                (
                    text
                    for text in sources
                    if any(
                        re.search(
                            rf"(?<!\w){re.escape(label.casefold())}(?!\w)",
                            text.casefold(),
                        )
                        for label in labels
                    )
                ),
                None,
            )
            if source is not None:
                interests.append(skill.name)
                evidence.append(
                    InterestEvidence(term=skill.name, source_text=source[:240])
                )
        return InterestProfile(
            interests=interests[:12],
            expanded_queries=interests[:12],
            evidence=evidence[:20],
        )


class OpenAICompatibleProfileExpander:
    """无额外 SDK 的可选结构化 LLM 适配器。"""

    def __init__(self, endpoint: str, model: str, api_key: str, timeout: float = 8.0):
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def expand(
        self, resume_text: str, recent_job_texts: Sequence[str]
    ) -> InterestProfile:
        source = json.dumps(
            {"resume": resume_text, "recent_jobs": list(recent_job_texts)},
            ensure_ascii=False,
        )
        prompt = (
            "Extract job interests as strict JSON with keys interests, "
            "negative_preferences, expanded_queries, evidence. Every interest or "
            "query must have evidence {term, source_text} copied from the input; "
            "do not infer unsupported skills. Input: " + source
        )
        payload = json.dumps(
            {
                "model": self.model,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
        return InterestProfile.model_validate_json(content)


class ResilientProfileExpander:
    def __init__(
        self,
        fallback: DeterministicProfileExpander,
        primary: ProfileExpander | None = None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback

    def expand(
        self, resume_text: str, recent_job_texts: Sequence[str]
    ) -> ExpansionResult:
        if self.primary is not None:
            try:
                return ExpansionResult(
                    self.primary.expand(resume_text, recent_job_texts),
                    "llm_structured",
                )
            except Exception:
                # 外部服务不可用或输出不合约时，返回可核对的输入词项。
                pass
        return ExpansionResult(
            self.fallback.expand(resume_text, recent_job_texts),
            "deterministic_fallback",
        )
