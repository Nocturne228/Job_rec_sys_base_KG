"""Switchable LLM adapters and strict output-schema validation."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError


class LearningStep(BaseModel):
    skill_id: str
    current_level: Optional[str] = None
    target_level: str
    resources: List[str] = Field(default_factory=list)
    estimated_time: str
    priority: Literal["high", "medium", "low"]


class CareerAdvice(BaseModel):
    summary: str
    critical_skills: List[str]
    learning_path: List[LearningStep]
    timeline_months: float = Field(ge=0.0, le=60.0)
    confidence_score: float = Field(ge=0.0, le=1.0)


def validate_advice(raw: str) -> CareerAdvice:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        if text.startswith("json"):
            text = text[4:].lstrip()
    return CareerAdvice.model_validate_json(text)


def fallback_advice(skill_gaps: Dict[str, Dict[str, Any]]) -> CareerAdvice:
    critical = list(skill_gaps)[:5]
    steps = [
        LearningStep(
            skill_id=skill,
            current_level=levels.get("user_level"),
            target_level=str(levels.get("required_level") or "intermediate"),
            resources=[
                f"Official {skill} documentation",
                f"Build one portfolio project using {skill}",
            ],
            estimated_time="4 weeks",
            priority="high" if index < 2 else "medium",
        )
        for index, (skill, levels) in enumerate(list(skill_gaps.items())[:5])
    ]
    return CareerAdvice(
        summary="Follow the evidence-backed learning sequence for the identified skill gaps.",
        critical_skills=critical,
        learning_path=steps,
        timeline_months=max(1.0, len(steps) * 0.75),
        confidence_score=0.55,
    )


class OpenAICompatibleLLM:
    """Minimal adapter for OpenAI-compatible chat-completions endpoints."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 20.0,
    ):
        self.endpoint = endpoint or os.environ.get("JOBREC_LLM_ENDPOINT", "")
        self.api_key = api_key or os.environ.get("JOBREC_LLM_API_KEY", "")
        self.model = model or os.environ.get("JOBREC_LLM_MODEL", "qwen2.5")
        self.timeout = timeout

    def generate(
        self, prompt: str, temperature: float = 0.3, max_tokens: int = 1000
    ) -> Dict[str, Any]:
        if not self.endpoint or not self.api_key:
            raise RuntimeError(
                "JOBREC_LLM_ENDPOINT and JOBREC_LLM_API_KEY are required"
            )
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        content = payload["choices"][0]["message"]["content"]
        return {
            "model": self.model,
            "response": content,
            "usage": payload.get("usage", {}),
        }
