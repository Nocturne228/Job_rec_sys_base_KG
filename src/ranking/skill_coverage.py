"""等级感知的技能覆盖和差距计算。"""

from __future__ import annotations

from typing import Any, Mapping

LEVELS = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}


def _level(value: object) -> str:
    return str(getattr(value, "value", value)).lower()


class SkillCoverageCalculator:
    def calculate_coverage(
        self,
        user_skills: Mapping[str, object],
        required_skills: Mapping[str, object],
        preferred_skills: Mapping[str, object] | None = None,
    ) -> dict[str, Any]:
        preferred = preferred_skills or {}
        required = self._match(user_skills, required_skills)
        optional = self._match(user_skills, preferred)
        if required_skills and preferred:
            score = 0.7 * required["coverage_ratio"] + 0.3 * optional["coverage_ratio"]
        elif required_skills:
            score = required["coverage_ratio"]
        elif preferred:
            score = optional["coverage_ratio"]
        else:
            score = 1.0
        return {
            "coverage_score": float(score),
            "required": required,
            "preferred": optional,
            "skill_gap": self._gaps(user_skills, required_skills),
        }

    @staticmethod
    def _match(
        user_skills: Mapping[str, object], job_skills: Mapping[str, object]
    ) -> dict[str, Any]:
        matched: list[str] = []
        missing: list[str] = []
        for skill_id, expected in job_skills.items():
            actual = user_skills.get(skill_id)
            if actual is None or LEVELS.get(_level(actual), 0) < LEVELS.get(
                _level(expected), 0
            ):
                missing.append(skill_id)
            else:
                matched.append(skill_id)
        return {
            "coverage_ratio": len(matched) / len(job_skills) if job_skills else 1.0,
            "matched_skills": matched,
            "missing_skills": missing,
        }

    @staticmethod
    def _gaps(
        user_skills: Mapping[str, object], required_skills: Mapping[str, object]
    ) -> list[dict[str, str | None]]:
        gaps: list[dict[str, str | None]] = []
        for skill_id, expected in required_skills.items():
            actual = user_skills.get(skill_id)
            if actual is None or LEVELS.get(_level(actual), 0) < LEVELS.get(
                _level(expected), 0
            ):
                gaps.append(
                    {
                        "skill_id": skill_id,
                        "user_level": _level(actual) if actual is not None else None,
                        "required_level": _level(expected),
                    }
                )
        return gaps
