"""
Hot trends analysis: popular jobs, skills, and education backgrounds.
Implements competition requirement FR-8 (extended feature).
"""
from collections import Counter
from typing import Dict, List, Any


class TrendAnalyzer:
    def __init__(self, jobs=None, users=None, interactions=None):
        self.jobs = jobs or []
        self.users = users or []
        self.interactions = interactions or []

    def hot_jobs(self, top_n: int = 10) -> List[Dict[str, Any]]:
        counter = Counter()
        job_map = {j.id: j for j in self.jobs}
        for inter in self.interactions:
            counter[inter.job_id] += 1
        return [
            {"job_id": jid, "title": job_map.get(jid, type("J", (), {"title": "?"})()).title if jid in job_map else "?",
             "count": cnt}
            for jid, cnt in counter.most_common(top_n)
        ]

    def hot_skills(self, top_n: int = 15) -> List[Dict[str, Any]]:
        counter = Counter()
        for job in self.jobs:
            for sid in job.required_skills:
                counter[sid] += 1
            for sid in job.preferred_skills:
                counter[sid] += 0.5
        return [{"skill_id": sid, "frequency": round(cnt, 1)} for sid, cnt in counter.most_common(top_n)]

    def hot_skills_by_category(self) -> Dict[str, List[str]]:
        groups: Dict[str, list] = {}
        for job in self.jobs:
            for sid in job.required_skills:
                groups.setdefault("required", []).append(sid)
            for sid in job.preferred_skills:
                groups.setdefault("preferred", []).append(sid)
        return {k: [f"{s}({c})" for s, c in Counter(v).most_common(10)] for k, v in groups.items()}

    def education_distribution(self) -> Dict[str, int]:
        counter = Counter()
        for u in self.users:
            if u.education:
                counter[u.education] += 1
        return dict(counter.most_common())

    def experience_distribution(self) -> Dict[str, int]:
        buckets = {"0-1年": 0, "1-3年": 0, "3-5年": 0, "5年+": 0}
        for u in self.users:
            y = u.experience_years
            if y <= 1: buckets["0-1年"] += 1
            elif y <= 3: buckets["1-3年"] += 1
            elif y <= 5: buckets["3-5年"] += 1
            else: buckets["5年+"] += 1
        return buckets

    def full_report(self) -> Dict[str, Any]:
        return {
            "hot_jobs": self.hot_jobs(10),
            "hot_skills": self.hot_skills(15),
            "education_distribution": self.education_distribution(),
            "experience_distribution": self.experience_distribution(),
        }
