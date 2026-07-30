#!/usr/bin/env python3
"""通过 USAJOBS 官方 API 获取岗位并写成 ExternalJob JSONL。

API key 和注册邮箱只从环境变量读取。输出保留来源 URL，但不包含求职者个人数据。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API_URL = "https://data.usajobs.gov/api/search"


def _text(value: Any) -> str:
    if isinstance(value, list):
        parts = [_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    return str(value).strip() if value is not None else ""


def normalize_items(payload: Dict[str, Any]) -> Iterable[dict]:
    """把 USAJOBS SearchResult 转成项目的来源无关岗位契约。"""
    items = payload.get("SearchResult", {}).get("SearchResultItems", [])
    for item in items:
        descriptor = item.get("MatchedObjectDescriptor", {})
        details = descriptor.get("UserArea", {}).get("Details", {})
        description_parts = [
            details.get("JobSummary"),
            details.get("MajorDuties"),
            details.get("QualificationSummary"),
        ]
        description = "\n\n".join(_text(part) for part in description_parts if part)
        job_id = str(descriptor.get("PositionID", "")).strip()
        title = str(descriptor.get("PositionTitle", "")).strip()
        company = str(descriptor.get("OrganizationName", "")).strip()
        if not all((job_id, title, company, description)):
            continue
        yield {
            "job_id": f"usajobs:{job_id}",
            "title": title,
            "company": company,
            "description": description,
            "required_skills": {},
            "preferred_skills": {},
            "source_url": descriptor.get("PositionURI"),
        }


def fetch(keyword: str, location: str | None, pages: int) -> Iterable[dict]:
    email = os.environ.get("USAJOBS_EMAIL")
    api_key = os.environ.get("USAJOBS_API_KEY")
    if not email or not api_key:
        raise RuntimeError("USAJOBS_EMAIL and USAJOBS_API_KEY are required")
    headers = {
        "Host": "data.usajobs.gov",
        "User-Agent": email,
        "Authorization-Key": api_key,
    }
    for page in range(1, pages + 1):
        params = {"Keyword": keyword, "Page": page, "ResultsPerPage": 100}
        if location:
            params["LocationName"] = location
        request = Request(f"{API_URL}?{urlencode(params)}", headers=headers)
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        yield from normalize_items(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", default="software data")
    parser.add_argument("--location")
    parser.add_argument("--pages", type=int, default=1)
    parser.add_argument("--output", default="data/external/raw/usajobs-jobs.jsonl")
    args = parser.parse_args()
    if args.pages < 1 or args.pages > 10:
        parser.error("--pages must be between 1 and 10")

    jobs_by_id = {
        job["job_id"]: job for job in fetch(args.keyword, args.location, args.pages)
    }
    jobs = list(jobs_by_id.values())
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in jobs) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(jobs)} jobs to {target}")


if __name__ == "__main__":
    main()
