#!/usr/bin/env python3
"""运行合成用户离线评估并生成可追溯 JSON 产物。"""

from __future__ import annotations

import argparse
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from src.generation import OpenAICompatibleLLM
from src.simulation import (
    DeterministicPersonaJudge,
    PositionAwareBehaviorModel,
    PromptedLLMPersonaJudge,
    run_simulation,
)
from src.simulation.judge import PersonaJudge
from src.simulation.offline import build_published_hybrid_slates
from src.simulation.schemas import SimulationArtifact

LITERATURE = [
    {
        "name": "RecSim: A Configurable Simulation Platform for Recommender Systems",
        "url": (
            "https://research.google/pubs/recsim-a-configurable-simulation-platform-"
            "for-recommender-systems/"
        ),
        "use": "将用户偏好、状态与选择/响应行为分开建模",
    },
    {
        "name": "Position Bias Estimation for Unbiased Learning to Rank",
        "url": (
            "https://research.google/pubs/position-bias-estimation-for-unbiased-"
            "learning-to-rank-in-personal-search/"
        ),
        "use": "位置观察偏差与相关性分离",
    },
    {
        "name": "Large Language Models as Simulated Economic Agents",
        "url": "https://aclanthology.org/2024.naacl-long.83/",
        "use": "LLM 用户模拟评估协议与模型/提示词敏感性边界",
    },
    {
        "name": "LLM-Powered User Simulator for Recommender System",
        "url": "https://ojs.aaai.org/index.php/AAAI/article/view/33456",
        "use": "LLM 推理与统计参与度模型组合",
    },
    {
        "name": "The Realism Gap in LLM-Based User Simulators",
        "url": "https://aclanthology.org/2026.eacl-long.244/",
        "use": "记录 prompt-only 模拟与真实用户之间的现实差距",
    },
]


def run(
    output_path: str = "results/synthetic_user_simulation.json",
    judge_mode: str = "deterministic",
    bundle_path: str = "models/jobrec_bundle.json",
    users: int = 20,
    top_k: int = 10,
    behavior_seed: int = 20260731,
    include_records: bool = False,
) -> SimulationArtifact:
    slates, bundle = build_published_hybrid_slates(
        bundle_path=bundle_path,
        user_limit=users,
        top_k=top_k,
    )
    judge: PersonaJudge
    if judge_mode == "llm":
        judge = PromptedLLMPersonaJudge(OpenAICompatibleLLM())
    elif judge_mode == "deterministic":
        judge = DeterministicPersonaJudge()
    else:
        raise ValueError("judge_mode must be deterministic or llm")

    summary = run_simulation(
        slates,
        judge=judge,
        behavior_model=PositionAwareBehaviorModel(seed=behavior_seed),
        top_k=top_k,
    )
    if not include_records:
        summary.records = []
    artifact = SimulationArtifact(
        generated_at=datetime.now(timezone.utc).isoformat(),
        evidence_label=(
            "已验证"
            if judge_mode == "deterministic"
            else "已验证" if summary.judge_mode == "llm" else "环境受限"
        ),
        run_metadata={
            "data_kind": "fixed-seed semi-synthetic",
            "data_seed": bundle.data_seed,
            "behavior_seed": behavior_seed,
            "model_version": bundle.model_version,
            "ranking": "published hybrid LightGCN + feature-hashing semantic + skill coverage",
            "seen_item_policy": "exclude bundle training interactions before ranking",
            "requested_judge_mode": judge_mode,
            "include_records": include_records,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "command": (
                "uv run python -m scripts.run_user_simulation "
                f"--judge {judge_mode} --users {users} --top-k {top_k}"
            ),
        },
        metric_definitions={
            "proxy_effectiveness_at_k": (
                "有效判断数/评估岗位数；有效=硬约束通过且 skill_fit>=4 "
                "且 willingness_to_consider>=4；仅全 LLM 模式可称 "
                "LLMProxyEffectiveness@K"
            ),
            "simulated_ctr_at_k": "合成点击数/合成曝光数",
            "simulated_feedback_effectiveness": (
                "合成满意反馈数/有合成反馈的曝光数；不等同真实用户调查"
            ),
        },
        literature=LITERATURE,
        summary=summary,
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    return artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/synthetic_user_simulation.json")
    parser.add_argument(
        "--judge", choices=("deterministic", "llm"), default="deterministic"
    )
    parser.add_argument("--bundle", default="models/jobrec_bundle.json")
    parser.add_argument("--users", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--behavior-seed", type=int, default=20260731)
    parser.add_argument(
        "--include-records",
        action="store_true",
        help="在结果中保留逐曝光明细；默认只保存聚合证据",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact = run(
        output_path=args.output,
        judge_mode=args.judge,
        bundle_path=args.bundle,
        users=args.users,
        top_k=args.top_k,
        behavior_seed=args.behavior_seed,
        include_records=args.include_records,
    )
    print(
        {
            "evidence_label": artifact.evidence_label,
            "judge_mode": artifact.summary.judge_mode,
            "personas": artifact.summary.personas,
            "impressions": artifact.summary.impressions,
            "proxy_effectiveness_at_k": round(
                artifact.summary.proxy_effectiveness_at_k, 4
            ),
            "simulated_ctr_at_k": round(artifact.summary.simulated_ctr_at_k, 4),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
