"""面向学习者的薄 CLI：每个命令委托给唯一的规范实现。"""

from __future__ import annotations

import argparse
from typing import Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jobrec",
        description="JobRec-KG：离线发布、实验和 API 服务入口",
    )
    subparsers = parser.add_subparsers(dest="command")

    serve = subparsers.add_parser("serve", help="加载已发布 bundle 并启动 API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    build = subparsers.add_parser(
        "build-bundle", help="训练并原子式生成匹配的 checkpoint 与 ModelBundle"
    )
    build.add_argument("--output", default="models/jobrec_bundle.json")
    build.add_argument("--checkpoint", default="models/lightgcn_model.pt")
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--epochs", type=int, default=30)

    experiments = subparsers.add_parser(
        "experiments", help="在共享协议下运行多种子基线与消融"
    )
    experiments.add_argument(
        "--seeds", type=int, nargs="+", default=[11, 19, 23, 31, 42]
    )
    experiments.add_argument("--epochs", type=int, default=15)
    experiments.add_argument("--output", default="results/experiment_summary.json")

    simulate = subparsers.add_parser(
        "simulate-users", help="运行 Persona 判断与位置偏置行为的离线模拟"
    )
    simulate.add_argument(
        "--judge", choices=("deterministic", "llm"), default="deterministic"
    )
    simulate.add_argument("--output", default="results/synthetic_user_simulation.json")
    simulate.add_argument("--bundle", default="models/jobrec_bundle.json")
    simulate.add_argument("--users", type=int, default=20)
    simulate.add_argument("--top-k", type=int, default=10)
    simulate.add_argument("--behavior-seed", type=int, default=20260731)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """解析命令；无子命令时只显示帮助，不执行训练或覆盖产物。"""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "serve":
        import uvicorn

        uvicorn.run("src.api.routes:app", host=args.host, port=args.port)
        return 0

    if args.command == "build-bundle":
        from scripts.build_model_bundle import build

        bundle = build(args.output, args.checkpoint, args.seed, args.epochs)
        print(f"Published {bundle.model_version} to {args.output}")
        return 0

    if args.command == "experiments":
        from src.experiments import run_experiment_suite

        result = run_experiment_suite(
            seeds=args.seeds,
            epochs=args.epochs,
            output_path=args.output,
        )
        for model, metrics in result["aggregate"].items():
            summary = {name: round(value["mean"], 4) for name, value in metrics.items()}
            print(model, summary)
        return 0

    if args.command == "simulate-users":
        from scripts.run_user_simulation import run

        artifact = run(
            output_path=args.output,
            judge_mode=args.judge,
            bundle_path=args.bundle,
            users=args.users,
            top_k=args.top_k,
            behavior_seed=args.behavior_seed,
        )
        print(
            "synthetic_user_simulation",
            {
                "evidence_label": artifact.evidence_label,
                "judge_mode": artifact.summary.judge_mode,
                "proxy_effectiveness_at_k": round(
                    artifact.summary.proxy_effectiveness_at_k, 4
                ),
            },
        )
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
