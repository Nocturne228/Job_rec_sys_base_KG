"""面向学习者的薄 CLI：每个命令委托给唯一的规范实现。"""

from __future__ import annotations

import argparse
from typing import Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jobrec",
        description="JobRec-Feed：离线发布、实验和 API 服务入口",
    )
    subparsers = parser.add_subparsers(dest="command")

    serve = subparsers.add_parser("serve", help="加载已发布 bundle 并启动 API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    build = subparsers.add_parser(
        "build-bundle", help="训练并原子式生成匹配的 checkpoint 与 ModelBundle"
    )
    build.add_argument("--output", default="models/jobrec_bundle.json")
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--epochs", type=int, default=20)

    experiments = subparsers.add_parser(
        "experiments", help="在共享协议下运行多种子基线与消融"
    )
    experiments.add_argument(
        "--seeds", type=int, nargs="+", default=[11, 19, 23, 31, 42]
    )
    experiments.add_argument("--epochs", type=int, default=15)
    experiments.add_argument("--output", default="results/experiment_summary.json")

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

        bundle = build(args.output, args.seed, args.epochs)
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

    parser.error(f"Unknown command: {args.command}")
    return 2
