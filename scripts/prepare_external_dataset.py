#!/usr/bin/env python3
"""校验外部 JSONL，生成不进入 Git 的标准化 staging snapshot。"""

from __future__ import annotations

import argparse
import os

from src.data.external import prepare_external_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True, help="ExternalJob JSONL")
    parser.add_argument("--interactions", help="ExternalInteraction JSONL")
    parser.add_argument("--output-dir", default="data/external/normalized")
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--license-name", required=True)
    parser.add_argument("--license-url", required=True)
    parser.add_argument("--retrieved-at", required=True, help="ISO date or timestamp")
    args = parser.parse_args()

    manifest = prepare_external_snapshot(
        args.jobs,
        args.output_dir,
        source_name=args.source_name,
        source_url=args.source_url,
        license_name=args.license_name,
        license_url=args.license_url,
        retrieved_at=args.retrieved_at,
        interactions_path=args.interactions,
        pseudonymization_secret=os.environ.get("JOBREC_IMPORT_PSEUDONYM_KEY"),
    )
    print(manifest.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
