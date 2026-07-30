#!/usr/bin/env python3
"""JobRec-KG 的安全命令行入口。

具体实现位于 ``src.cli``。根入口只保留兼容性，不再维护第二套训练与演示流程。
"""

from src.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
