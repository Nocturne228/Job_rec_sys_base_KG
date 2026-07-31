# 结果产物目录

本目录保存可复核检查生成的机器可读证据。每份结果只对自身协议负责，不是生产基准。

| 产物 | 生成入口 | 含义 |
|---|---|---|
| `experiment_summary.json` | `uv run python -m scripts.run_experiments` | 五种子基线和消融的逐次值、均值与总体标准差 |
| `coverage.json` | pytest/coverage | 记录运行的源码覆盖率明细 |
| `load_test_summary.json` | `uv run python -m scripts.load_test` | 本地 API 小规模负载冒烟 |
| `synthetic_user_simulation.json` | `uv run python -m scripts.run_user_simulation` | Persona 潜在匹配与位置偏置行为的聚合模拟；不是真实用户效果 |

新实验产物必须记录数据/生成器版本、规模、切分、候选池、种子、模型配置、环境、
指标定义和生成时间。不得手工修改结果数字。解释和限制由
[`../docs/data-and-evaluation.md`](../docs/data-and-evaluation.md) 统一维护。
