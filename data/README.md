# 数据目录

本目录只存放可重建的半合成输入和被 Git 忽略的本地运行状态，不是版本化真实数据集。

| 产物 | 用途 | 生命周期 |
|---|---|---|
| `mock_data.pkl` | 固定种子 `GraphEntities` 快照 | 可由项目管线重建；仅信任本地 pickle |
| `jobrec_events.sqlite3` | 本地曝光与反馈事件 | 可变运行状态；不提交真实或个人数据 |

可执行实体契约在 `src/data/models.py`，切分与交互矩阵语义在
`src/data/loader.py`，完整边界见
[`../docs/data-and-evaluation.md`](../docs/data-and-evaluation.md)。

禁止反序列化不可信 pickle；禁止在本目录放置真实简历、生产导出、密钥、口令或未经
治理的个人数据。原始真实数据已经丢失，不得用新生成数据冒充恢复数据。
