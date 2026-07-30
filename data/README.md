# 数据目录

本目录只存放可重建的半合成输入、被 Git 忽略的本地运行状态和外部数据 staging，
不是版本化真实数据集。

| 产物 | 用途 | 生命周期 |
|---|---|---|
| `mock_data.pkl` | 固定种子 `GraphEntities` 快照 | 可由项目管线重建；仅信任本地 pickle |
| `jobrec_events.sqlite3` | 唯一曝光与反馈关联事件 | 可变运行状态；不提交真实或个人数据 |
| `jobrec_profiles.sqlite3` | 四项个人字段的 AES-GCM 密文 | 被 Git 忽略；共享部署必须使用外部密钥管理 |
| `external/raw/` | 按原许可本地取得的来源数据 | 被 Git 忽略；按来源条款保留或删除 |
| `external/normalized/` | 校验、伪名化后的 JSONL 与 manifest | 被 Git 忽略；不自动进入训练或服务 |

可执行实体契约在 `src/data/models.py`，切分与交互矩阵语义在
`src/data/loader.py`，完整边界见
[`../docs/data-and-evaluation.md`](../docs/data-and-evaluation.md)。

禁止反序列化不可信 pickle；禁止在本目录放置真实简历、生产导出、密钥、口令或未经
治理的个人数据。原始真实数据已经丢失，不得用新生成数据冒充恢复数据。

公开岗位的示例获取入口为 `scripts/fetch_usajobs.py`，来源无关的校验与 lineage
入口为 `scripts/prepare_external_dataset.py`。使用前必须阅读来源许可；完整协议和
命令见 [`../docs/data-and-evaluation.md`](../docs/data-and-evaluation.md#3-外部数据获取与隔离准备)。

事件库启动时会迁移旧 schema；无法关联唯一曝光的旧 feedback 只保留供本地检查，不
进入新统计。本文件不承诺跨版本保留演示统计，需要时可删除被忽略的数据库后重建。

个人档案库只写入 `name/phone/email/address` 四列密文；字段在应用边界解密，不把
明文写入 SQLite。删除接口执行物理行删除，但 SQLite 文件安全擦除、备份清除和密钥
销毁仍属于共享部署的数据生命周期工作。
