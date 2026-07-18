# 数据、实验协议与验证结果

> 状态：数据语义和数值证据的唯一详细说明。机器可读结果以 `../results/` 为准。

## 1. 数据事实与适用边界

原始真实数据已经丢失，仓库没有可恢复的真实简历、岗位、投递、面试或线上曝光日志。
当前默认数据由 `src/data/generator.py` 按固定种子生成，属于**半合成开发数据**。

它可以支持：

- 构造稳定的实体、关系和交互以运行完整链路；
- 验证切分、负采样、seen masking、指标和模型发布契约；
- 比较同一生成分布、候选池和种子策略下的基线与消融；
- 演示 API、解释、反馈持久化和外部适配器边界。

它不能支持：

- 真实求职者的推荐有效性、投递率、面试率或满意度结论；
- 真实市场分布、冷启动难度、季节性和反馈偏差估计；
- 受保护群体公平性、隐私合规或真实个人数据安全结论；
- 生产数据规模、吞吐、可用性、成本或服务等级结论。

旧材料中的真实规模或业务指标统一视为**历史未验证**，不进入当前实验基线。

## 2. 数据域与可执行契约

实体字段由 `src/data/models.py` 的 Pydantic 模型定义；本文说明语义，不复制完整
schema。

| 数据域 | 当前来源 | 主要用途 | 若未来获得合法新数据时的缺口 |
|---|---|---|---|
| 用户/简历 | 固定种子生成 | 文本召回、技能覆盖、能力评估 | 授权采集、解析/NER、身份和保留治理 |
| 岗位 | 固定种子生成 | 文本召回、技能匹配、反向匹配 | 合法 feed、时效、去重、地区/币种/类型 |
| 技能 | 维护中的演示词表 | 用户/岗位画像、GAT、解释 | 版本化本体、别名消歧和专家治理 |
| 技能关系 | 人工定义类型边 | 学习路径证据 | 来源审查、有效期、版本和复核状态 |
| 交互 | 兼容度驱动模拟 | LightGCN、离线评估、趋势 | 唯一事件、曝光 ID、时间语义、去重和偏差处理 |
| 在线事件 | API 本地写入 | 曝光后反馈诊断 | 持久事件平台、同意、删除、实验和迟到事件规则 |

### 2.1 核心实体

- `Skill`：稳定 ID、展示名、类别和可选说明；等级为 `beginner`、
  `intermediate`、`advanced`、`expert`。
- `User`：内部 ID、演示姓名、教育/经验、技能映射和简历文本。简历文本在真实场景
  中属于敏感数据；当前 API 只做确定性词表匹配，不支持文件上传、OCR 或训练过的 NER。
- `JobPosting`：ID、标题、公司、描述、必需/加分技能和可选薪资范围。生成器把结构化
  技能名写入 JD，保证文本与结构化字段不自相矛盾。
- `Interaction`：用户、岗位、类型、时间。类型为 `view`、`click`、`save`、`apply`；
  稀疏矩阵对同一用户/岗位保留最强权重 0.5、1.0、1.5、2.0。这些权重是建模假设，
  不是转化率。
- `SkillRelation`：`source_skill_id -> target_skill_id` 表示
  `PREREQUISITE_OF`，同时保留 `[0,1]` 置信度和来源。

### 2.2 本地产物

| 产物 | 用途 | 生命周期 |
|---|---|---|
| `data/mock_data.pkl` | 可重建 `GraphEntities` 快照 | 可信本地 pickle，不是正式数据集 |
| `data/jobrec_events.sqlite3` | 曝光和反馈运行状态 | 被 Git 忽略，可随本机演示重建 |
| `models/jobrec_bundle.json` | 在线映射、seen set、权重和版本 | 发布契约 |
| `models/lightgcn_model.pt` | bundle 引用的 checkpoint | 与 bundle 成对生成 |
| `results/experiment_summary.json` | 五种子基线/消融 | 当前实验数值来源 |
| `results/coverage.json` | 记录运行的覆盖率明细 | 质量证据快照 |
| `results/load_test_summary.json` | 单机 API 负载冒烟 | 协议限定的性能证据 |

不要反序列化不可信 pickle/checkpoint，不要手工编辑模型权重和结果数字。

## 3. 半合成生成协议

默认生成器创建用户、岗位、技能、类型化关系、申请和交互。交互概率与强度由一个只在
生成阶段使用的 compatibility oracle 调节，信号包括：

- 必需技能的等级覆盖；
- 加分技能重合；
- 有界的经验匹配项。

这样做的目的不是模拟真实劳动力市场，而是让测试数据存在透明的可学习信号，避免用
完全随机交互评价推荐器。oracle 不进入模型特征，也不直接作为测试标签。

构造式测试验证观察交互对的平均 compatibility 明显高于未观察对，并验证生成 JD
包含其结构化必需技能。改变生成概率或分布时必须更新测试与本文。

## 4. 加载、切分与防泄漏协议

```mermaid
flowchart TD
  A["Pydantic 校验后的 GraphEntities"] --> B["按最小交互过滤"]
  B --> C["建立连续用户/岗位映射"]
  C --> D["加权稀疏矩阵 R"]
  D --> E["固定种子逐用户留出"]
  E --> F["train_R"]
  E --> G["test_R"]
  F --> H["训练邻接图"]
  H --> I["LightGCN/BPR"]
  F --> J["服务 seen set"]
  G --> K["屏蔽 seen 后的排序评估"]
  J --> K
```

`src/data/loader.py` 必须满足：

1. `test_ratio` 位于 `[0,1)`；
2. 少于两次交互的用户只进入训练集；
3. 每个评估用户至少保留一个训练交互和一个测试正例；
4. `train_R + test_R` 的非零项与过滤后的原矩阵一致；
5. LightGCN 邻接只由 `train_R` 生成；
6. 评估和服务屏蔽该用户所有训练期岗位；
7. BPR 负候选排除该用户观察过的岗位；无合法负样本的用户从该批次剔除。

测试位于 `tests/test_core_pipeline.py`、`tests/test_data_graph_and_ranking.py` 和
`tests/test_security_persistence_and_statistics.py`。规则变化需要构造式回归测试，不能
只比较一个最终指标是否波动。

## 5. 指标定义

当前多模型实验统一计算：

- `Recall@10`：前 10 中命中的测试正例数 / 该用户测试正例数；
- `NDCG@10`：按排名折损的命中增益 / 理想增益；
- `MRR@10`：首个测试正例排名的倒数，无命中为 0；
- `Catalogue coverage@10`：所有评估用户 Top-10 覆盖的不同岗位数 / 岗位总数。

这些指标只衡量当前半合成 holdout 协议。SQLite 的 `effectiveness` 是已记录满意反馈
占全部反馈的比例；API 只接收有同用户、同岗位、同模型版本曝光的反馈。它仍不是旧
赛题要求的真实用户调查，因为缺少样本人群、曝光 ID、无响应处理和调查流程。

## 6. 多种子基线与消融

**已验证：2026-07-19 重跑，结果与 2026-07-15 保留产物逐字节一致。** 命令：

```bash
uv run python -m scripts.run_experiments
```

协议：每个种子生成 40 用户、100 岗位；种子为 11、19、23、31、42；逐用户留出；
所有模型共享候选池和 seen masking；LightGCN 训练 15 epoch；默认文本模型为离线
feature hashing（产物沿用 `B3_sbert` 历史键名，但本次没有加载预训练 SBERT）。

| 模型 | Recall@10 均值±总体标准差 | NDCG@10 均值±总体标准差 | MRR@10 均值 | Catalogue coverage@10 |
|---|---:|---:|---:|---:|
| B0 Random | 0.1271 ± 0.0182 | 0.0985 ± 0.0160 | 0.1664 | 0.984 |
| B1 Popularity | 0.1048 ± 0.0213 | 0.0707 ± 0.0197 | 0.1070 | 0.210 |
| B2 Skill | **0.2145 ± 0.0335** | **0.1714 ± 0.0254** | **0.2734** | 0.922 |
| B3 Offline text | 0.1805 ± 0.0287 | 0.1329 ± 0.0155 | 0.2061 | 0.828 |
| B4 LightGCN | 0.1215 ± 0.0088 | 0.0866 ± 0.0080 | 0.1395 | 0.972 |
| E1 LightGCN + text | 0.1488 ± 0.0198 | 0.1191 ± 0.0122 | 0.2074 | 0.930 |
| E2 + uniform skill | 0.2018 ± 0.0295 | 0.1620 ± 0.0169 | 0.2700 | 0.936 |
| E3 + GAT skill | 0.2057 ± 0.0293 | 0.1643 ± 0.0161 | 0.2701 | 0.926 |

完整逐种子值在 [`../results/experiment_summary.json`](../results/experiment_summary.json)。
重跑前后 SHA-256 均为
`1c0dd1d2f2aa3e493b51945b37982140d70460bc4ed3091c5746c304c8c8f704`。

### 6.1 可以得出的结论

- 技能基线 B2 的均值最好，说明生成器的主要信号确实是技能兼容度。
- E3 相比 E2 只有很小的均值差异，当前协议不足以证明 GAT 权重有稳定收益。
- LightGCN 单路在该小型稀疏分布上弱于技能和文本基线；保留该负面结果比选择性报告
  复杂模型更可信。
- 融合链路的价值在当前阶段主要是接口、发布和解释能力，不是“效果领先”的证明。

### 6.2 不能得出的结论

- 表中没有真实标签，不能外推到真实投递、面试或用户满意度。
- 五个种子不是独立真实数据集，也没有统计显著性或外部基准复现。
- GAT 使用代理监督，不能把学到的权重解释为真实岗位市场的重要度。
- `B3_sbert` 的产物键名不代表本次运行了预训练 Sentence-BERT。

## 7. 工程验证

### 7.1 本地质量门禁

**已验证日期：2026-07-19；环境：macOS、Python 3.13、锁定 uv 环境。**

```bash
uv run python -m compileall -q src scripts tests main.py
uv run python -m pytest -q --cov=src --cov-report=term-missing --cov-fail-under=50
uv run black --check src scripts tests main.py
uv run isort --check-only src scripts tests main.py
uv run mypy src/api/routes.py src/models/bundle.py src/data/graph_store.py \
  src/generation/adapters.py src/security.py src/data/models.py \
  src/metrics/event_store.py
```

当前复核结果：17 个测试通过，`src` 总行覆盖率 56.43%，compile、Black、Isort 与
上述关键服务边界的 mypy 均通过。精确覆盖率快照见
[`../results/coverage.json`](../results/coverage.json)。测试运行出现依赖侧弃用警告，
不影响本次通过结论，但不应误报为“零警告”。

### 7.2 API 负载冒烟

**已验证产物日期：2026-07-15。** 环境为本地 macOS、单 uvicorn worker、离线
feature-hashing 编码器。协议为 100 个请求、并发 10：

| 指标 | 结果 |
|---|---:|
| 错误率 | 0% |
| 吞吐 | 328.40 req/s |
| 平均延迟 | 26.05 ms |
| P50 | 7.90 ms |
| P95 | 103.91 ms |
| P99 | 223.80 ms |

原始摘要见 [`../results/load_test_summary.json`](../results/load_test_summary.json)。这是
功能/小负载冒烟，不是耐久、多 worker、1000 并发、故障恢复或生产 SLO 证明。

### 7.3 环境受限项

- Dockerfile 与 Compose 清单已静态提供，保留记录没有证明镜像实际构建和运行。
- Neo4j 适配器、Cypher 和导入脚本已实现；没有外部 Neo4j 的集成与故障证据。
- 可选预训练 Sentence-BERT 和外部 LLM 不在默认离线质量门禁中。
- 公平性模块只能展示指标计算，缺少合法真实群体属性与治理流程。

## 8. 若未来获得合法新数据

不得尝试“复原”已丢失数据或从公开个人页面无授权抓取。只有在获得新的合法、授权、
可治理数据后，才应另行定义：

- 数据所有者、使用目的、授权、保留与删除；
- 唯一事件/曝光 ID、事件时间、模型版本、位置和实验分组；
- 岗位时效、去重、技能本体版本和简历解析证据；
- 按时间切分、反馈延迟、冷启动与候选池构造；
- 漂移、缺失、选择偏差、受保护属性使用和独立审计；
- 模型注册、回滚、线上实验和容量/故障测试。

在这些条件满足前，项目的诚实结论是“面试级、协议可复现原型”，不是“真实业务
推荐系统”。
