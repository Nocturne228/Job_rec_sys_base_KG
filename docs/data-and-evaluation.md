# 数据、指标与证据

本文是数据语义和数值结论的唯一详细来源。实测位于
[`../results/experiment_summary.json`](../results/experiment_summary.json)；推导/预期不写入
结果文件。

## 1. 半合成曝光数据

原始真实数据已经丢失。固定种子生成用户、岗位、技能、发布时间、先修关系和 Feed
曝光。每次曝光包含 `timestamp`、`position`、`clicked`、`dwell_seconds`、`saved` 和
`applied`。点击概率由透明技能兼容度、噪声和位置衰减共同影响；推荐模型看不到生成器
内部的 oracle。

排序正例定义为：

```text
clicked OR saved OR applied OR dwell_seconds >= 20
```

未满足条件的曝光是排序负例；只有正向动作生成 LightGCN 协同边。服务 Bundle 使用
20 用户 × 50 岗位；实验每个种子使用 40 用户 × 100 岗位。五个种子的排序训练曝光约
1551–1588 条，正例率约 10.3%–12.4%。

该分布适合验证曝光样本、时间切分、多阶段接口和负面消融，不代表抖音或真实招聘流量。

## 2. 时间切分和防泄漏

每个用户的正向交互按 ISO 时间排序，最后约 20% 作为 `test_R`，其余进入 `train_R`；
至少保留一条训练边。随后：

1. LightGCN 二值邻接图和 BPR 正例只来自 `train_R`；
2. BPR 负样本只能来自该用户训练期未见岗位；
3. 热度和近期五条行为兴趣只使用训练交互；
4. Pointwise 只使用训练期曝光，并排除协同测试正例；
5. 离线评估和在线已知用户都先屏蔽 seen，再构造候选与排序特征；
6. 无协同交互岗位仍保留在完整目录中。

曝光和正向交互并不是完整真实日志，排序训练特征仍是训练窗口末端快照，而不是为每条
历史曝光重放当时状态；因此当前只能称时间留出原型，不能声称完成严格 point-in-time
特征回放。

## 3. 模型和阶段

| 名称 | 定义 |
|---|---|
| Random | 用户级固定种子随机分数 |
| Popularity | 训练正向交互列和 |
| Skill | 等级感知必需/加分技能覆盖 |
| Text | 512 维 unigram/bigram feature hashing |
| LightGCN | 32 维、2 层、二值二部图、BPR |
| Fusion | 候选内归一化的 0.4/0.3/0.3 协同/文本/技能基线 |
| Pointwise | 六特征标准化 Logistic 排序 |
| Multistage Feed | 四路各 Top-20 合并、Pointwise、确定性重排 |

Pointwise 的六项特征为协同、文本、技能、热度、新鲜度和近期兴趣。排序权重从训练曝光
学习；当前不是 DeepFM、DIN、Transformer 或多任务模型。

## 4. 指标定义

- `Recall@10`：用户测试相关岗位出现在 Top-10 的比例，再对测试用户平均；
- `NDCG@10`：二值相关性的折损累计增益；
- `MRR@10`：第一个相关岗位排名的倒数；
- `Candidate Recall@80`：四路各 Top-20 合并候选对测试相关岗位的覆盖；80 是各路上限
  之和，去重后实际候选通常更少；
- `Catalogue Coverage@10`：所有用户 Top-10 中不同岗位数除以目录岗位数；
- `Intra-list Diversity@10`：Top-10 岗位两两技能 Jaccard 距离均值；
- `Fresh Job Share@10`：Top-10 中距离目录最新发布时间不超过 14 天的比例。

反馈 API 的 `satisfaction_rate` 只统计显式提供满意度的已记录反馈，不是 CTR。点击、停留、
收藏和投递可以独立记录，但当前没有线上聚合结论。

## 5. 五种子实测

**已验证日期：2026-08-02。** 命令：

```bash
uv run python main.py experiments
```

协议：种子 `11,19,23,31,42`；40 用户、100 岗位；LightGCN 15 epoch；逐用户时间
留出；seen masking；四路召回各 Top-20。

| 模型 | Recall@10 | NDCG@10 | MRR@10 | Diversity@10 | Fresh Share@10 | Coverage@10 |
|---|---:|---:|---:|---:|---:|---:|
| Random | 0.1011 | 0.0478 | 0.0392 | 0.8656 | 0.3229 | 0.990 |
| Popularity | 0.0958 | 0.0469 | 0.0369 | 0.8591 | 0.4182 | 0.140 |
| Skill | 0.2509 | 0.1358 | 0.1137 | 0.8399 | 0.3107 | 0.888 |
| Text | 0.2202 | 0.1043 | 0.0749 | 0.7950 | 0.3457 | 0.846 |
| LightGCN | 0.1144 | 0.0519 | 0.0396 | 0.8649 | 0.3456 | 0.936 |
| Fusion | **0.2551** | **0.1376** | **0.1159** | 0.8225 | 0.3345 | 0.946 |
| Pointwise | 0.1410 | 0.0764 | 0.0652 | 0.8433 | 0.3745 | 0.826 |
| Multistage Feed | 0.1436 | 0.0773 | 0.0652 | 0.8445 | 0.3766 | 0.826 |

Multistage Feed 的 `Candidate Recall@80 = 0.7109`。表中是五种子均值；标准差和逐种子
值见机器可读结果。

### 可以得出的结论

- 固定权重融合仍最好，说明透明的领域先验在小型半合成数据上优于学得模型；
- Pointwise 明显低于 Fusion，约 10% 正例率、静态训练快照和有限曝光不足以证明学习
  排序优势；
- 多阶段候选损失了约 29% 测试相关项，显示每路 Top-20 是明确的效果/计算取舍；
- 重排相对 Pointwise 的多样性和新岗位占比只小幅增加，链路有效但效果很弱；
- LightGCN 单路在当前技能驱动且稀疏的分布上没有优势。

### 不能得出的结论

- 不能外推真实 CTR、停留时长、投递或面试效果；
- 不能证明 Logistic 排序在真实数据中弱于固定融合；
- 五个生成种子不是五份独立真实数据，没有统计显著性结论；
- 当前没有外部 LLM 运行产物，不能声称兴趣扩展提高召回；
- 多样性和新岗位比例不是公平性、生态健康或业务价值认证。

## 6. 模型产物

Bundle v3 的 `serving_sha256` 覆盖数据、checkpoint、Pointwise 参数、文本配置和重排配置。
当前发布版本为 `jobrec-feed-07a2a9ac74aae85e`，排序训练使用 372 条曝光、39 条正例。
服务还校验完整数据哈希、checkpoint 哈希、ID 映射和张量维度。

服务仍通过固定 seed 重新生成演示目录并校验哈希，不是可移植真实数据快照；接入真实
数据后应发布不可变 catalog/feature snapshot。

## 7. 工程验证

默认门禁：

```bash
uv run python -m compileall -q src scripts tests main.py
uv run python -m pytest -q
uv run black --check src scripts tests main.py
uv run isort --check-only src scripts tests main.py
uv run mypy src/api/routes.py src/models/bundle.py src/metrics/event_store.py \
  src/ranking src/recall src/generation/profile_expansion.py src/data/loader.py
```

测试保护时间切分、负采样、seen masking、候选来源、排序可加解释、生成证据约束、技能
路径、Bundle 身份、已知/冷启动 API 和唯一曝光反馈。

## 8. 推导/预期

当前精确召回对目录规模 `N` 的协同点积约 `O(Nd)`，文本稀疏相似度取决于非零项，路内
Top-K 可用 `O(N log K)`，候选排序约 `O(CF)`，重排最坏约 `O(KC)`。在 50 个岗位上
预期为毫秒到几十毫秒量级，但尚未在固定硬件采集分位数，不能写成 P95。

当目录达到十万级且精确文本/向量计算成为瓶颈时，再比较 ANN 相对精确 Top-K 的 Recall、
P50/P95/P99、索引构建时长和内存。
