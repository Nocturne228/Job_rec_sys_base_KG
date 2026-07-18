# 外部资料目录

`ref/` 保存论文、技术文章导出和历史赛题文件，只用于理解问题与解释技术选型。

> 外部资料不是本项目实现、实验、性能或业务效果的证据。引用标题、作者、年份、
> 数据集和数字前必须回到原文件核对；技术博客和二次转载还应核对原始来源。

## 使用规则

1. 本项目行为以代码和测试为准，数值以 `results/` 为准。
2. 可以说“受某论文/实践启发”，不能说“本项目复现了其指标”，除非仓库保留了
   相同数据、协议、代码和结果。
3. 不把外部生产规模、CTR、延迟或模型提升复制到简历和项目文档中。
4. 新增资料时保留可识别标题，记录完整书目信息和获取来源；不要再创建逐篇长篇
   摘要、路线图或“预期收益”表。
5. PDF 可能受版权和许可约束，不对外重新分发前应确认使用权限。

## 核心阅读入口

| 主题 | 本地资料 | 用途 |
|---|---|---|
| 协同过滤 | [`LightGCN.pdf`](papers/LightGCN.pdf) | LightGCN 简化传播与隐式反馈推荐背景 |
| 文本向量 | [`Sentence-BERT`](<papers/Sentence-BERT- Sentence Embeddings using Siamese BERT-Networks.pdf>) | 双塔句向量与文本召回背景 |
| 招聘推荐综述 | [`systematic literature review`](<papers/Job recommender systems- a systematic literature review, applications, open issues, and challenges.pdf>) | 冷启动、双向匹配、解释和评估问题 |
| 可解释招聘推荐 | [`OKRA`](<papers/OKRA An Explainable, Heterogeneous, Multi-stakeholder Job Recommender System.pdf>) | 异构实体、路径解释和多方目标背景 |
| 人岗匹配 | [`Modeling Two-Way Selection Preference`](<papers/Modeling Two-Way Selection Preference for Person-Job Fit.pdf>) | 双向选择问题定义 |
| 行业系统 | [`LinkedIn job recommendation`](<papers/Personalized Job Recommendation System at LinkedIn- Practical Challenges and Lessons Learned.pdf>) | 工业流程与约束的外部案例 |
| 图岗位推荐 | [`graph-based employer recommendation`](<papers/Job Seeker Recommendation for Employers A Graph-Based Recommendation Approach Using Node Embedding.pdf>) | 招聘方反向匹配背景 |
| 技能与岗位 | [`Skills2Job`](<papers/Skills2Job--A-recommender-system-that-encodes-job-offer-_2021_Applied-Soft-C.pdf>) | 技能表达与岗位推荐背景 |
| 历史任务 | [`赛题.pdf`](赛题.pdf) | 项目最初题目和目标；目标不等于当前已验证结果 |

其他材料保存在 [`papers/`](papers/)；文件名即当前的最小本地索引。需要正式发表或
提交材料时，应另行生成经核对的标准参考文献列表，而不是依赖文件名猜测书目信息。

## 与当前项目的边界

- 当前文本实验默认是 feature hashing，并未复现 SBERT 论文结果。
- 当前 GAT 使用半合成图和代理监督，并未复现外部图推荐论文结果。
- 当前 Neo4j、FastAPI、LLM 和向量检索资料只提供工程思路，外部平台指标不能外推。
- 当前公平性材料只能帮助设计未来审计，无法替代合法真实属性与治理流程。

本项目的可引用本地证据见
[`../docs/data-and-evaluation.md`](../docs/data-and-evaluation.md)。
