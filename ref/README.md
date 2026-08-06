# 外部资料边界

`ref/` 保存论文、技术文章导出和历史赛题文件，只用于理解问题与解释技术选择，不是本
项目实现、实验、性能或业务效果的证据。PDF 可能受版权和许可约束，对外分发前需核对。

## 核心阅读

| 主题 | 本地资料 | 当前用途 |
|---|---|---|
| 协同过滤 | [`LightGCN.pdf`](papers/LightGCN.pdf) | 解释简化图传播与隐式反馈 |
| 文本向量 | [`Sentence-BERT`](<papers/Sentence-BERT- Sentence Embeddings using Siamese BERT-Networks.pdf>) | 未来预训练文本方案背景；当前实现只用 hashing |
| 招聘推荐综述 | [`systematic review`](<papers/Job recommender systems- a systematic literature review, applications, open issues, and challenges.pdf>) | 冷启动、双向匹配和评估问题背景 |
| 可解释推荐 | [`OKRA`](<papers/OKRA An Explainable, Heterogeneous, Multi-stakeholder Job Recommender System.pdf>) | 图路径解释的外部思路 |
| 历史任务 | [`赛题.pdf`](赛题.pdf) | 原始题目和目标；目标不等于当前结果 |

## 使用规则

1. 外部论文可以支持技术选择和推导依据，不能把论文指标写成本项目实测；
2. 引用数字时回到原文件核对数据、协议、硬件和年份；
3. 生产规模、CTR、延迟和提升不得移植到简历；
4. 若用外部基准推导预期，必须说明差异、假设、范围和本地验证计划；
5. `papers/` 中其余文件是候选阅读材料，不意味着对应技术已经实现。
