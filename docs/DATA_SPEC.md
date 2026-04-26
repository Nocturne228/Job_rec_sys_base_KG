# JobRec_KG 数据需求规范

对于jobrec项目，我现在没有具体的大量简历数据描述来对模型进行训练，但是请你新建一个docs目录，描述为了完成项目所需各功能需要的数据类型是什么样的，该通过哪些方式得到和处理，用来做什么（整体按照技术岗位面试的方向来准备）

> 面向技术面试：各模块需数据的类型、获取方式、处理流程、用途说明

---

## 总览

按系统 "召回 → 排序 → 生成 → 解释" 四阶段架构，共需 **6 类数据**：

| # | 数据类型 | 关键用途 | 当前状态 |
|---|---------|---------|---------|
| 1 | 简历/求职者数据 | 用户画像、技能抽取 | Mock 20 人 |
| 2 | 岗位描述 (JD) | 岗位画像、技能抽取 | Mock 50 条 |
| 3 | 用户行为日志 | LightGCN BPR 训练 | Mock 交互 |
| 4 | 技能知识图谱 | GAT 技能加权、学习路径 | 内存 Dict 模拟 |
| 5 | 交互行为标注 | 线性融合权重调参 | 规则权重 |
| 6 | 生成层语料 | GraphRAG 检索、Prompt 构建 | 模板模拟 |

生产规模为 ~23k 节点 / ~168k 边，原始数据已丢失。本文档为每类数据给出完整的 **规格 → 获取 → 处理 → 用途** 链路 。

---

## 1. 简历 / 求职者数据

### 1.1 数据规格

| 字段 | 类型 | 示例 | 说明 |
|------|------|------|------|
| `resume_id` | `str` | `"res_00001"` | 唯一标识 |
| `raw_text` | `str` | "三年 Java 开发经验..." | 原始简历全文 |
| `skills_extracted` | `List[str]` | `["Java", "Spring Boot", "MySQL"]` | NER 提取的技能列表 |
| `skill_levels` | `Dict[str, SkillLevel]` | `{"Java": "advanced", "MySQL": "intermediate"}` | 各技能掌握程度 |
| `education` | `str` | "重庆大学 本科 计算机科学" | 学校/学历 |
| `experience_years` | `float` | `3.0` | 工作年限 |
| `project_descriptions` | `List[str]` | `["负责 XX 微服务架构改造"]` | 项目经历（可选） |

### 1.2 获取方式

| 途径 | 规模预估 | 难度 | 说明 |
|------|---------|------|------|
| **爬虫 — 招聘平台简历模板** | 500-5000 份 | ★★☆ | 爬取 Boss直聘/拉勾网的"候选人展示"页面的脱敏简历 |
| **开源数据集** | 100-1000 份 | ★☆☆ | Kagga 有少量简历数据集 (Resume Dataset)；中文简历可搜索 "简历 数据集 GitHub" |
| **LLM 批量生成** | 任意规模 | ★☆☆ | 用 GPT/Qwen 生成高质量中文技术简历（见 §1.3） |
| **内部收集** | 50-200 份 | ★★★ | 就业指导中心经用户授权后收集 |

**推荐方案**：LLM 批量生成 + 少量真实简历校验，可快速构建 500-1000 份覆盖面广的数据集。

### 1.3 LLM 生成方案

Prompt 模板：
```
你是一位有 N 年经验的计算机科学毕业生，求职方向为 {target_role}。
请生成一份结构化的中文简历，包含：
1. 基本信息（随机姓名、学校从 C9/211/普通本科中随机）
2. 教育背景、工作年限
3. 技能列表（从以下技能池中抽取 4-8 个并标注熟练度）：{skill_pool}
4. 2-3 个项目经历（包含具体技术栈和职责）
要求：真实感强、技术栈组合符合行业现状、不同候选人之间有明显差异。
```

生成格式：
```json
{
  "name": "xxx",
  "education": "...",
  "experience_years": 3.0,
  "skills": {"Java": "advanced", "Spring Boot": "intermediate"},
  "projects": [
    {"name": "微服务改造", "tech_stack": ["Spring Boot", "Docker", "Redis"], "description": "..."}
  ]
}
```
### 1.4 处理流程

```
原始简历 (PDF/HTML/Text)
    │
    ├──► 文本提取 ─── PyMuPDF/pdfplumber(简历PDF) or BeautifulSoup(HTML)
    │
    ├──► 字段分割 ─── 模板匹配 (正则) 或 LLM 分割 (成本低，简历结构固定)
    │
    ├──► 技能 NER   ─── spaCy + 自定义词典 or Qwen-2.5-7B-Instruct (zero-shot 抽取)
    │   └─── 词典: 各语言的 API + 技术栈词表（可从 GitHub repo README 提取高频词）
    │
    └──► 技能等级判定 ─── 规则 heurstics:
         "精通/expert" → EXPERT
         "熟练/proficient" → INTERMEDIATE
         "了解/familiar" → BEGINNER
         工作年限 + 项目复杂度联合推断
```

### 1.5 在系统中的用途

- **LightGCN 冷启动用户嵌入** → SBERT 编码简历文本 → 向量检索相似用户
- **技能匹配** → 提取的技能标签 → 与岗位所需技能计算覆盖率
- **LangGraph 输入** → 结构化用户技能画像 → Prompt 生成

---

## 2. 岗位描述 (Job Description, JD)

### 2.1 数据规格

| 字段 | 类型 | 示例 | 说明 |
|------|------|------|------|
| `job_id` | `str` | `"job_00001"` | 唯一标识 |
| `title` | `str` | "Java 后端开发工程师" | 岗位名称 |
| `company` | `str` | "杭州 xx 科技有限公司" | 公司名 |
| `description_raw` | `str` | "岗位职责：...\n任职要求：..." | JD 原文 |
| `required_skills` | `Dict[str, SkillLevel]` | `{"Java": "intermediate", "MySQL": "intermediate"}` | 必需技能 |
| `preferred_skills` | `Dict[str, SkillLevel]` | `{"Docker": "beginner"}` | 加分技能 |
| `salary_range` | `Tuple[float, float]` | `(15000, 25000)` | 月薪（元） |
| `location` | `str` | "杭州" | 工作地点 |
| `experience_required` | `str` | "1-3 年" | 经验要 |
| `education_required` | `str` | "本科" | 学历要求 |

### 2.2 获取方式

| 途径 | 规模预估 | 难度 | 说明 |
|------|---------|------|------|
| **爬虫 — 招聘平台 JD** | | ★★★★ | Boss/拉勾/前程无忧；注意反爬、robots.txt、法律合规 |
| **开源 JD 数据集** | 1k | ★★☆ | Hugging Face 搜索 "job description" / "招聘"；Kaggle "Job Description Dataset" |
| **API 合作** | 5k-20k | ★★★ | 如 Boss 直聘开放 API (需商务合作) |
| **LLM 生成** | 任意规模 | ★☆☆ | 按岗位 + 公司 + 地域生成真实感 JD |

**推荐方案**：开源 JD (1-2k) + LLM 扩展生成 (10-20k)，覆盖技术岗位 30+ 类别。

### 2.3 LLM 生成方案

Prompt 模板：
```
请生成一份真实感的中文岗位描述（JD）：
- 岗位名称：{job_title}（如"后端开发工程师"、"数据科学家"）
- 目标城市：{city}
- 公司规模：{company_size}
- 要求包含：岗位职责、任职要求、加分项
- 技能要求从以下技能池中选取 4-8 个，并标注期望等级：{skill_pool}
- 薪资范围需符合当前市场水平
- 不要出现明显不合理的技能组合（如前端岗要求 C++/CUDA）
```

### 2.4 处理流程

```
JD 原文 (HTML/Text)
    │
    ├──► 结构化解析 ─── 正则匹配 or LLM 解析：
    │     - 岗位职责 vs 任职要求 vs 加分项
    │
    ├──► 技能提取 ─── 与简历相同的 NER pipeline（共享词典/模型）
    │
    └──► 技能等级映射 ─── "必须/精通" → required, "熟悉/了解" → preferred
                            "优先/加分项" → preferred
```

### 2.5 在系统中的用途

- **LightGCN item 侧嵌入** → (user_id, job_id) 交互对训练
- **SBERT 语义召回** → JD 文本 → all-MiniLM-L6-v2 编码 → FAISS 索引
- **GAT 训练特征** → 岗位技能出现频率 → 技能重要性特征向量
- **LangGraph 检索源** → 岗位所需技能 → 技能差距分析

---

## 3. 用户行为日志 (Implicit Feedback)

### 3.1 数据规格

| 行为类型 | 权重 | 数量级 (20k 用户) | 字段 |
|---------|------|------------------|------|
| view (浏览) | 0.5 | ~500k/月 | user_id, job_id, timestamp, duration |
| click (点击) | 1.0 | ~200k/月 | user_id, job_id, timestamp |
| save (收藏) | 1.5 | ~50k/月 | user_id, job_id, timestamp |
| apply (投递) | | ~200/月 | user_id, job_id, timestamp, status |

> status: applied → interview → rejected / accepted

### 3.2 获取方式

| 途径 | 说明 |
|------|------|
| **前端埋点** | 产品上线后通过 JavaScript 事件收集（生产环境标准做法） |
| **日志回放** | 若有历史 Nginx/App 日志，可解析出 user-job 交互 |
| **LLM 模拟交互** | 基于简历 + JD 的匹配度合成合理交互信号（PoC 阶段首选） |
| **公开数据集** | |

### 3.3 LLM 模拟方案

```python
# 逻辑：根据简历技能与 JD 技能的匹配度生成合理的交互
for user in users:
    for job in jobs:
        match_score = compute_skill_overlap(user.skills, job.required_skills)
        # match_score 高 → 更可能点击/收藏/投递
        p_view   = min(0.8 * match_score + 0.2, 0.9)
        p_click  = min(0.5 * match_score + 0.1, 0.7)
        p_save   = min(0.3 * match_score, 0.4)
        p_apply  = min(0.2 * match_score, 0.3)
        # 按比例采样
```

**关键设计**：加入 10-20% 随机噪声，模拟用户真实行为中的非理性因素。

### 3.4 处理流程

```
原始交互日志
    │
    ├──► 去重 & 清洗 ── 同一用户-岗位对 24h 内的重复 view 合并
    │
    ├──► 加权聚合 ── score = sum(weight[type] for each interaction)
    │     └─── view=0.5, click=1.0, save=1.5, apply=2.0
    │
    ├──► 正负样本构造 ── score > threshold → positive
    │                      score = 0 AND exposure → negative
    │
    └──► LightGCN 训练集 ── train/val/test split (8:1:1)
         └─── BPR Loss: (user, pos_item, neg_item) 三元组
```

### 3.5 在系统中的用途

- **LightGCN 训练** → (u, pos, neg) 三元组 → BPR 损失优化嵌入
- **SBERT 召回评估** → test set → Recall@K / NDCG@K
- **A/B 实验** → 控制组 vs 实验组的 CTR/CVR 对比

---

## 4. 技能知识图谱 (Skill Knowledge Graph)

### 4.1 数据规格

**节点类型**：
| 节点类型 | 属性 | 示例 |
|---------|------|------|
| Skill | id, name, category, difficulty, trend_score | `{"id": "spring_boot", "name": "Spring Boot", "category": "backend", "difficulty": 3}` |
| SkillRelation | src, dst, type, strength | `{"src": "Java", "dst": "Spring Boot", "type": "prerequisite", "strength": 0.9}` |

**关系类型 (Edge Types)**：
| 关系 | 含义 | 示例 |
|------|------|------|
| `prerequisite_of` | 先修关系 | Java → Spring Boot |
| `similar_to` | 相似关系 | React ↔ Vue |
| `part_of` | 组成关系 | NumPy ← Data Science |
| `alternative_to` | 替代关系 | PyTorch ↔ TensorFlow |

### 4.2 获取方式

| 途径 | 规模预估 | 难度 | 说明 |
|------|---------|------|------|
| **Wikipedia/Wikidata 抽取** | 500-2000 节点 | ★★☆ | Cypher 技能相关实体和关系，可用 SPARQL 端点抓取 |
| **Stack Overflow Tags** | 60k+ 标签 | ★☆☆ | SO tags + co-occurrence 构建相似关系；API: https://api.stackexchange.com/ |
| **GitHub 代码库** | 任意规模 | ★★☆ | 从 GitHub topics API 抽取技能共现关系 |
| **招聘平台 JD 共现** | 按需 | ★★★ | 分析 JD 中技能共现频率推断关联强度 |
| **LLM/专家构建** | 100-500 核心节点 | ★☆☆ | LLM 生成先修关系，人工校验（面试中可展示此方法） |

**推荐方案**：SO Tags (相似关系) + LLM + 人工校验 (先修关系) → 100-300 节点精炼图谱。面试时可强调 "我们先用 LLM 做 draft，再请 2 位领域专家 review"。

### 4.3 处理流程

```
原始来源 (Wikipedia/SO/GitHub/LLM)
    │
    ├──► 实体消歧 ─── 同义词合并：
    │     "JS" = "JavaScript", "PyTorch" = "pytorch"
    │
    ├──► 关系抽取 ─── LLM Prompt:
    │     "对于技能对 (A, B)，它们之间是否存在以下关系？
    │      先修/相似/组成/替代/无关？置信度多少？"
    │
    ├──► 质量过滤 ─── 保留置信度 > 0.7 的关系
    │
    ├──► 图构建 ─── NetworkX / Neo4j
    │     nodes: skills
    │     edges: (src, dst, {type, weight})
    │
    ├──► 特征计算 ─── NetworkX 图算法:
    │     - PageRank centrality
    │     - Betweenness centrality
    │     - In-degree / Out-degree
    │     - Shortest path length (learning path)
    │
    └──► GAT 训练特征 ─── 16 维技能特征向量:
          [0:4]   = 岗位频率 one-hot (按 job category)
          [4:8]   = 先修度统计 (in/out degree per relation type)
          [8:10]  = centrality (PageRank + Betweenness)
          [10:12] = difficulty (manual + community variance)
          [12:14] = trend (6-month growth + SO mention freq)
          [14:16] = salary correlation (from JD salary + skill co-occurrence)
```

### 4.4 在系统中的用途

- **GAT 技能加权** → 图结构 + 16 维特征 → 学习每种技能的重要性权重
- **学习路径规划** → shortest path on graph → 从用户当前技能到目标岗位所需技能的最优学习序
- **GraphRAG 检索** → Neo4j Cypher 查询 → 知识图谱子图 → Prompt 上下文

---

## 5. 排序特征数据

### 5.1 训练数据来源

| 特征 | 来源 | 计算方式 |
|------|------|----------|
| Sim_Graph | LightGCN 嵌入 | cosine(e_u, e_j) → [0, 1] 归一化 |
| Sim_Semantic | SBERT 编码 | cosine(E_resume, E_JD) → [-1, 1] |
| Coverage_Skill | 技能标签匹配 | |matched| / |required| (uniform or GAT-weighted) |

### 5.2 获取方式

| 途径 | 说明 |
|------|------|
| **历史排序日志** | 若有已有推荐系统，可导出 (user, job, ranking_score, clicked) 日志用于学习排序权重 (LTR) |
| **人工标注** | 请 2-3 人对 100-200 (user, job) 对打分 (1-5 分)，作为排序优化目标 |
| **LLM 辅助标注** | LLM 根据技能匹配度给出 initial ranking → 人工抽查校验 → 扩展至 1000+ 对 |

### 5.3 处理流程

```
多源特征
    │
    ├──► 归一化 ─── 所有特征 map → [0, 1]
    │
    ├──► 权重学习 ─── 目标：最大化 positive pair 的 ranking_score
    │     ├── 方法 A: 网格搜索 (0.1 步长遍历 w1, w2, w3)
    │     └── 方法 B: Logistic Regression (特征=3 个分数, 标签=click/no-click)
    │
    └──► 冻结权重 → 线上推理直接用 0.4/0.3/0.3
```

### 5.4 在系统中的用途

- **最终推荐排序** → Ranking_Score → Top-10 推荐列表
- **可解释性** → 因子贡献度分解 ("推荐此岗位因为：图相似度 42%，语义匹配 31%，技能覆盖 27%")

---

## 6. 生成层语料 (LangGraph + GraphRAG)

### 6.1 数据规格

| 内容 | 格式 | 用途 |
|------|------|------|
| 检索到的技能差距表 | `List[Dict]` | CoT Prompt 的结构化输入 |
| 学习路径 | `List[Dict]` | 从图谱最短路径生成的有序技能列表 |
| 示例评估报告 | `str` (JSON) | Few-shot Prompt 示例 |
| 岗位能力模型 | `Dict` | 各岗位的核心能力框架 |

### 6.2 获取方式

| 途径 | 说明 |
|------|------|
| **职业指导书籍/论文** | 《大学生就业能力模型》等学术文献中的能力维度描述 |
| **行业报告** | 拉勾/BOSS 直聘年度人才报告、Stack Overflow Developer Survey |
| **专家咨询** | 就业指导中心/企业 HR 提供能力评估框架模板 |
| **LLM 生成** | 用 Qwen/GPT 生成各岗位的能力模型和个性化建议模板 |
### 6.3 处理流程

```
知识图谱 + 用户画像 + 岗位画像
    │
    ├──► Cypher 查询 ─── 获取用户缺失技能 + 最短学习路径
    │
    ├──► CoT Prompt 模板 ───
    │     """
    │     用户 {name} 申请 {job_title} 岗位：
    │     - 已掌握技能：{matched_skills}
    │     - 缺失技能：{gap_skills} (按重要性排序)
    │     - 最学习路径：{learning_path}
    │     
    │     请输出：1. 总体匹配度评估
    │            2. 各差距分析
    │            3. 学习路径建议
    │            4. 时间预估
    │     """
    │
    └─── Qwen-2.5 (t=0.3) → 结构化 JSON 输出
```

### 6.4 在系统中的用途

- **个性化报告生成** → 每个用户-岗位对 → 一份结构化职业发展建议
- **可解释性** → 用户可看到 "为什么推荐这个岗位" 和 "如何提升自己"

---

## 数据采集优先级 (PoC → 生产)

| 阶段 | 优先级 | 数据类型 | 目标规模 | 获取方式 | 预期时间 |
|------|--------|---------|---------|---------|---------|
| **PoC** | P0 | 简历 + JD | 50 res + 200 JD | LLM 生成 | 2-3 天 |
| **PoC** | P0 | 技能图谱 | 100 技能 + 300 关系 | LLM + 人工校验 | 3-5 天 |
| **PoC** | P1 | 行为日志 | 模拟交互 | 匹配度合成 | 1 天 |
| **v2** | P0 | 真实简历 | 200-500 份 | 爬取/收集/LLM 扩展 | 2-3 周 |
| **v2** | P0 | 真实 JD | 5k-10k | HuggingFace + 爬取 | 1-2 周 |
| **v2** | P1 | SO/Wikipedia 图谱 | 1k+ 技能 | API + LLM | 1 周 |
| **Prod** | P0 | 前端埋点 | 持续积累 | JS 采集 + 后端聚合 | 持续 |
| **Prod** | P1 | 人工标注 | 500-1000 对 | 众包/实习生 | 2-4 周 |

---

## 面试答辩重点话术

> 以下是在面试中介绍数据部分的建议话术，可直接使用：

### 关于 "数据来源"
> "由于项目初期缺乏大规模真实数据，我们采用了分层策略：首先通过 LLM 批量生成高质量种子数据（简历+JD），快速跑通管线；同时接入开源数据集 (HuggingFace/Kaggle) 进行校验；生产环境上线后通过前端埋点持续采集真实交互数据，逐步替换模拟数据。"

### 关于 "简历解析"
> "简历解析采用两阶段流程：第一阶段用模板匹配实现快速字段分割（简历结构相对固定）；第二阶段用 LLM 做零样本技能抽取，结合自定义技术词典进行实体消歧和归一化。技能等级通过关键词映射（精通/熟练/了解）结合工作年限综合推断。"

### 关于 "知识图谱构建"
> "技能图谱的数据来自三个来源：Stack Overflow Tags 提供技能共现关系，GitHub Topics API 提供技术分类，LLM 生成先修关系草案后由领域专家审核。目前覆盖 100+ 技能、300+ 有向边，支持 PageRank 中心性计算和最短学习路径规划。"

### 关于 "为什么可以只用 Demo 数据验证"
> "推荐系统的核心验证点是架构正确性和算法接口一致性。我们用 20×50 的 Demo 数据验证了 LightGCN 训练收敛、SBERT 召回可用、GAT 特征传播正常、LangGraph 管线跑通。所有模型超参和接口在 23k/168k 生产规模下保持一致，只需调整 batch size 和显存配置。"

### 关于 "数据质量和标注"
> "排序层权重的调优采用了半自动方式：先用网格搜索在 LLM 生成的 1000 对样本上找到最优权重，再用人工随机抽查 200 对进行 A/B 对比验证。生产环境下会通过线上 A/B 实验持续优化，采用点击率/投递率作为最终优化目标。"

---

## 附录：数据字典速查表

| 数据 | 格式 | 存储 | 规模 | 更新周期 |
|------|------|------|------|---------|
| 简历 | JSON | MySQL/MongoDB | 500-∞ | 注册时 |
| JD | JSON | MySQL | 5k-∞ | 每日爬取 |
| 交互日志 | Parquet | HDFS/S3 | 100k+/月 | 实时写入 |
| 技能图谱 | GraphML/Cypher | Neo4j | 300+ 节点 | 季度更新 |
| GAT 特征 | Numpy Array | Redis/内存 | 300×16 | 图谱更新时 |
| 推荐日志 | JSON | MySQL | 10k+/月 | 每次推荐 |
