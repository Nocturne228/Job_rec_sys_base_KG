# data/ — 数据存储目录

## 用途

存储项目运行所需的输入数据文件，供 `src/data/loader.py` 加载使用。

## 当前内容

| 文件 | 说明 |
|------|------|
| `mock_data.pkl` | Demo 运行生成的模拟数据快照（20 用户 × 50 岗位 × 22 技能） |

## 期望数据格式

### 生产数据: GraphEntities (Pickle 序列化)

```python
from src.data.models import GraphEntities
# 结构: users[User], jobs[JobPosting], skills[Skill], 
#       applications[Application], interactions[Interaction]
```

### 替代方案: 各实体独立存储

```
data/
├── users.json         # [{"id": "user_001", "skills": {"python": "advanced"}, ...}, ...]
├── jobs.json          # [{"id": "job_001", "required_skills": {...}, ...}, ...]
├── skills.json        # [{"id": "python", "name": "Python", "category": "programming"}, ...]
├── applications.json  # [{"user_id": "user_001", "job_id": "job_001", "status": "applied"}, ...]
└── interactions.jsonl # {"user_id": "user_001", "job_id": "job_001", "type": "click", "ts": "..."}
```

## 当前状态

- Demo: `mock_data.pkl` 由 `main.py:generate_and_save_data()` 自动生成
- 生产: 原始 23k 节点/168k 边数据已丢失，需重建
- 重建方案见 `docs/02-数据需求规格.md`
