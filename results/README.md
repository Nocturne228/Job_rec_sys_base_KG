# results/ — 实验结果存储目录

## 用途

存储 Demo 运行和离线评估的实验结果。

## 当前内容

| 文件 | 说明 |
|------|------|
| `demo_results.pkl` | `main.py:run_complete_demo()` 运行输出的聚合结果 |

## 期望输出格式

### 完整实验输出结构（生产环境）

```
results/
├── demo_results.pkl              # Demo 一键运行结果
├── experiment_YYYYMMDD_HHMMSS/   # 每次实验独立子目录
│   ├── config.json                # 超参数快照
│   ├── metrics.json               # {"recall@10": 0.29, "ndcg@10": 0.33, ...}
│   ├── training_curves.png        # Loss + 指标曲线
│   ├── confusion_matrix.png       # 技能覆盖率混淆矩阵（可选）
│   └── model_weights/             # 该次实验的模型权重
├── ab_test_results/               # A/B 实验报告
│   └── experiment_*.json          # {"cvr": {"p_value": 0.003, "significant": true}, ...}
└── llm_eval_results/              # LLM 生成质量评估
    └── eval_*.json                # {"relevance": 0.8, "feasibility": 0.7, ...}
```

### demo_results.pkl 格式

```python
{
    "data_stats": {
        "n_users": 20, "n_jobs": 50, "n_skills": 22
    },
    "model_info": {
        "lightgcn_params": 123456,
        "sbert_stats": {"n_users": 20, "n_jobs": 50, "using_faiss": true}
    }
}
```

## 当前状态

- Demo: `demo_results.pkl` 由 `main.py` 最后一步自动生成
- 生产: 实验追踪需从当前 pickle 升级为结构化 JSON + 图表
- 文件已被 `.gitignore` 排除
