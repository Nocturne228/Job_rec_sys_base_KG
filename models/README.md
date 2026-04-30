# models/ — 模型权重存储目录

## 用途

存储训练好的模型权重文件，供推理和评估使用。

## 当前内容

| 文件 | 大小 | 说明 |
|------|------|------|
| `lightgcn_model.pt` | ~1 MB | LightGCN 训练后的权重（Demo, 50 epochs, BPR Loss） |

## 期望文件格式

### LightGCN 模型权重 (`.pt`)

`src/recall/lightgcn.py:LightGCN.save()` 保存的 PyTorch state_dict:

```python
{
    "user_embedding.weight": Tensor[num_users, 64],
    "item_embedding.weight": Tensor[num_jobs,  64],
    "model_config": {"n_users": N, "n_items": M, "embedding_dim": 64, "n_layers": 3}
}
```

加载方式: `lightgcn_model.save("models/lightgcn_model.pt")` / `lightgcn_model.load("models/lightgcn_model.pt")`

### 生产环境扩展

```
models/
├── lightgcn_prod.pt        # LightGCN 生产模型（23k 节点规模）
├── gat_skill_weights.pt     # GAT 训练后的技能重要性权重
└── convnext_tiny_ema.pth    # 如需多模态特征提取（参考 med/ 项目）
```

## 当前状态

- Demo: `lightgcn_model.pt` 由 `main.py:train_lightgcn_model()` 自动生成
- 文件已被 `.gitignore` 排除（不提交到版本控制）
