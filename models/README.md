# 模型产物目录

本目录保存离线训练发布给本地服务的成对产物。

| 产物 | 用途 |
|---|---|
| `jobrec_bundle.json` | `ModelBundle`：schema/model 版本、checkpoint、连续 ID 映射、训练期 seen set、技能与排序权重 |
| `lightgcn_model.pt` | bundle 引用的 LightGCN checkpoint |

统一重建：

```bash
uv run python -m scripts.build_model_bundle --epochs 20
```

bundle 是离线/在线契约；API 不得重建不兼容映射或静默训练。模型文件不能证明效果，
也不得手工修改或加载不可信 checkpoint。设计约束见
[`../docs/architecture.md`](../docs/architecture.md)。
