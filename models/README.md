# 模型产物

`jobrec_bundle.json` 是活动版本指针；`lightgcn-<hash>.pt` 是不可变 checkpoint。

```bash
uv run python -m scripts.build_model_bundle --epochs 20
```

发布流程先训练临时 checkpoint，以 SHA-256 命名不可变文件，再原子替换 bundle。bundle
Bundle v3 记录数据、训练配置、权重、映射、seen set、Pointwise 参数、文本配置和重排
配置，并用 `serving_sha256` 标识完整在线行为。服务加载时逐项校验，不能手工修改权重
或 bundle 数字。
