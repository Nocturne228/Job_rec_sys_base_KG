# logs/ — 日志文件存储目录

## 用途

存储训练日志和系统运行日志。由 `src/config/settings.py:SystemConfig.logs_dir` 配置（默认值 `"logs"`）。

## 当前状态

目录为空。当前 Demo 阶段的训练日志输出到 stdout，未持久化到文件。

## 期望内容（生产环境）

```
logs/
├── training_YYYYMMDD.log       # 训练日志（epoch, loss, metrics）
├── recall_YYYYMMDD.log         # 召回延迟 + 命中率日志
├── ranking_YYYYMMDD.log        # 排序请求/响应日志
├── generation_YYYYMMDD.log     # LLM 调用日志（success/fail/latency）
├── ab_test_YYYYMMDD.log        # A/B 实验事件日志
└── errors_YYYYMMDD.log         # 错误和异常日志
```

### 训练日志格式示例

```
2024-01-15 10:30:01 [INFO] Epoch 1/50 - Loss: 0.6932
2024-01-15 10:30:05 [INFO] Epoch 10/50 - Loss: 0.4231, Recall@20: 0.183
2024-01-15 10:30:10 [INFO] Epoch 20/50 - Loss: 0.3124, Recall@20: 0.241
2024-01-15 10:30:15 [INFO] Early stopping at epoch 35, best loss: 0.2891
```

## 集成方式

在 `main.py` 或训练脚本中添加:

```python
import logging
logging.basicConfig(
    filename=f'logs/training_{datetime.now():%Y%m%d}.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
```

## 当前状态

- 目录由 `main.py:setup_directories()` 自动创建（exist_ok=True）
- 日志写入功能未实现，属待建设项
- 文件已被 `.gitignore` 排除
