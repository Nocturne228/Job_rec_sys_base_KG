"""
工作推荐系统的配置管理模块。
使用 Pydantic BaseModel 进行类型安全的配置定义，支持嵌套配置组和动态更新。
整体设计遵循 "配置与代码分离" 的原则，所有超参默认通过配置文件管理。
"""
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class DeviceType(str, Enum):
    """模型训练设备类型枚举。"""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Metal（Mac GPU 加速）


class ModelConfig(BaseModel):
    """模型超参数配置组。

    涵盖 LightGCN、SBERT、排序融合和 LLM 四个子模块的参数设置。
    """

    # LightGCN 参数
    lightgcn_embedding_dim: int = Field(default=64, description="用户/岗位嵌入向量维度（d=64）")
    lightgcn_n_layers: int = Field(default=3, description="图传播层数（K=3）")
    lightgcn_dropout: float = Field(default=0.0, description="Dropout 丢弃率")
    lightgcn_learning_rate: float = Field(default=0.001, description="优化器学习率")
    lightgcn_weight_decay: float = Field(default=1e-4, description="L2 权重衰减系数")

    # SBERT 参数
    sbert_model_name: str = Field(default="all-MiniLM-L6-v2", description="Sentence-BERT 预训练模型名称")
    sbert_embedding_dim: int = Field(default=384, description="SBERT 嵌入向量维度")
    sbert_use_faiss: bool = Field(default=True, description="是否启用 FAISS 加速近似最近邻搜索")

    # 排序融合权重
    ranking_lightgcn_weight: float = Field(default=0.4, description="LightGCN 图相似度权重")
    ranking_sbert_weight: float = Field(default=0.3, description="SBERT 语义相似度权重")
    ranking_skill_coverage_weight: float = Field(default=0.3, description="技能覆盖率权重")

    # LLM 参数
    llm_model_name: str = Field(default="qwen-2.5-simulated", description="LLM 模拟器模型名")
    llm_temperature: float = Field(default=0.3, description="采样温度（越低越确定）")
    llm_max_tokens: int = Field(default=1000, description="LLM 生成的最大令牌数")


class DataConfig(BaseModel):
    """数据集配置组。

    控制模拟数据的规模、交互率、测试集划分比例等，
    仅用于开发/测试阶段。生产环境使用真实数据加载器替换。
    """

    # 数据集规模
    n_users: int = Field(default=20, description="模拟数据中的用户数量")
    n_jobs: int = Field(default=50, description="模拟数据中的岗位数量")
    n_skills: int = Field(default=21, description="模拟数据中的技能数量")

    # 交互率
    application_rate: float = Field(default=0.2, description="用户投递岗位的概率")
    interaction_rate: float = Field(default=0.3, description="用户与岗位交互（浏览、点击、收藏等）的概率")

    # 训练/测试集划分
    test_ratio: float = Field(default=0.2, description="测试集占交互数据的比例")

    # 图结构参数
    min_interactions: int = Field(default=1, description="用户/岗位被包含在交互图中的最小交互次数阈值")


class SystemConfig(BaseModel):
    """系统运行配置组。

    控制设备分配、随机种子、日志级别、系统路径和高性能训练参数。
    """

    device: DeviceType = Field(default=DeviceType.CPU, description="模型训练设备（CPU/CUDA/MPS）")
    random_seed: int = Field(default=42, description="随机种子，保证实验可重复性")
    log_level: str = Field(default="INFO", description="日志级别")

    # 系统路径
    data_dir: str = Field(default="data", description="数据存储目录")
    model_dir: str = Field(default="models", description="模型权重存储目录")
    logs_dir: str = Field(default="logs", description="日志文件目录")

    # 训练性能参数
    num_workers: int = Field(default=0, description="数据加载工作线程数")
    batch_size: int = Field(default=1024, description="训练批次大小")


class Settings(BaseModel):
    """顶层配置容器，聚合 Model/Data/System 三个配置组。

    提供单例模式访问、字典转换和动态更新功能。
    """

    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)

    # 单例实例引用
    _instance: Optional['Settings'] = None

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def get_default(cls) -> 'Settings':
        """获取默认配置实例。"""
        return cls()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Settings':
        """从字典构建 Settings 对象。"""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """将 Settings 对象序列化为字典。"""
        return {
            "model": self.model.dict(),
            "data": self.data.dict(),
            "system": self.system.dict()
        }

    def update(self, **kwargs) -> None:
        """动态更新配置。

        自动在 model/data/system 三个子配置组中查找并设置属性值。
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif hasattr(self.model, key):
                setattr(self.model, key, value)
            elif hasattr(self.data, key):
                setattr(self.data, key, value)
            elif hasattr(self.system, key):
                setattr(self.system, key, value)



# 全局配置单例
_settings = Settings.get_default()


def get_settings() -> Settings:
    """获取全局配置单例实例。

    线程安全：在模块加载阶段由 Python 保证（GIL 下模块只加载一次）。
    """
    return _settings


def update_settings(**kwargs) -> None:
    """更新全局配置。

    Args:
        **kwargs: 任意关键字参数，键名需匹配 Model/Data/System 子组属性
    """
    _settings.update(**kwargs)