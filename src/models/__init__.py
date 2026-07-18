"""
Additional model architectures for the job recommendation system.
"""

from .gat import GATLayer, GraphAttentionNetwork, MultiHeadGATLayer

__all__ = ["GraphAttentionNetwork", "GATLayer", "MultiHeadGATLayer"]
from .bundle import ModelBundle, StaticSkillWeighter

__all__ = ["ModelBundle", "StaticSkillWeighter"]
