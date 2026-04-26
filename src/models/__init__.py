"""
Additional model architectures for the job recommendation system.
"""

from .gat import GraphAttentionNetwork, GATLayer, MultiHeadGATLayer

__all__ = ["GraphAttentionNetwork", "GATLayer", "MultiHeadGATLayer"]
