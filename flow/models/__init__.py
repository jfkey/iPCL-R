#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   __init__.py
@Time    :   2025/01/14
@Author  :   Junfeng Liu
@Version :   1.0
@Desc    :   Geometry-aware models for EDA routing trajectory generation.
             Includes Fourier Position Embedding, Geometric Attention (GeoPE/LARA),
             and the combined GeoT5Gemma model.
"""

from .position_embedding import FourierPositionEmbedding
from .geometric_attention import GeometricPositionEmbedding, LieAlgebraRelativeAttention
from .geo_t5gemma import GeoT5GemmaForConditionalGeneration

__all__ = [
    "FourierPositionEmbedding",
    "GeometricPositionEmbedding",
    "LieAlgebraRelativeAttention",
    "GeoT5GemmaForConditionalGeneration",
]
