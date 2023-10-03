# Adapted from MEGVII Inc.
# https://github.com/megvii-research/FQ-ViT/tree/main

from .bit_type import BIT_TYPE_DICT
from .layers import QAct, QConv2d, QIntLayerNorm, QIntSoftmax, QLinear
