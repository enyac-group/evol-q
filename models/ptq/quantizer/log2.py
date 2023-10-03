# Adapted from MEGVII Inc.
# https://github.com/megvii-research/FQ-ViT/tree/main

import torch
import torch.nn.functional as F
from .base import BaseQuantizer

# class STEFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input):
#         return (input > 0).float()

#     @staticmethod
#     def backward(ctx, grad_output):
#         return F.hardtanh(grad_output)
    
def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x

class Log2Quantizer(BaseQuantizer):

    def __init__(self, bit_type, observer, module_type, bcorr_weights):
        super(Log2Quantizer, self).__init__(
            bit_type,
            observer,
            module_type,
            bcorr_weights
        )
        self.softmax_mask = None

    def quant(self, inputs):
        rounds = round_ste(-1 * inputs.log2())
        self.softmax_mask = rounds >= 2**self.bit_type.bits
        outputs = torch.clamp(rounds, 0, 2**self.bit_type.bits - 1)
        return outputs

    def dequantize(self, inputs):
        outputs = 2**(-1 * inputs)
        outputs[self.softmax_mask] = 0
        return outputs
