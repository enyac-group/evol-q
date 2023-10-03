# Adapted from MEGVII Inc.
# https://github.com/megvii-research/FQ-ViT/tree/main

import torch
import torch.nn as nn


class BaseQuantizer(nn.Module):

    def __init__(self, bit_type, observer, module_type, bcorr_weights=False):
        super(BaseQuantizer, self).__init__()
        self.bit_type = bit_type
        self.observer = observer
        self.module_type = module_type
        self.bcorr_weights = bcorr_weights

    def get_reshape_range(self, inputs):
        range_shape = None
        if self.module_type == 'conv_weight':
            range_shape = (-1, 1, 1, 1)
        elif self.module_type == 'linear_weight':
            range_shape = (-1, 1)
        elif self.module_type == 'activation':
            if len(inputs.shape) == 2:
                range_shape = (1, -1)
            elif len(inputs.shape) == 3:
                range_shape = (1, 1, -1)
            elif len(inputs.shape) == 4:
                range_shape = (1, -1, 1, 1)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return range_shape

    def update_quantization_params(self, *args, **kwargs):
        pass

    def bias_correction(self, inputs, weight_q):
        # adaptedfrom CNN Quantization for Rapid Deployment
        # https://github.com/submission2019/cnn-quantization/blob/master/pytorch_quantizer/quantization/inference/inference_quantization_manager.py

        bias_q = weight_q.reshape(weight_q.shape[0], -1).mean(-1)
        bias_q = bias_q.reshape(bias_q.numel(), 1, 1, 1) if len(weight_q.shape) == 4 else bias_q.reshape(bias_q.numel(), 1)
        bias_orig = inputs.reshape(inputs.shape[0], -1).mean(-1)
        bias_orig = bias_orig.reshape(bias_orig.numel(), 1, 1, 1) if len(weight_q.shape) == 4 else bias_orig.reshape(bias_orig.numel(), 1)

        eps = torch.tensor([1e-8]).to(weight_q.device)
        var_corr = inputs.reshape(inputs.shape[0], -1).std(dim=-1)
        var_corr = torch.divide(var_corr, weight_q.reshape(weight_q.shape[0], -1).std(dim=-1) + eps)
        var_corr = (var_corr.reshape(var_corr.numel(), 1, 1 , 1) if len(weight_q.shape) == 4 else var_corr.reshape(var_corr.numel(), 1))
        
        # correct (1) variance and (2) mean
        outputs = (weight_q - bias_q)*var_corr + bias_q
        outputs = outputs - bias_q + bias_orig
        return outputs

    def quant(self, inputs, scale=None, zero_point=None):
        raise NotImplementedError

    def dequantize(self, inputs, scale=None, zero_point=None):
        raise NotImplementedError

    def forward(self, inputs):
        outputs = self.quant(inputs)
        outputs = self.dequantize(outputs)
        if self.bcorr_weights:
            outputs = self.bias_correction(inputs, outputs)
        return outputs
