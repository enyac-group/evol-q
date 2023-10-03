# LeViT Implementation adapted from Facebook, Inc.
# https://github.com/facebookresearch/LeViT

import torch
import itertools
from collections import OrderedDict
from .utils import replace_batchnorm, collect_act
import copy

from timm.models.vision_transformer import trunc_normal_
from timm.models.registry import register_model
from .ptq import QAct, QConv2d, QIntLayerNorm, QIntSoftmax, QLinear
from .base_quant import BaseQuant
import numpy as np


__all__ = [
    'levit_128s', 'levit_128', 'levit_192', 'levit_256', 'levit_384'
]

specification = {
    'LeViT_128S': {
        'C': '128_256_384', 'D': 16, 'N': '4_6_8', 'X': '2_3_4', 'drop_path': 0,
        'weights': 'https://dl.fbaipublicfiles.com/LeViT/LeViT-128S-96703c44.pth'},
    'LeViT_128': {
        'C': '128_256_384', 'D': 16, 'N': '4_8_12', 'X': '4_4_4', 'drop_path': 0,
        'weights': 'https://dl.fbaipublicfiles.com/LeViT/LeViT-128-b88c2750.pth'},
    'LeViT_192': {
        'C': '192_288_384', 'D': 32, 'N': '3_5_6', 'X': '4_4_4', 'drop_path': 0,
        'weights': 'https://dl.fbaipublicfiles.com/LeViT/LeViT-192-92712e41.pth'},
    'LeViT_256': {
        'C': '256_384_512', 'D': 32, 'N': '4_6_8', 'X': '4_4_4', 'drop_path': 0,
        'weights': 'https://dl.fbaipublicfiles.com/LeViT/LeViT-256-13b5763e.pth'},
    'LeViT_384': {
        'C': '384_512_768', 'D': 32, 'N': '6_9_12', 'X': '4_4_4', 'drop_path': 0.1,
        'weights': 'https://dl.fbaipublicfiles.com/LeViT/LeViT-384-9bdaf2e2.pth'},
}

# __all__ = [specification.keys()]


@register_model
def LeViT_128S(num_classes=1000, distillation=True,
               pretrained=False, fuse=False, quant=False,
               calibrate=False, cfg=None):
    return model_factory(**specification['LeViT_128S'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, fuse=fuse,
                         quant=quant, calibrate=calibrate, cfg=cfg)


@register_model
def LeViT_128(num_classes=1000, distillation=True,
              pretrained=False, fuse=False, quant=False,
               calibrate=False, cfg=None):
    return model_factory(**specification['LeViT_128'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, fuse=fuse,
                         quant=quant, calibrate=calibrate, cfg=cfg)


@register_model
def LeViT_192(num_classes=1000, distillation=True,
              pretrained=False, fuse=False, quant=False,
               calibrate=False, cfg=None):
    return model_factory(**specification['LeViT_192'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, fuse=fuse,
                         quant=quant, calibrate=calibrate, cfg=cfg)


@register_model
def LeViT_256(num_classes=1000, distillation=True,
              pretrained=False, fuse=False, quant=False,
               calibrate=False, cfg=None):
    return model_factory(**specification['LeViT_256'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, fuse=fuse,
                         quant=quant, calibrate=calibrate, cfg=cfg)


@register_model
def LeViT_384(num_classes=1000, distillation=True,
              pretrained=False, fuse=False, quant=False,
               calibrate=False, cfg=None):
    return model_factory(**specification['LeViT_384'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, fuse=fuse,
                         quant=quant, calibrate=calibrate, cfg=cfg)

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000, quant=False, calibrate=False, cfg=None):
        super().__init__()

        self.quant = quant
        self.calibrate = calibrate
        self.cfg = cfg

        self.add_module('c', QConv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False, 
            quant=self.quant, calibrate=self.calibrate,
            bit_type=self.cfg.BIT_TYPE_W,
            calibration_mode=self.cfg.CALIBRATION_MODE_W_CONV,
            observer_str=self.cfg.OBSERVER_W_CONV,
            quantizer_str=self.cfg.QUANTIZER_W,
            bcorr_weights=self.cfg.BCORR_W))
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = QConv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            quant=self.quant, calibrate=self.calibrate,
            bit_type=self.cfg.BIT_TYPE_W,
            calibration_mode=self.cfg.CALIBRATION_MODE_W_CONV,
            observer_str=self.cfg.OBSERVER_W_CONV,
            quantizer_str=self.cfg.QUANTIZER_W,
            bcorr_weights=self.cfg.BCORR_W)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Linear_BN(torch.nn.Sequential):
    def __init__(self, a, b, bn_weight_init=1, resolution=-100000, quant=False, calibrate=False, cfg=None):
        super().__init__()

        self.quant = quant
        self.calibrate = calibrate
        self.cfg = cfg

        self.add_module('c', QLinear(a, b, bias=False, 
            quant=self.quant, calibrate=self.calibrate,
            bit_type=self.cfg.BIT_TYPE_W,
            calibration_mode=self.cfg.CALIBRATION_MODE_W,
            observer_str=self.cfg.OBSERVER_W,
            quantizer_str=self.cfg.QUANTIZER_W,
            bcorr_weights=self.cfg.BCORR_W))
        bn = torch.nn.BatchNorm1d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

    @torch.no_grad()
    def fuse(self):
        l, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[:, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = QLinear(w.size(1), w.size(0), 
            quant=self.quant, calibrate=self.calibrate,
            bit_type=self.cfg.BIT_TYPE_W,
            calibration_mode=self.cfg.CALIBRATION_MODE_W,
            observer_str=self.cfg.OBSERVER_W,
            quantizer_str=self.cfg.QUANTIZER_W,
            bcorr_weights=self.cfg.BCORR_W)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

    def forward(self, x):
        l, bn = self._modules.values()
        x = l(x)
        return bn(x.flatten(0, 1)).reshape_as(x)

class Linear_BN1(torch.nn.Sequential):
    def __init__(self, a, b, bn_weight_init=1, resolution=-100000, quant=False, calibrate=False, cfg=None):
        super().__init__()

        self.quant = quant
        self.calibrate = calibrate
        self.cfg = cfg

        self.add_module('c', QLinear(a, b, bias=False, 
            quant=self.quant, calibrate=self.calibrate,
            bit_type=self.cfg.BIT_TYPE_W,
            calibration_mode=self.cfg.CALIBRATION_MODE_W,
            observer_str=self.cfg.OBSERVER_W,
            quantizer_str=self.cfg.QUANTIZER_W,
            bcorr_weights=self.cfg.BCORR_W))
        bn = torch.nn.BatchNorm1d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

    @torch.no_grad()
    def fuse(self):
        l, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[:, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = QLinear(w.size(1), w.size(0), 
            quant=self.quant, calibrate=self.calibrate,
            bit_type=self.cfg.BIT_TYPE_W_LIN,
            calibration_mode=self.cfg.CALIBRATION_MODE_W,
            observer_str=self.cfg.OBSERVER_W,
            quantizer_str=self.cfg.QUANTIZER_W,
            bcorr_weights=self.cfg.BCORR_W)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

    def forward(self, x):
        l, bn = self._modules.values()
        x = l(x)
        return bn(x.flatten(0, 1)).reshape_as(x)


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02, quant=False, calibrate=False, cfg=None):
        super().__init__()

        self.quant = quant
        self.calibrate = calibrate
        self.cfg = cfg

        self.add_module('bn', torch.nn.BatchNorm1d(a))
        l = QLinear(a, b, bias=bias, 
            quant=self.quant, calibrate=self.calibrate,
            bit_type=self.cfg.BIT_TYPE_W,
            calibration_mode=self.cfg.CALIBRATION_MODE_W,
            observer_str=self.cfg.OBSERVER_W,
            quantizer_str=self.cfg.QUANTIZER_W,
            bcorr_weights=self.cfg.BCORR_W)
        trunc_normal_(l.weight, std=std)
        if bias:
            torch.nn.init.constant_(l.bias, 0)
        self.add_module('l', l)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = QLinear(w.size(1), w.size(0), 
            quant=self.quant, calibrate=self.calibrate,
            bit_type=self.cfg.BIT_TYPE_W,
            calibration_mode=self.cfg.CALIBRATION_MODE_W,
            observer_str=self.cfg.OBSERVER_W,
            quantizer_str=self.cfg.QUANTIZER_W,
            bcorr_weights=self.cfg.BCORR_W)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


def b16(n, activation, resolution=224, quant=False, calibrate=False, cfg=None):
    return torch.nn.Sequential(OrderedDict([
        ('0', Conv2d_BN(3, n // 8, 3, 2, 1, resolution=resolution, quant=quant, calibrate=calibrate, cfg=cfg)),
        ('qact0', QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A_LN, observer_str=cfg.OBSERVER_A_LN, quantizer_str=cfg.QUANTIZER_A_LN)), 
        ('1', activation()),
        ('qact1', QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A_LN, observer_str=cfg.OBSERVER_A_LN, quantizer_str=cfg.QUANTIZER_A_LN)),
        ('2', Conv2d_BN(n // 8, n // 4, 3, 2, 1, resolution=resolution // 2, quant=quant, calibrate=calibrate, cfg=cfg)),
        ('qact2', QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A_LN, observer_str=cfg.OBSERVER_A_LN, quantizer_str=cfg.QUANTIZER_A_LN)), 
        ('3', activation()),
        ('qact3', QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A_LN, observer_str=cfg.OBSERVER_A_LN, quantizer_str=cfg.QUANTIZER_A_LN)), 
        ('4', Conv2d_BN(n // 4, n // 2, 3, 2, 1, resolution=resolution // 4, quant=quant, calibrate=calibrate, cfg=cfg)),
        ('qact4', QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A_LN, observer_str=cfg.OBSERVER_A_LN, quantizer_str=cfg.QUANTIZER_A_LN)), 
        ('5', activation()),
        ('qact5', QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A_LN, observer_str=cfg.OBSERVER_A_LN, quantizer_str=cfg.QUANTIZER_A_LN)), 
        ('6',Conv2d_BN(n // 2, n, 3, 2, 1, resolution=resolution // 8, quant=quant, calibrate=calibrate, cfg=cfg)),
        ('qact6', QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A_LN, observer_str=cfg.OBSERVER_A_LN, quantizer_str=cfg.QUANTIZER_A_LN)) 
        ]))


class Residual(torch.nn.Module):
    def __init__(self, m, drop, quant=False, calibrate=False, cfg=None):
        super().__init__()
        self.m = m
        self.drop = drop

        self.qact0 = QAct(quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_A,
            calibration_mode=cfg.CALIBRATION_MODE_A,
            observer_str=cfg.OBSERVER_A,
            quantizer_str=cfg.QUANTIZER_A)

        self.qact_out = QAct(quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_A,
            calibration_mode=cfg.CALIBRATION_MODE_A_LN,
            observer_str=cfg.OBSERVER_A_LN,
            quantizer_str=cfg.QUANTIZER_A_LN)

    def forward(self, x):

        # x = self.qact0(x)

        if self.training and self.drop > 0:
            x = x + self.m(x) * torch.rand(x.size(0), 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            x =  x + self.m(x)

        return self.qact_out(x)


class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 activation=None,
                 resolution=14, quant=False, calibrate=False, cfg=None):
        super().__init__()
        self.calibrate_stage2 = False
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.qkv = Linear_BN(dim, h, resolution=resolution, quant=quant, calibrate=calibrate, cfg=cfg)
        self.proj = torch.nn.Sequential(OrderedDict([
                ('0', activation()), 
                # ('qact0', QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A, observer_str=cfg.OBSERVER_A, quantizer_str=cfg.QUANTIZER_A)),
                ('1', Linear_BN(self.dh, dim, bn_weight_init=0, resolution=resolution, quant=quant, calibrate=calibrate, cfg=cfg))
        ]))

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))


        self.softmax = QIntSoftmax(
            log_i_softmax=cfg.INT_SOFTMAX,
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_S,
            calibration_mode=cfg.CALIBRATION_MODE_S,
            observer_str=cfg.OBSERVER_S,
            quantizer_str=cfg.QUANTIZER_S)

        self.qact_q = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)

        self.qact_k = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)
        
        self.qact_v = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)

        self.qact_attn1 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)

        self.qact_attn2 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)

        self.qact_softmax = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)

        self.qact1 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)
        
        self.qact_out = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)
        
        self.handle = self.register_forward_hook(collect_act)
        self.inputs = []
        self.outputs = []
        
    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,N,C)

        B, N, C = x.shape
        qkv = self.qkv(x)

        q, k, v = qkv.view(B, N, self.num_heads, -
                           1).split([self.key_dim, self.key_dim, self.d], dim=3)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        q = self.qact_q(q)
        k = self.qact_k(k)
        v = self.qact_v(v)

        attn = (q @ k.transpose(-2, -1)) 
        
        attn = self.qact_attn1(attn)
            
        attn = attn * self.scale + (self.attention_biases[:, self.attention_bias_idxs] if self.training else self.ab)
        attn = self.qact_attn2(attn)
        attn = attn.softmax(dim=-1)
        attn = self.qact_softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.qact1(x)
        x = self.proj(x)
        
        x = self.qact_out(x)

        return x


class Subsample(torch.nn.Module):
    def __init__(self, stride, resolution):
        super().__init__()
        self.stride = stride
        self.resolution = resolution

    def forward(self, x):
        B, N, C = x.shape
        x = x.view(B, self.resolution, self.resolution, C)[
            :, ::self.stride, ::self.stride].reshape(B, -1, C)
        return x


class AttentionSubsample(torch.nn.Module):
    def __init__(self, in_dim, out_dim, key_dim, num_heads=8,
                 attn_ratio=2,
                 activation=None,
                 stride=2,
                 resolution=14, resolution_=7, quant=False, calibrate=False, cfg=None):
        super().__init__()
        self.calibrate_stage2 = False
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * self.num_heads
        self.attn_ratio = attn_ratio
        self.resolution_ = resolution_
        self.resolution_2 = resolution_**2
        h = self.dh + nh_kd
        self.kv = Linear_BN(in_dim, h, resolution=resolution, quant=quant, calibrate=calibrate, cfg=cfg)

        self.q = torch.nn.Sequential(
            Subsample(stride, resolution),
            Linear_BN(in_dim, nh_kd, resolution=resolution_, quant=quant, calibrate=calibrate, cfg=cfg))

        self.proj = torch.nn.Sequential(OrderedDict([
            ('0', activation()), 
            ('qact0', QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A, observer_str=cfg.OBSERVER_A, quantizer_str=cfg.QUANTIZER_A)),
            ('1', Linear_BN(
            self.dh, out_dim, resolution=resolution_, quant=quant, calibrate=calibrate, cfg=cfg))
            ]))

        self.stride = stride
        self.resolution = resolution
        points = list(itertools.product(range(resolution), range(resolution)))
        points_ = list(itertools.product(
            range(resolution_), range(resolution_)))
        N = len(points)
        N_ = len(points_)
        attention_offsets = {}
        idxs = []
        for p1 in points_:
            for p2 in points:
                size = 1
                offset = (
                    abs(p1[0] * stride - p2[0] + (size - 1) / 2),
                    abs(p1[1] * stride - p2[1] + (size - 1) / 2))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N_, N))

        self.softmax = QIntSoftmax(
            log_i_softmax=cfg.INT_SOFTMAX,
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_S,
            calibration_mode=cfg.CALIBRATION_MODE_S,
            observer_str=cfg.OBSERVER_S,
            quantizer_str=cfg.QUANTIZER_S)

        self.qact_q = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)

        self.qact_k = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)

        self.qact_v = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)

        self.qact_attn1 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)

        self.qact_attn2 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)
        
        self.qact_softmax = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)

        self.qact1 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)

        self.qact_out = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)

        self.handle = self.register_forward_hook(collect_act)
        self.inputs = []
        self.outputs = []

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, N, C = x.shape
        k, v = self.kv(x).view(B, N, self.num_heads, -
                               1).split([self.key_dim, self.d], dim=3)
        k = k.permute(0, 2, 1, 3)  # BHNC
        v = v.permute(0, 2, 1, 3)  # BHNC
        q = self.q(x).view(B, self.resolution_2, self.num_heads,
                           self.key_dim).permute(0, 2, 1, 3)

        q = self.qact_q(q)
        k = self.qact_k(k)
        v = self.qact_v(v)

        attn = (q @ k.transpose(-2, -1))
        attn = self.qact_attn1(attn)
        attn = attn * self.scale + \
            (self.attention_biases[:, self.attention_bias_idxs]
             if self.training else self.ab)

        attn = self.qact_attn2(attn)
        attn = attn.softmax(dim=-1)
        # attn = self.softmax(attn, self.qact_attn.quantizer.scale)
        attn = self.qact_softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, self.dh)
        x = self.qact1(x)
        x = self.proj(x)
        x = self.qact_out(x)

        return x


class LeViT(BaseQuant):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=[192],
                 key_dim=[64],
                 depth=[12],
                 num_heads=[3],
                 attn_ratio=[2],
                 mlp_ratio=[2],
                 hybrid_backbone=None,
                 down_ops=[],
                 attention_activation=torch.nn.Hardswish,
                 mlp_activation=torch.nn.Hardswish,
                 distillation=True,
                 drop_path=0,
                 quant=False,
                 calibrate=False, cfg=None
               ):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = embed_dim[-1]
        self.embed_dim = embed_dim
        self.distillation = distillation

        self.patch_embed = hybrid_backbone

        self.blocks = []
        down_ops.append([''])
        resolution = img_size // patch_size
        for i, (ed, kd, dpth, nh, ar, mr, do) in enumerate(
                zip(embed_dim, key_dim, depth, num_heads, attn_ratio, mlp_ratio, down_ops)):
            for _ in range(dpth):
                self.blocks.append(
                    Residual(Attention(
                        ed, kd, nh,
                        attn_ratio=ar,
                        activation=attention_activation,
                        resolution=resolution,
                        quant=quant, calibrate=calibrate, cfg=cfg
                    ), drop_path, quant=quant, calibrate=calibrate, cfg=cfg))
                if mr > 0:
                    h = int(ed * mr)
                    self.blocks.append(
                        Residual(torch.nn.Sequential(OrderedDict([
                            ('0', Linear_BN(ed, h, resolution=resolution, quant=quant, calibrate=calibrate, cfg=cfg)),
                            ('qact0', QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A, observer_str=cfg.OBSERVER_A, quantizer_str=cfg.QUANTIZER_A)),
                            ('1', mlp_activation()),
                            ('qact1', QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A, observer_str=cfg.OBSERVER_A, quantizer_str=cfg.QUANTIZER_A)),
                            ('2', Linear_BN(h, ed, bn_weight_init=0,
                                      resolution=resolution, quant=quant, calibrate=calibrate, cfg=cfg)),
                            # ('qact2', QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A, observer_str=cfg.OBSERVER_A, quantizer_str=cfg.QUANTIZER_A)),
                        ])), drop_path, quant=quant, calibrate=calibrate, cfg=cfg))
            if do[0] == 'Subsample':
                #('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
                resolution_ = (resolution - 1) // do[5] + 1
                self.blocks.append(
                    AttentionSubsample(
                        *embed_dim[i:i + 2], key_dim=do[1], num_heads=do[2],
                        attn_ratio=do[3],
                        activation=attention_activation,
                        stride=do[5],
                        resolution=resolution,
                        resolution_=resolution_, quant=quant, calibrate=calibrate, cfg=cfg))
                resolution = resolution_

                if do[4] > 0:  # mlp_ratio
                    h = int(embed_dim[i + 1] * do[4])
                    self.blocks.append(
                        Residual(torch.nn.Sequential(OrderedDict([
                            ('0', Linear_BN(embed_dim[i + 1], h,
                                      resolution=resolution, quant=quant, calibrate=calibrate, cfg=cfg)),
                            ('qact0', QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A, observer_str=cfg.OBSERVER_A, quantizer_str=cfg.QUANTIZER_A)),
                            ('1', mlp_activation()),
                            ('qact1', QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A, observer_str=cfg.OBSERVER_A, quantizer_str=cfg.QUANTIZER_A)),
                            ('2', Linear_BN(
                                h, embed_dim[i + 1], bn_weight_init=0, resolution=resolution, quant=quant, calibrate=calibrate, cfg=cfg)),
                            # ('qact2', QAct(quant=quant, calibrate=calibrate, bit_type=cfg.BIT_TYPE_A, calibration_mode=cfg.CALIBRATION_MODE_A, observer_str=cfg.OBSERVER_A, quantizer_str=cfg.QUANTIZER_A)),
                        ])), drop_path, quant=quant, calibrate=calibrate, cfg=cfg))
        self.blocks = torch.nn.Sequential(*self.blocks)

        # Classifier head
        self.head = BN_Linear(
            embed_dim[-1], num_classes, quant=quant, calibrate=calibrate, cfg=cfg) if num_classes > 0 else torch.nn.Identity()
        if distillation:
            self.head_dist = BN_Linear(
                embed_dim[-1], num_classes, quant=quant, calibrate=calibrate, cfg=cfg) if num_classes > 0 else torch.nn.Identity()

        self.qact_input = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)

        self.qact_embed = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)

        self.qact_postblocks = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)

        self.qact_mean = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)

        self.qact0 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)

        self.qact1 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)

        self.qact_out = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)

        self.cfg = cfg
        self.quant = quant
        self.calibrate = calibrate

    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    def forward(self, x):
        # x = self.qact_input(x)
        x = self.patch_embed(x)
        x = self.qact_embed(x)

        x = x.flatten(2).transpose(1, 2)

        x = self.blocks(x)
        x = self.qact_postblocks(x)
        x = x.mean(1)
        x = self.qact_mean(x)

        if self.distillation:
            x = self.qact0(self.head(x)), self.qact1(self.head_dist(x))

            if not self.training:
                x = (x[0] + x[1]) / 2
                x = self.qact_out(x)
            else:
                x = self.qact_out(x[0])
        else:
            x = self.head(x)
            x = self.qact_out(x)
        return x

    # overwrite inhereted version from BaseQuant
    def model_close_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.calibrate = False
            if type(m) in [Attention, AttentionSubsample]:
                m.handle.remove()



def model_factory(C, D, X, N, drop_path, weights,
                  num_classes, distillation, pretrained, fuse, quant=False,
                calibrate=False, cfg=None):
    embed_dim = [int(x) for x in C.split('_')]
    num_heads = [int(x) for x in N.split('_')]
    depth = [int(x) for x in X.split('_')]
    act = torch.nn.Hardswish
    model = LeViT(
        patch_size=16,
        embed_dim=embed_dim,
        num_heads=num_heads,
        key_dim=[D] * 3,
        depth=depth,
        attn_ratio=[2, 2, 2],
        mlp_ratio=[2, 2, 2],
        down_ops=[
            #('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
            ['Subsample', D, embed_dim[0] // D, 4, 2, 2],
            ['Subsample', D, embed_dim[1] // D, 4, 2, 2],
        ],
        attention_activation=act,
        mlp_activation=act,
        hybrid_backbone=b16(embed_dim[0], activation=act, calibrate=calibrate, quant=quant, cfg=cfg),
        num_classes=num_classes,
        drop_path=drop_path,
        distillation=distillation,
        quant=quant,
        calibrate=calibrate, cfg=cfg
    )
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            weights, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    if fuse:
        print("replacing batchnorm")
        replace_batchnorm(model)

    return model

def levit_128s(pretrained=False,
                         quant=False,
                         calibrate=False,
                         cfg=None,
                         **kwargs):
    name = "LeViT_128S"
    net = globals()[name](fuse=True, pretrained=pretrained, quant=quant, calibrate=calibrate, cfg=cfg)
    net.eval()
    return net

def levit_128(pretrained=False,
                         quant=False,
                         calibrate=False,
                         cfg=None,
                         **kwargs):
    name = "LeViT_128"
    net = globals()[name](fuse=True, pretrained=pretrained, quant=quant, calibrate=calibrate, cfg=cfg)
    net.eval()
    return net

def levit_192(pretrained=False,
                         quant=False,
                         calibrate=False,
                         cfg=None,
                         **kwargs):
    name = "LeViT_192"
    net = globals()[name](fuse=True, pretrained=pretrained, quant=quant, calibrate=calibrate, cfg=cfg)
    net.eval()
    return net

def levit_256(pretrained=False,
                         quant=False,
                         calibrate=False,
                         cfg=None,
                         **kwargs):
    name = "LeViT_256"
    net = globals()[name](fuse=True, pretrained=pretrained, quant=quant, calibrate=calibrate, cfg=cfg)
    net.eval()
    return net

def levit_384(pretrained=False,
                         quant=False,
                         calibrate=False,
                         cfg=None,
                         **kwargs):
    name = "LeViT_384"
    net = globals()[name](fuse=True, pretrained=pretrained, quant=quant, calibrate=calibrate, cfg=cfg)
    net.eval()
    return net


if __name__ == '__main__':
    for name in specification:
        net = globals()[name](fuse=True, pretrained=True)
        net.eval()
        net(torch.randn(4, 3, 224, 224))
        print(name,
              net.FLOPS, 'FLOPs',
              sum(p.numel() for p in net.parameters() if p.requires_grad), 'parameters')
