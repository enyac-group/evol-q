from models import BIT_TYPE_DICT


class Config:

    def __init__(self, args):
        '''
        ptf stands for Power-of-Two Factor activation quantization for Integer Layernorm.
        lis stands for Log-Int-Softmax.
        These two are proposed in our "FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer".
        '''
        self.BIT_TYPE_W = BIT_TYPE_DICT[args.w_bit_type]
        # self.BIT_TYPE_W_LIN = BIT_TYPE_DICT["uint8"]
        self.BIT_TYPE_A = BIT_TYPE_DICT[args.a_bit_type]

        self.OBSERVER_W = 'minmax'
        self.OBSERVER_A = args.quant_method
        self.BCORR_W = args.bias_corr
        self.OBSERVER_W_CONV = 'minmax'

        self.QUANTIZER_W = 'uniform'
        self.QUANTIZER_A = 'uniform'
        self.QUANTIZER_A_LN = 'uniform'

        self.CALIBRATION_MODE_W = 'channel_wise'
        self.CALIBRATION_MODE_W_CONV = 'channel_wise'
        self.CALIBRATION_MODE_A = 'layer_wise'
        self.CALIBRATION_MODE_S = 'layer_wise'

        if args.lis:
            self.INT_SOFTMAX = True
            self.BIT_TYPE_S = BIT_TYPE_DICT['uint4']
            self.OBSERVER_S = 'minmax'
            self.QUANTIZER_S = 'log2'
        else:
            self.INT_SOFTMAX = False
            self.BIT_TYPE_S = BIT_TYPE_DICT['uint8']
            self.OBSERVER_S = self.OBSERVER_A
            self.QUANTIZER_S = self.QUANTIZER_A
        if args.ptf:
            self.INT_NORM = True
            self.OBSERVER_A_LN = 'ptf'
            self.CALIBRATION_MODE_A_LN = 'channel_wise'
        else:
            self.INT_NORM = False
            self.OBSERVER_A_LN = self.OBSERVER_A
            self.CALIBRATION_MODE_A_LN = self.CALIBRATION_MODE_A
