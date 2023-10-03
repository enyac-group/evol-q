import argparse
import os
from utils import *
from models import *
from joint_evol_opt import JointQuantization

parser = argparse.ArgumentParser(description='CPT-V')

parser.add_argument('model',
                    choices=[
                        'deit_tiny', 'deit_small', 'deit_base', 'vit_base',
                        'vit_large', 'swin_tiny', 'swin_small', 'swin_base',
                        'levit_128s', 'levit_128', 'levit_192', 'levit_256', 'levit_384'
                    ],
                    help='model')
parser.add_argument('data', metavar='DIR', help="ImageNet file path")
parser.add_argument('--save_folder', default=False, help='path for storing checkpoints and results')
parser.add_argument('--ptf', default=False, action='store_true', help="power of two activation quantization")
parser.add_argument('--lis', default=False, action='store_true', help="log-int-softmax from FQ-ViT. Not used in CPT-V initialization due to poor performance")
parser.add_argument('--bias-corr', default=False, action='store_true')
parser.add_argument('--mode', default="layerwise", choices=["fp_no_quant", "fq_vit", "fq++", "evolq", "e2e"])
parser.add_argument('--quant-method', default='minmax', choices=['minmax', 'ema', 'omse', 'percentile'], help="quantization scheme for initialized model")
parser.add_argument('--w_bit_type', default='int8', choices=['int3', 'uint3', 'uint4', 'uint8', 'int4', 'int8', 'fp32',])
parser.add_argument('--a_bit_type', default='uint8', choices=['uint4', 'uint8', 'int4', 'int8', 'fp32',])
parser.add_argument('--calib-batchsize', default=100, type=int, help='batchsize of calibration set')
parser.add_argument('--calib-size', default=1000, type=int, help="size of calibration dataset")
parser.add_argument('--val-batchsize', default=8, type=int, help='batchsize of validation set')
parser.add_argument('--num-workers',
                    default=16,
                    type=int,
                    help='number of data loading workers (default: 16)')
parser.add_argument('--device', default='cuda', type=str, help='device')
parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
parser.add_argument('--seed', default=0, type=int, help='seed')
parser.add_argument('--num_passes', default=10, type=int, help="number of passes across all blocks (P)")
parser.add_argument('--num_cycles', default=3, type=int, help="number of cycles per blocks (K)")
parser.add_argument('--temp', default=3.0, type=float, help='temperature')
parser.add_argument('--loss', default='contrastive', choices=['contrastive','mse', 'kl', 'cosine'], help="loss function for evolutionary search's fitness function")

def str2model(name):
    d = {
        'deit_tiny': deit_tiny_patch16_224,
        'deit_small': deit_small_patch16_224,
        'deit_base': deit_base_patch16_224,
        'vit_base': vit_base_patch16_224,
        'vit_large': vit_large_patch16_224,
        'swin_tiny': swin_tiny_patch4_window7_224,
        'swin_small': swin_small_patch4_window7_224,
        'swin_base': swin_base_patch4_window7_224,
        'levit_128s': levit_128s,
        'levit_128': levit_128,
        'levit_192': levit_192,
        'levit_256': levit_256,
        'levit_384': levit_384,
    }
    print('Model: %s' % d[name].__name__)
    return d[name]


def seed(seed=0):
    import os
    import random
    import sys

    import numpy as np
    import torch
    sys.setrecursionlimit(100000)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def main():
    args = parser.parse_args()
    seed(args.seed)

    device = torch.device(args.device)

    if args.mode == "fq_vit" or args.mode == "e2e":
        from config_fq import Config
    else:
        from config import Config

    cfg = Config(args)
    model = str2model(args.model)(pretrained=True, cfg=cfg)
    model = model.to(device)

    # Note: Different models have different strategies of data preprocessing.
    model_type = args.model.split('_')[0]

    train_transform = build_transform(model_type)
    val_transform = build_transform(model_type)

    # Data
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    val_dataset = datasets.ImageFolder(valdir, val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    # switch to evaluate mode
    model.eval()

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().to(device)

    if not args.mode == "fp_no_quant": #check if in a quantization mode
        train_dataset = datasets.ImageFolder(traindir, train_transform)
        _, calib_dataset = torch.utils.data.random_split(train_dataset, [len(train_dataset)-args.calib_size, args.calib_size])
        calib_loader = torch.utils.data.DataLoader(
            calib_dataset,
            batch_size=args.calib_batchsize,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        if args.mode == "fq++" or args.mode == "fq_vit" or args.mode == "e2e":
            
            model.model_open_calibrate()
            with torch.no_grad():

                for i, (image, target) in enumerate(calib_loader):
                    image = image.to(device)
                    if i == len(calib_loader) - 1:
                        # This is used for OMSE method to
                        # calculate minimum quantization error
                        model.model_open_last_calibrate()
                    model(image)
            model.model_close_calibrate()
            
            print("Saving Model... ")
            torch.save(model, args.save_folder+ "/model_layerwise.pt")
            model.model_quant()

            print('Validating layerwise quantization...')
            val_loss, val_prec1, val_prec5 = validate(args, val_loader, model,
                                                    criterion, device)
            with open(args.save_folder+"/layerwise.txt", "a") as f:
                f.write(str(val_prec1)+"\n")
                
        if args.mode == "evolq" or args.mode == "e2e":

            print("Loading Model...")
            model = torch.load(args.save_folder+"/model_layerwise.pt").to("cpu")
            optim = JointQuantization(model, calib_loader, device, args, val_loader=val_loader)
            model = optim.opt()

            print('Validating Evol-Q optimization...')
            model.model_quant()
            val_loss, val_prec1, val_prec5 = validate(args, val_loader, model,
                                                    criterion, device)
            with open(args.save_folder+"/evolq.txt", "w") as f:
                f.write(str(val_prec1)+"\n")
    else:

        print('Validating full precision...')
        val_loss, val_prec1, val_prec5 = validate(args, val_loader, model,
                                                  criterion, device)

if __name__ == '__main__':
    main()
