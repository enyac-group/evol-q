
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import time
from PIL import Image, ImageFilter, ImageOps
import math
import random
import torchvision.transforms.functional as tf
import torch.nn.functional as F
from misc import all_reduce_mean
class TwoCropsTransform:
    """Take two random crops of one image"""

    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        return [im1, im2]


class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)
    
def contrastive_loss(q, k, T=1.0):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long)).to(q.device)
        return nn.CrossEntropyLoss()(logits, labels) * (2 * T)

def get_preds(args, val_loader, model, criterion, device):

    # switch to evaluate mode
    model.eval()
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    fig, (ax1, ax2) = plt.subplots(nrows=2)
    targets = {}
    ts= []
    ps = []
    corr_mat = np.zeros([1,1000])
    for i, (data, target) in enumerate(val_loader):
        data = data.to(device)
        target = target.to(device)
        
        with torch.no_grad():
            output = model(data)
        
        _, pred = output.topk(1, 1, True, True)
        # correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        # for i in range(0,target.shape[0]):
        t = target.detach().cpu().numpy().flatten()
        p = pred.detach().cpu().numpy().flatten()
        for it in range(0,len(t)):
            if t[it] == p[it]:
                corr_mat[0,t[it]] = 1

        print(len(t), len(p))
 

    # corr_mat = corr_mat[:,50:450]
    
    from matplotlib.colors import ListedColormap
    cmapmine = ListedColormap(['b', 'w'], N=2)
    sns.heatmap(corr_mat, ax = ax2, cmap="Blues", cbar=False)

    model = torch.load(args.save_folder+"/model_layerwise.pt")
    model.model_quant()
    model.eval()
    corr_mat = np.zeros([1,1000])
    for i, (data, target) in enumerate(val_loader):
        data = data.to(device)
        target = target.to(device)
        
        with torch.no_grad():
            output = model(data)
        
        _, pred = output.topk(1, 1, True, True)
        # correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        # for i in range(0,target.shape[0]):
        t = target.detach().cpu().numpy().flatten()
        p = pred.detach().cpu().numpy().flatten()
        for it in range(0,len(t)):
            if t[it] == p[it]:
                corr_mat[0,t[it]] = 1

        print(len(t), len(p))
 

    # corr_mat = corr_mat[:,50:450]
    
    from matplotlib.colors import ListedColormap
    cmapmine = ListedColormap(['b', 'w'], N=2)
    sns.heatmap(corr_mat, ax = ax1, cmap="Blues", cbar=False)
    import matplotlib.ticker as ticker
    ax1.xaxis.set_major_locator(ticker.NullLocator())
    ax1.yaxis.set_major_locator(ticker.NullLocator())
    ax2.xaxis.set_major_locator(ticker.NullLocator())
    ax2.yaxis.set_major_locator(ticker.NullLocator())
    ax2.set_xlabel("Predicted Class")
    ax2.set_ylabel("CPT-V")
    ax1.set_ylabel("FQ-ViT")
    ax1.set_title("One-Hot Visualization of Validation Accuracy")
    # plt.axis('off')
    plt.savefig("ground_truth.png", dpi=300)
       

def validate(args, val_loader, model, criterion, device):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    val_start_time = end = time.time()
    for i, (data, target) in enumerate(val_loader):
        data = data.to(device)
        target = target.to(device)

        with torch.no_grad():
            output = model(data)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), data.size(0))
        top1.update(prec1.data.item(), data.size(0))
        top5.update(prec5.data.item(), data.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      i,
                      len(val_loader),
                      batch_time=batch_time,
                      loss=losses,
                      top1=top1,
                      top5=top5,
                  ))
    val_end_time = time.time()
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Time {time:.3f}'.
          format(top1=top1, top5=top5, time=val_end_time - val_start_time))

    return all_reduce_mean(losses.avg), all_reduce_mean(top1.avg), all_reduce_mean(top5.avg)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def build_transform(model_type, input_size=224,
                    interpolation='bicubic',
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    crop_pct=0.875):
    
    if model_type == 'deit' or model_type == 'levit':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        crop_pct = 0.875
    elif model_type == 'vit':
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        crop_pct = 0.9
    elif model_type == 'swin':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        crop_pct = 0.9
    else:
        raise NotImplementedError
    
    def _pil_interp(method):
        if method == 'bicubic':
            return Image.BICUBIC
        elif method == 'lanczos':
            return Image.LANCZOS
        elif method == 'hamming':
            return Image.HAMMING
        else:
            return Image.BILINEAR

    resize_im = input_size > 32
    t = []
    if resize_im:
        size = int(math.floor(input_size / crop_pct))
        ip = _pil_interp(interpolation)
        t.append(
            transforms.Resize(
                size,
                interpolation=ip),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

# functions from BRECQ
class GradSaverHook:
    def __init__(self, store_grad=True):
        self.store_grad = store_grad
        self.stop_backward = False
        self.grad_out = {}

    def __call__(self, module, grad_input, grad_output):
        if self.store_grad:
            self.grad_out[grad_output[0].device] = grad_output[0].detach()
        if self.stop_backward:
            raise Exception
        
class FisherLoss:
    def __call__(self,  pred, tgt, grad):
        rec_loss = ((pred - tgt).pow(2) * grad.pow(2)).sum(1).mean()
        return rec_loss
    
def eval_model(test_loader, calib_loader, model, args, print_freq=100):
    device = next(model.parameters()).device
    # else:
    model = model.to(device)
    if args.distributed:
        net = model.module
    else:
        net = model

    ce_loss = torch.nn.CrossEntropyLoss()
    calib_cross_entropy = AverageMeter()
    test_cross_entropy = AverageMeter()
    fim = AverageMeter()
    fisher_loss = FisherLoss()
    mse_loss = torch.nn.MSELoss()
    mean_squared = AverageMeter()
    cossim_loss = torch.nn.CosineSimilarity(dim=0)
    cos_similarity = AverageMeter()
    contrastive = AverageMeter()

    # switch to evaluate mode
    model.eval()
    net.model_quant()
    with torch.no_grad():
        for i, (images, target) in enumerate(test_loader):
            model.zero_grad()
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            test_ce = ce_loss(output, target).item()
            test_cross_entropy.update(test_ce, images.size(0))

    data_saver = GradSaverHook(True)

    with torch.enable_grad():
        for i, (images, target) in enumerate(calib_loader):
            handle = net.head.register_full_backward_hook(data_saver)
            model.zero_grad()
            images = images.to(device)
            target = target.to(device)

            # compute output
            net.model_quant()
            output = model(images)
            handle.remove()
            ce = ce_loss(output, target).item()
            calib_cross_entropy.update(ce, images.size(0))
            
            net.model_dequant()
            output_fp = model(images)
            
            loss = F.kl_div(F.log_softmax(output, dim=0), F.softmax(output_fp, dim=0), reduction='batchmean')
            loss.backward()

            # grad = model.head.weight.grad
            grad = torch.cat([k.to(device) for k in data_saver.grad_out.values()])
            assert torch.count_nonzero(grad) != 0
            fisher = fisher_loss(output, output_fp, grad).item()
            fim.update(fisher, images.size(0))
            
            loss = contrastive_loss(output, output_fp, T=1.0).item()
            contrastive.update(loss, images.size(0))

            mse = mse_loss(output, output_fp).item()
            mean_squared.update(mse, images.size(0))
            
            
            # transform output from [bsize, nclasses] to [bsize]
            topk=(1,)
            maxk = max(topk)
            batch_size = target.size(0)
            _, pred = output.topk(maxk, 1, True, True)
            output = pred.flatten().type(torch.cuda.FloatTensor)
            
            _, pred = output_fp.topk(maxk, 1, True, True)
            output_fp = pred.flatten().type(torch.cuda.FloatTensor)

            cossim = cossim_loss(output, output_fp).item()
            cos_similarity.update(cossim, images.size(0))
    
    # if args.distributed:
    return all_reduce_mean(test_cross_entropy.avg), all_reduce_mean(calib_cross_entropy.avg), all_reduce_mean(contrastive.avg), all_reduce_mean(mean_squared.avg), all_reduce_mean(cos_similarity.avg), all_reduce_mean(fim.avg)
    # else:
    #     return all_reduce_mean(test_cross_entropy.avg), all_reduce_mean(calib_cross_entropy.avg), all_reduce_mean(contrastive.avg), all_reduce_mean(mean_squared.avg, cos_similarity.avg, fim.avg