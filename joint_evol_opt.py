
from models.levit_quant import Attention, AttentionSubsample
from models.vit_quant import Attention_ViT
from models.swin_quant import WindowAttention
import numpy as np
from models import *
import torch
from utils import *
import heapq
import random
import time

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
 
class JointQuantization:
    def __init__(self, inf_model, calib_loader, device, args, val_loader=None):
        self.device = device
        inf_model = inf_model.to(device)
        self.inf_model = inf_model
        self.calib_loader = calib_loader
        self.calib_samples = len(calib_loader)
        self.criterion = torch.nn.CrossEntropyLoss() #.to(device)
        self.args = args
        if val_loader is not None:
            self.val_loader = val_loader
        self.inf_model.model_quant()
        self.inf_model.eval()


        self.mods_to_optimize = [QLinear]


    def loss(self, scales):
        self.set_scales(scales)
        
        res = 0
        for i, (x, _) in enumerate(self.calib_loader):
            x = x.cuda()

            with torch.no_grad(): 
                o_quant_x0 = self.inf_model(x)
                o_quant_x0 = o_quant_x0.to("cpu")

                o_fp = self.o_fps[i].to("cpu")

                if self.args.loss == "contrastive":
                    loss = contrastive_loss(o_quant_x0, o_fp, self.args.temp)
                elif self.args.loss == "mse":
                    loss = self.mse(o_quant_x0, o_fp)
                elif self.args.loss == "cosine":
                    loss = torch.sum(self.cos(o_quant_x0, o_fp))
                elif self.args.loss == "kl":
                    loss = self.kl(o_quant_x0, o_fp)
                res += float(loss.item())
                
                torch.cuda.empty_cache()
        res = res / len(self.calib_loader)
        return res

    def set_scales(self, scales):
        ind = 0
        for mod in self.m.modules():
            if type(mod) in self.mods_to_optimize:
                current_scale = scales[self.m.indices[ind]:self.m.indices[ind+1]]
                current_scale = current_scale.reshape(mod.quantizer.scale.shape)
                mod.quantizer.scale = torch.nn.Parameter(torch.Tensor(current_scale).to(mod.quantizer.scale.device))
                ind += 1

    def get_scales(self):
        scales = []

        index = 0
        self.m.indices = [index]

        for mod in self.m.modules():
            if type(mod) in self.mods_to_optimize:
                scale = mod.quantizer.scale
                scale = scale.cpu().detach().numpy().flatten()
                scales.append(scale)
                index += scale.size
                self.m.indices.append(index)

        return np.concatenate(scales)

    def mutate(self, scales):
        if self.bits == 4 or self.bits == 3:
            rng = 1e-3
        elif self.bits == 8:
            rng = 1e-4
        else:
            print("Range is not tested for # bits != 3, 4 or 8.")
            rng=1e-3

        perturbations = np.random.uniform(low=-1*rng, high=rng, size=scales.shape)
        return np.abs(scales + perturbations)

    def evolutionary_algorithm(self, scales, num_cycles):

        pop_size = 15
        population = []

        # make best layer-wise scales into initial population
        obj = self.loss(scales)
        for p in range(0, pop_size):
            population.append((obj, scales))
        num_samples = 10
        best_prev = obj
        for i in range(0, num_cycles):
            # get sampling of population
            samples = random.choices(population, k=num_samples)

            # get sample with smallest loss
            heapq.heapify(samples)
            parent = samples[0]

            mutated_scales = self.mutate(parent[1])

            tic = time.perf_counter()
            obj = self.loss(mutated_scales)
            toc = time.perf_counter()
            population.append( (obj, mutated_scales) )
            population = sorted(population, key=lambda t: t[0])
            population.pop()

        heapq.heapify(population)
        return heapq.heappop(population)

    def opt(self):
        
        attn_modules = []
        for m in self.inf_model.modules():
            if type(m) in [Attention, AttentionSubsample, Attention_ViT]:
                attn_modules.append(m)
                m.inputs = []
                m.outputs = []
                m.handle.remove()

            if type(m) in [WindowAttention]:
                attn_modules.append(m)
        
        self.o_fps = []
        self.inf_model.model_dequant()
        for i, (x, target) in enumerate(self.calib_loader):
            x = x.cuda()
            with torch.no_grad():
                o_fp = self.inf_model(x)

            self.o_fps.append(o_fp.cpu())
        self.inf_model.model_quant()

        tic = time.perf_counter()
        print("beginning Evol-Q..")

        attn_modules.reverse()
        for i in range(0, self.args.num_passes):
            obj = ""
            j = 0
            for m in attn_modules:
                self.m = m
                j += 1
                init = self.get_scales()

                m_name = m.__class__.__name__
                print("module: ", m_name, str(j))

                if type(m) in [Attention_ViT, WindowAttention]:
                    self.bits = m.proj.quantizer.bit_type.bits
                elif type(m) in [Attention, AttentionSubsample]:
                    self.bits = m.proj[1].quantizer.bit_type.bits

                final_obj, final_scales = self.evolutionary_algorithm(init, self.args.num_cycles)
                self.set_scales(final_scales)

                self.inf_model.model_quant()
                loss, top1, top5 = validate(self.args, self.val_loader, self.inf_model, self.criterion, self.device)
                if i == 0 or top1 > current_top1:
                    current_top1 = top1
                    with open(self.args.save_folder+"/" + self.args.mode +".txt", "w") as f:
                            f.write("Current Top1: " + str(top1) + "\n")
                    torch.save(self.inf_model, self.args.save_folder+ "/evolq.pt")

        toc = time.perf_counter()
        print(f"==== FULL EVOL Q Completed in {toc - tic:0.4f} seconds", "== ", str(i))
        self.inf_model =  torch.load(self.args.save_folder+"/evolq.pt")
        return self.inf_model
