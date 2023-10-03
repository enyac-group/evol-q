## Jumping Through Local Minima: Quantization in the Jagged Loss Landscape of Vision Transformers (Evol-Q) <br><sub>Official PyTorch implementation of the ICCV 2023 paper</sub>



**Jumping Through Local Minima: Quantization in the Jagged Loss Landscape of Vision Transformers**<br>
Natalia Frumkin, Dibakar Gope, and Diana Marculescu
<br>https://arxiv.org/pdf/2308.10814<br>

Abstract: *Quantization scale and bit-width are the most important parameters when considering how to quantize a neural network. Prior work focuses on optimizing quantization scales in a global manner through gradient methods (gradient descent \& Hessian analysis). Yet, when applying perturbations to quantization scales, we observe a very jagged, highly non-smooth test loss landscape. In fact, small perturbations in quantization scale can greatly affect accuracy, yielding a 0.5−0.8% accuracy boost in 4-bit quantized vision transformers (ViTs). In this regime, gradient methods break down, since they cannot reliably reach local minima. In our work, dubbed Evol-Q, we use evolutionary search to effectively traverse the non-smooth landscape. Additionally, we propose using an infoNCE loss, which not only helps combat overfitting on the small calibration dataset (1,000 images) but also makes traversing such a highly non-smooth surface easier. Evol-Q improves the top-1 accuracy of a fully quantized ViT-Base by 10.30%, 0.78%, and 0.15% for 3-bit, 4-bit, and 8-bit weight quantization levels. Extensive experiments on a variety of CNN and ViT architectures further demonstrate its robustness in extreme quantization scenarios.*

## Requirements
* One high-end GPU for inference such as an RTX A5000, V100, or A100.
* Python 3.8 and PyTorch 1.9.1 (or later). We have provided [environment.yml](./environment.yml) which has our library dependencies. To create a conda environment to reproduce our setup, see below:
    - `conda env create -f environment.yml -n evol-q`
    - `conda activate evol-q`

## Getting Started

To run an initial experiment with 8-bit quantized DeiT-Tiny:

```sh scripts/run.sh```

> This script contains details for how to quantize 3,4,8-bit models as well as model architectures.

Our repository is adapted from [FQ-ViT](https://github.com/megvii-research/FQ-ViT) and [Meta's LeVit implementation](https://github.com/facebookresearch/LeViT). We have created our novel block-wise evolutionary algorithm in [joint_evol_opt.py](./joint_evol_opt.py) which can quantize the following model flavors out of the box:

* ViT-Base
* DeiT-Tiny, DeiT-Small, DeiT-Base
* LeViT-128S, LeViT-128, LeViT-192, LeViT-246, LeViT-384

Below, we show the accuracy for each 4-bit quantized model, averaged across seeds {0 ,1, 2}:

| Model    | Top-1 Accuracy |
| -------- | ------- |
| ViT-Base |  |
| DeiT-Tiny|  |
| DeiT-Small|  |
| DeiT-Base|  |
| LeViT-128S|  |
| LeViT-128|  |
| LeViT-192|  |
| LeViT-246|  |
| LeViT-384|  |
