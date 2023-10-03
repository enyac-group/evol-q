#!/bin/bash

device=0            # CUDA device
seed=0              # random seed
model="deit_tiny"   # model flavor
mode="e2e"          # mode from main.py

# specify # of bits for weights & activations ex: 3,4,8
weight=4
w=uint${weight}
a=uint8

# output folder for checkpoint saving & script outputs
output_folder="output/${model}_${weight}W8A_s${seed}"
out_file=$output_folder/logs.txt
mkdir -p $output_folder

date
CUDA_VISIBLE_DEVICES=$device python3 main.py \
    $model ~/ImageNet --ptf \
    --mode ${mode} \
    --seed ${seed} \
    --w_bit_type ${w} \
    --a_bit_type ${a} \
    --quant-method omse \
    --bias-corr \
    --save_folder $output_folder \
    2>&1 | tee -a $out_file # append to output folder

date
