#!/bin/sh
devices=$1
method=$2
dim=$3
scale=$4
for DATASET in  cifar caltech101 dtd oxford_flowers102 oxford_iiit_pet svhn sun397 patch_camelyon eurosat resisc45 diabetic_retinopathy clevr_count clevr_dist dmlab kitti dsprites_loc dsprites_ori smallnorb_azi smallnorb_ele
    do
        CUDA_VISIBLE_DEVICES=$devices  python train.py --dataset $DATASET --method $method --dim $dim --scale $scale
    done