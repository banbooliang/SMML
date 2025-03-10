#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python train.py --batch_size=2 --datapath ./BRATS2015_Training_none_npy --savepath ./output_2015 --num_epochs 1000 --dataname BRATS2015

#eval:
#resume=output/model_last.pth
# python train.py --batch_size=1 --datapath ./BRATS2015_Training_none_npy --savepath ./output_2015 --num_epochs 0 --dataname BRATS2015 --resume1 ./output_2015/model1_last.pth --resume2 ./output_2015/model2_last.pth