#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python train.py --batch_size=2 --datapath ./BRATS2018_Training_none_npy --savepath ./output_2018 --num_epochs 1000 --dataname BRATS2018
#eval:
#resume=output_2018/model_last.pth
# python train.py --batch_size=1 --datapath ./BRATS2018_Training_none_npy --savepath ./output_2018 --num_epochs 0 --dataname BRATS2018 --resume1 ./output_2018/model1_last.pth --resume2 ./output_2018/model2_last.pth