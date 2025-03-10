#coding=utf-8
import argparse
import os
import time
import logging
import random
import numpy as np
from collections import OrderedDict

import torch
import torch.optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import smml
from data.transforms import *
from data.datasets_nii import Brats_loadall_nii, Brats_loadall_test_nii
from data.data_utils import init_fn
from utils import Parser,criterions
from utils.parser import setup 
from utils.lr_scheduler import LR_Scheduler, record_loss, MultiEpochsDataLoader 
from predict import AverageMeter, test_softmax
from utils.criterions import kl_diverse_loss, prototypical_loss
from utils.criterions import feat_loss, kl_loss, MAD

parser = argparse.ArgumentParser()

parser.add_argument('-batch_size', '--batch_size', default=1, type=int, help='Batch size')
parser.add_argument('--datapath', default=None, type=str)
parser.add_argument('--dataname', default='BRATS2018', type=str)
parser.add_argument('--savepath', default=None, type=str)
parser.add_argument('--resume1', default=None, type=str)
parser.add_argument('--resume2', default=None, type=str)
parser.add_argument('--pretrain', default=None, type=str)
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--iter_per_epoch', default=150, type=int)
parser.add_argument('--region_fusion_start_epoch', default=0, type=int)
parser.add_argument('--seed', default=1024, type=int)
path = os.path.dirname(__file__)

## parse arguments
args = parser.parse_args()
setup(args, 'training')
args.train_transforms = 'Compose([RandCrop3D((128,128,128)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'

ckpts = args.savepath
os.makedirs(ckpts, exist_ok=True)

###tensorboard writer
writer = SummaryWriter(os.path.join(args.savepath, 'summary'))

###modality missing mask
masks = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]
masks_torch = torch.from_numpy(np.array(masks))
mask_name = ['t2', 't1c', 't1', 'flair', 
            't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
            'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
            'flairt1cet1t2']
print (masks_torch.int())

def main():
    ##########setting seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    ##########setting models
    if args.dataname in ['BRATS2021', 'BRATS2020', 'BRATS2018']:
        num_cls = 4
    elif args.dataname == 'BRATS2015':
        num_cls = 5
    else:
        print ('dataset is error')
        exit(0)
    model1 = smml.Model(num_cls=num_cls)
    model2 = smml.Model(num_cls=num_cls)
    # print (model)
    model1 = torch.nn.DataParallel(model1).cuda()
    model2 = torch.nn.DataParallel(model2).cuda()
    mad_loss_function = MAD()
    ##########Setting learning schedule and optimizer
    lr_schedule = LR_Scheduler(args.lr, args.num_epochs)
    train_params1 = [{'params': model1.parameters(), 'lr': args.lr, 'weight_decay':args.weight_decay}]
    optimizer1 = torch.optim.Adam(train_params1,  betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
    train_params2 = [{'params': model2.parameters(), 'lr': args.lr, 'weight_decay':args.weight_decay}]
    optimizer2 = torch.optim.Adam(train_params2,  betas=(0.9, 0.999), eps=1e-08, amsgrad=True)

    ##########Setting data
    if args.dataname in ['BRATS2020', 'BRATS2015']:
        train_file = 'train.txt'
        test_file = 'test.txt'
    elif args.dataname == 'BRATS2018':
        ####BRATS2018 contains three splits (1,2,3)
        train_file = 'train1.txt'
        test_file = 'test1.txt'

    logging.info(str(args))
    train_set = Brats_loadall_nii(transforms=args.train_transforms, root=args.datapath, num_cls=num_cls, train_file=train_file)
    test_set = Brats_loadall_test_nii(transforms=args.test_transforms, root=args.datapath, test_file=test_file)
    train_loader = MultiEpochsDataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=init_fn)
    test_loader = MultiEpochsDataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    ##########Evaluate
    if args.resume1 is not None:
        checkpoint1 = torch.load(args.resume1)
        logging.info('best epoch: {}'.format(checkpoint1['epoch']))
        model1.load_state_dict(checkpoint1['state_dict'])
        checkpoint2 = torch.load(args.resume2)
        model2.load_state_dict(checkpoint2['state_dict'])
        test_score = AverageMeter()
        with torch.no_grad():
            logging.info('###########test set wi post process###########')
            for i, mask in enumerate(masks[::-1]):
                logging.info('{}'.format(mask_name[::-1][i]))
                dice_score = test_softmax(
                    test_loader,
                    model1,
                    model2,
                    save_path=args.savepath,
                    dataname = args.dataname,
                    feature_mask = mask,
                    mask_name = mask_name[::-1][i])
                test_score.update(dice_score)
            logging.info('Avg scores: {}'.format(test_score.avg))
            exit(0)

    ##########Training
    start = time.time()
    torch.set_grad_enabled(True)
    logging.info('#############training############')
    # iter_per_epoch = args.iter_per_epoch
    
    iter_per_epoch = len(train_loader)
    train_iter = iter(train_loader)
    for epoch in range(args.num_epochs):
        step_lr1 = lr_schedule(optimizer1, epoch)
        step_lr2 = lr_schedule(optimizer2, epoch)
        
        writer.add_scalar('lr', step_lr1, global_step=(epoch+1))
        writer.add_scalar('lr', step_lr2, global_step=(epoch+1))
        b = time.time()
        for i in range(iter_per_epoch):
            step = (i+1) + epoch*iter_per_epoch
            ###Data load
            try:
                data = next(train_iter)
            except:
                train_iter = iter(train_loader)
                data = next(train_iter)
            
            x, target, sam_target, mask1, mask2 = data[:5]
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            mask1 = mask1.cuda(non_blocking=True)
            mask2 = mask2.cuda(non_blocking=True)
            sam_target = sam_target.cuda(non_blocking=True)

            model1.module.is_training = True
            model2.module.is_training = True
            
            fuse_pred1, sep_preds1, prm_preds1, feature1, sam_pred1 = model1(x, mask1, sam_target, target)  # feature:[2, 512, 8, 8, 8] 
            # torch.cuda.empty_cache()
            fuse_pred2, sep_preds2, prm_preds2, feature2, sam_pred2 = model2(x, mask2, sam_target, target)
            # TODO 
            soft = torch.nn.Softmax(dim=1)
            # preds_level loss
            kl_loss_2to1, kl_loss_1to2  = kl_diverse_loss(fuse_pred1, fuse_pred2, target, num_cls=num_cls)  # b k h w z
            # feature_level loss
            mad_loss = mad_loss_function(feature1, feature2, fuse_pred1, fuse_pred2, target)
            
            sam_consistent_loss1 = kl_loss(fuse_pred1, sam_pred1)
            sam_consistent_loss2 = kl_loss(fuse_pred2, sam_pred2)
            
            sam_cross_loss1 = criterions.softmax_weighted_loss(sam_pred1, target, num_cls=num_cls)
            sam_dice_loss1 = criterions.dice_loss(sam_pred1, target, num_cls=num_cls)
            sam_fuse_loss1 = sam_cross_loss1 + sam_dice_loss1
            sam_cross_loss2 = criterions.softmax_weighted_loss(sam_pred2, target, num_cls=num_cls)
            sam_dice_loss2 = criterions.dice_loss(sam_pred2, target, num_cls=num_cls)
            sam_fuse_loss2 = sam_cross_loss2 + sam_dice_loss2
            
            ###Loss compute      
            fuse_cross_loss1 = criterions.softmax_weighted_loss(soft(fuse_pred1), target, num_cls=num_cls)
            fuse_dice_loss1 = criterions.dice_loss(soft(fuse_pred1), target, num_cls=num_cls)
            fuse_loss1 = fuse_cross_loss1 + fuse_dice_loss1
            fuse_cross_loss2 = criterions.softmax_weighted_loss(soft(fuse_pred2), target, num_cls=num_cls)
            fuse_dice_loss2 = criterions.dice_loss(soft(fuse_pred2), target, num_cls=num_cls)
            fuse_loss2 = fuse_cross_loss2 + fuse_dice_loss2

            sep_cross_loss1 = torch.zeros(1).cuda().float()
            sep_dice_loss1 = torch.zeros(1).cuda().float()
            for sep_pred in sep_preds1:
                sep_cross_loss1 += criterions.softmax_weighted_loss(sep_pred, target, num_cls=num_cls)
                sep_dice_loss1 += criterions.dice_loss(sep_pred, target, num_cls=num_cls)
            sep_loss1 = sep_cross_loss1 + sep_dice_loss1
            sep_cross_loss2 = torch.zeros(1).cuda().float()
            sep_dice_loss2 = torch.zeros(1).cuda().float()
            for sep_pred in sep_preds2:
                sep_cross_loss2 += criterions.softmax_weighted_loss(sep_pred, target, num_cls=num_cls)
                sep_dice_loss2 += criterions.dice_loss(sep_pred, target, num_cls=num_cls)
            sep_loss2 = sep_cross_loss2 + sep_dice_loss2

            prm_cross_loss1 = torch.zeros(1).cuda().float()
            prm_dice_loss1 = torch.zeros(1).cuda().float()
            for prm_pred in prm_preds1:
                prm_cross_loss1 += criterions.softmax_weighted_loss(prm_pred, target, num_cls=num_cls)
                prm_dice_loss1 += criterions.dice_loss(prm_pred, target, num_cls=num_cls)
            prm_loss1 = prm_cross_loss1 + prm_dice_loss1
            prm_cross_loss2 = torch.zeros(1).cuda().float()
            prm_dice_loss2 = torch.zeros(1).cuda().float()
            for prm_pred in prm_preds2:
                prm_cross_loss2 += criterions.softmax_weighted_loss(prm_pred, target, num_cls=num_cls)
                prm_dice_loss2 += criterions.dice_loss(prm_pred, target, num_cls=num_cls)
            prm_loss2 = prm_cross_loss2 + prm_dice_loss2
                    
            if epoch < args.region_fusion_start_epoch:
                loss1 = fuse_loss1 * 0.0+ sep_loss1 + prm_loss1
                loss2 = fuse_loss1 * 0.0+ sep_loss2 + prm_loss2
            else:  
                if torch.isnan(mad_loss):
                    print("Loss is nan")
                    mad_loss = torch.zeros(1).float().cuda()
                loss1 = fuse_loss1 + sep_loss1 + prm_loss1 + sam_fuse_loss1 + mad_loss
                loss2 = fuse_loss2 + sep_loss2 + prm_loss2 + sam_fuse_loss2 + mad_loss
                
            if epoch > 100:
                loss1 = loss1 + kl_loss_2to1 + sam_consistent_loss1 
                loss2 = loss2 + kl_loss_1to2 + sam_consistent_loss2 
            # torch.cuda.empty_cache()
            optimizer1.zero_grad()
            loss1.backward(retain_graph=True)
            optimizer1.step()
            
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()
            ###log
            writer.add_scalar('loss1', loss1.item(), global_step=step)
            writer.add_scalar('loss2', loss2.item(), global_step=step)
            writer.add_scalar('mad_loss', (mad_loss).item(), global_step=step)
            # branch 1
            # writer.add_scalar('kl_loss_2to1', kl_loss_2to1.item(), global_step=step)
            writer.add_scalar('fuse_cross_loss1', fuse_cross_loss1.item(), global_step=step)
            writer.add_scalar('fuse_dice_loss1', fuse_dice_loss1.item(), global_step=step)
            writer.add_scalar('sep_cross_loss1', sep_cross_loss1.item(), global_step=step)
            writer.add_scalar('sep_dice_loss1', sep_dice_loss1.item(), global_step=step)
            writer.add_scalar('prm_cross_loss1', prm_cross_loss1.item(), global_step=step)
            writer.add_scalar('prm_dice_loss1', prm_dice_loss1.item(), global_step=step)
            writer.add_scalar('sam_consistent_loss1', sam_consistent_loss1.item(), global_step=step)
            writer.add_scalar('sam_cross_loss1', sam_cross_loss1.item(), global_step=step)
            writer.add_scalar('sam_dice_loss1', sam_dice_loss1.item(), global_step=step)
            # branch 2
            # writer.add_scalar('kl_loss_1to2', kl_loss_1to2.item(), global_step=step)
            writer.add_scalar('fuse_cross_loss2', fuse_cross_loss2.item(), global_step=step)
            writer.add_scalar('fuse_dice_loss2', fuse_dice_loss2.item(), global_step=step)
            writer.add_scalar('sep_cross_loss2', sep_cross_loss2.item(), global_step=step)
            writer.add_scalar('sep_dice_loss2', sep_dice_loss2.item(), global_step=step)
            writer.add_scalar('prm_cross_loss2', prm_cross_loss2.item(), global_step=step)
            writer.add_scalar('prm_dice_loss2', prm_dice_loss2.item(), global_step=step)
            writer.add_scalar('sam_consistent_loss2', sam_consistent_loss2.item(), global_step=step)
            writer.add_scalar('sam_cross_loss2', sam_cross_loss2.item(), global_step=step)
            writer.add_scalar('sam_dice_loss2', sam_dice_loss2.item(), global_step=step)

            msg = 'Epoch {}/{}, Iter {}/{}, Loss1 {:.4f}, '.format((epoch+1), args.num_epochs, (i+1), iter_per_epoch, loss1.item())
            msg += 'Loss2 {:.4f}, '.format(loss2.item())
            msg += 'mad_loss:{:.4f},'.format((mad_loss).item())
            
            msg += 'kl_fuse_loss2to1:{:.4f},'.format(kl_loss_2to1.item())
            msg += 'fusecross1:{:.4f}, fusedice1:{:.4f},'.format(fuse_cross_loss1.item(), fuse_dice_loss1.item())
            msg += 'sepcross1:{:.4f}, sepdice1:{:.4f},'.format(sep_cross_loss1.item(), sep_dice_loss1.item())
            msg += 'prmcross1:{:.4f}, prmdice1:{:.4f},'.format(prm_cross_loss1.item(), prm_dice_loss1.item())
            msg += 'sam_consistent_loss1:{:.4f},'.format(sam_consistent_loss1.item())
            msg += 'samcross1:{:.4f}, samdice1:{:.4f},'.format(sam_cross_loss1.item(), sam_dice_loss1.item())
            
            msg += 'kl_fuse_loss1to2:{:.4f},'.format(kl_loss_1to2.item())
            msg += 'fusecross2:{:.4f}, fusedice2:{:.4f},'.format(fuse_cross_loss2.item(), fuse_dice_loss2.item())
            msg += 'sepcross2:{:.4f}, sepdice2:{:.4f},'.format(sep_cross_loss2.item(), sep_dice_loss2.item())
            msg += 'prmcross2:{:.4f}, prmdice2:{:.4f},'.format(prm_cross_loss2.item(), prm_dice_loss2.item())
            msg += 'sam_consistent_loss2:{:.4f},'.format(sam_consistent_loss2.item())
            msg += 'samcross2:{:.4f}, samdice2:{:.4f},'.format(sam_cross_loss2.item(), sam_dice_loss2.item())
            
            logging.info(msg)
            del loss1, loss2
            # torch.cuda.empty_cache()
        logging.info('train time per epoch: {}'.format(time.time() - b))

        ##########model save
        file_name1 = os.path.join(ckpts, 'model1_last.pth')
        torch.save({
            'epoch': epoch,
            'state_dict': model1.state_dict(),
            'optim_dict': optimizer1.state_dict(),
            },
            file_name1)
        file_name2 = os.path.join(ckpts, 'model2_last.pth')
        torch.save({
            'epoch': epoch,
            'state_dict': model2.state_dict(),
            'optim_dict': optimizer2.state_dict(),
            },
            file_name2)
        

    msg = 'total time: {:.4f} hours'.format((time.time() - start)/3600)
    logging.info(msg)

    ##########Evaluate the last epoch model
    test_score = AverageMeter()
    with torch.no_grad():
        logging.info('###########test set wi/wo postprocess###########')
        for i, mask in enumerate(masks):
            logging.info('{}'.format(mask_name[i]))
            dice_score = test_softmax(
                            test_loader,
                            model1,
                            model2,
                            dataname = args.dataname,
                            feature_mask = mask)
            test_score.update(dice_score)
        logging.info('Avg scores: {}'.format(test_score.avg))

if __name__ == '__main__':
    main()
