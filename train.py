from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import argparse
import json
import os
import sys
import os.path as osp
from os.path import dirname, abspath
import random

import numpy as np
import pandas as pd
import copy
import collections 
from collections import OrderedDict
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader


pdvc_dir = dirname(abspath(__file__))
sys.path.insert(0, pdvc_dir)

from embryo_former.model import build # Import model
from metrics.meter import AverageMeter

from eval_utils import evaluate
from data.dataset import PropSeqDataset, collate_fn
from misc.detr_utils import box_ops
from utils.writer import print_metrics_frames
from utils.logger import create_logger
from metrics.evaluator import (
    create_dict_metrics
)
from utils.logger import create_logger
from utils.scheduler import WarmupCosineScheduler

# -------- GLOBAL VARIABLES --------
DEBUG = False

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def init_args(opt, config_path):
    with open(config_path, 'r') as f:
        load_opt = json.load(f)
        for k, v in load_opt.items():
            vars(opt).update({k:v})
    return opt


def main(opt, config_path: str):
    init_args(opt, config_path)
    set_seed(opt.seed)
        
    SAVE_DIR = opt.save_dir
    TRAIN_SAVE_DIR = f'{SAVE_DIR}/vis_train'
    VAL_SAVE_DIR = f'{SAVE_DIR}/vis_val'
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(TRAIN_SAVE_DIR, exist_ok=True)
    os.makedirs(VAL_SAVE_DIR, exist_ok=True)
    logger = create_logger(osp.join(SAVE_DIR, 'log.txt'))

    save_cfg_path = osp.join(SAVE_DIR, 'config.json')
    with open(save_cfg_path, 'w') as f:
        json.dump(vars(opt), f, indent=2)

    # Create model
    model, criterion, postprocessors = build(opt)
    model = model.to(opt.device)

    # Create dataset
    train_dataset = PropSeqDataset(opt.train_annot, True, opt)
    val_dataset = PropSeqDataset(opt.val_annot, False, opt)

    logger.info(f'Create dataset successfully')
    train_loader = DataLoader(
        train_dataset, batch_size=opt.batch_size,
        shuffle=True, num_workers=8, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1,
        shuffle=False, num_workers=8, collate_fn=collate_fn
    )
    if opt.optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    elif opt.optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    if opt.scheduler == 'CosineAnnealing':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epoch, eta_min=1e-7, verbose=True)
    elif opt.scheduler == 'WarmupCosine':
        lr_scheduler = WarmupCosineScheduler(optimizer, warmup_steps=opt.learning_rate_decay_start, t_total=opt.epoch, verbose=True)
    else:
        milestone = [
            opt.learning_rate_decay_start + opt.learning_rate_decay_every * _ for _ in range(int((opt.epoch - opt.learning_rate_decay_start) / opt.learning_rate_decay_every))]
        logger.info(f'LR scheduler milestone: {milestone}')
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestone, gamma=opt.learning_rate_decay_rate, verbose=True)
    
    weight_dict = criterion.weight_dict
    n_epochs = opt.epoch
    
    refine_frame_metrics = create_dict_metrics(opt.num_classes)
    center_metrics = create_dict_metrics(opt.num_classes)
    iou_metrics = create_dict_metrics(opt.num_classes)
    width_metrics = create_dict_metrics(opt.num_classes)

    center_scores = {
        'accuracy': AverageMeter(), 'precision': AverageMeter(), 'recall': AverageMeter()
    }
    class_accuracy = {}
    segment_metrics = {'iou': {}, 'giou': {}, 'prec': {}, 'rec': {}}
    for c in range(opt.num_classes):
        segment_metrics['iou'][c] = AverageMeter()
        segment_metrics['giou'][c] = AverageMeter()
        segment_metrics['prec'][c] = AverageMeter()
        segment_metrics['rec'][c] = AverageMeter()
        class_accuracy[c] = AverageMeter()

    loss_manager = {}
    for loss in criterion.weight_dict:
        loss_manager[loss] = AverageMeter()

    best_refine_acc, best_refine_f1 = -1, -1
    logger.info(f'--------------- START TRAINING ---------------')
    for e in range(opt.epoch):    
        for loss in loss_manager:
            loss_manager[loss].reset()

        # -------------------- TRAINING --------------------
        model.train()
        criterion.training=True
        for batch_id, dt in tqdm(enumerate(train_loader)):
            dt = {key: _.to(opt.device) if isinstance(_, torch.Tensor) else _ for key, _ in dt.items()}
            dt['video_target'] = [
                {key: _.to(opt.device) if isinstance(_, torch.Tensor) else _ for key, _ in vid_info.items()} for vid_info in
                dt['video_target']
            ]
            dt = collections.defaultdict(lambda: None, dt)
            orig_target_sizes = dt['video_length'][:, 1]
            
            optimizer.zero_grad()
            output, loss = model(dt, criterion)

            for k in loss:
                loss_manager[k].update(loss[k].item())
            final_loss = sum(loss[k] * weight_dict[k] for k in loss.keys() if k in weight_dict)
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()

        print_str = (f'-------------------- Epoch {e}/{n_epochs} ----------------------\nTRAIN')
        logger.info(print_str)
        loss_str = [f'{loss}: {loss_manager[loss].avg:.4f}' for loss in loss_manager if criterion.weight_dict[loss] > 0]
        loss_str = ', '.join(loss_str)
        logger.info(loss_str)



        # -------------------- VALIDATION --------------------
        model.eval()
        criterion.training=False

        for loss in loss_manager:
            loss_manager[loss].reset()

        for metric in refine_frame_metrics:
            refine_frame_metrics[metric].reset() 

        evaluate(model, criterion, postprocessors, val_loader, 
                loss_manager, refine_frame_metrics,
                logger=logger, device=opt.device)

        lr_scheduler.step()

        # criterion.training=False
        # for metric in refine_frame_metrics:
        #     refine_frame_metrics[metric].reset() 

        # for loss in loss_manager:
        #     loss_manager[loss].reset()
        
        # for batch_id, dt in enumerate(val_loader):
        #     dt = {key: _.to(opt.device) if isinstance(_, torch.Tensor) else _ for key, _ in dt.items()}
        #     dt['video_target'] = [
        #         {key: _.to(opt.device) if isinstance(_, torch.Tensor) else _ for key, _ in vid_info.items()} for vid_info in
        #         dt['video_target']]
        #     dt = collections.defaultdict(lambda: None, dt)
        #     orig_target_sizes = dt['video_length'][:, 1]

        #     # Model inference
        #     output, loss = model(dt, criterion, eval_mode=True)
        #     for k in loss:
        #         loss_manager[k].update(loss[k].item())

        # print_str = '\nVALIDATION:'
        # logger.info(print_str)
      

        # loss_str = [f'{loss}: {loss_manager[loss].avg:.4f}' for loss in loss_manager if criterion.weight_dict[loss] > 0]
        # loss_str = ', '.join(loss_str)
        # logger.info(loss_str)
        # logger.info(f'\n')        
        # lr_scheduler.step() 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg_path', type=str, default=None)
    parser.add_argument('--gpu_id', type=str, nargs='+', default=['0'])
    
    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in opt.gpu_id])
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
    main(opt, opt.cfg_path)

