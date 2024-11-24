from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import collections
import torch
import numpy as np
import json
from collections import OrderedDict
from tqdm import tqdm
from os.path import dirname, abspath
from utils.writer import print_metrics_frames

pdvc_dir = dirname(abspath(__file__))
sys.path.insert(0, pdvc_dir)
# sys.path.insert(0, os.path.join(pdvc_dir, 'densevid_eval3'))
# sys.path.insert(0, os.path.join(pdvc_dir, 'densevid_eval3/SODA'))


# from densevid_eval3.eval_soda import eval_soda
# from densevid_eval3.eval_para import eval_para
# from densevid_eval3.eval_dvc import eval_dvc


#=====================================================================================================


def evaluate(model, criterion, postprocessors, loader, loss_manager, frame_metrics,
            logger=None, score_threshold=0, alpha=0.3, device='cuda'):

    opt = loader.dataset.opt
    with torch.set_grad_enabled(False):
        for dt in tqdm(loader):
            dt = {key: _.to(device) if isinstance(_, torch.Tensor) else _ for key, _ in dt.items()}
            dt['video_target'] = [
                    {key: _.to(device) if isinstance(_, torch.Tensor) else _ for key, _ in vid_info.items()} for vid_info in
                    dt['video_target']
                    ]
            dt = collections.defaultdict(lambda: None, dt)
            frame_labels = [sample['frame_labels'] for sample in dt['video_target']]
            orig_target_sizes = dt['video_length'][:, 1]
            batch_size = len(orig_target_sizes)

            # Model inference
            output, loss = model(dt, criterion, eval_mode=True)
            for k in loss:
                loss_manager[k].update(loss[k].item())

            query_classes, query_widths = postprocessors['bbox'](output, orig_target_sizes)
            for b_i in range(batch_size):
                query_class = query_classes[b_i]
                query_width = query_widths[b_i]
                seq_len = orig_target_sizes[b_i]
                frame_label = frame_labels[b_i]

                pred_frame = torch.zeros([len(frame_label)], dtype=torch.int64).to(opt.device)
                stg_end_idx = torch.cumsum(query_width, dim=-1)
                stg_end_idx[-1] = seq_len
                stg_beg_idx = torch.cat([torch.tensor([0]).to(opt.device), stg_end_idx])[:-1]
                for q_i, (beg, end) in enumerate(zip(stg_beg_idx, stg_end_idx)):
                    pred_frame[beg:end] = query_class[q_i]

                for metric in frame_metrics:
                    frame_metrics[metric].update(pred_frame, frame_label, is_prob=False)


        print_str = '\nVALIDATION:'
        logger.info(print_str)

        print_metrics_frames(
            metric_dict={
                'Refine': frame_metrics, 
            }, 
            n_classes = opt.num_classes,
            logger = logger
        )

        loss_str = [f'{loss}: {loss_manager[loss].avg:.4f}' for loss in loss_manager if criterion.weight_dict[loss] > 0]
        loss_str = ', '.join(loss_str)
        logger.info(loss_str)
        logger.info(f'\n')

    return 0