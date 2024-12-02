# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F

from misc.detr_utils.box_ops import box_cl_to_xy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1,
                 cost_dice: float = 1,
                 cost_mask: float = 1,
                 cost_alpha = 0.25,
                 cost_gamma = 2):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_dice = cost_dice
        self.cost_mask = cost_mask
        self.cost_alpha = cost_alpha
        self.cost_gamma = cost_gamma

        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0 or cost_caption!=0, "all costs cant be 0"

    def forward(self, outputs, targets, verbose=False, many_to_one=False):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 2] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 2] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            # out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_prob = F.softmax(outputs["pred_logits"].flatten(0, 1), dim=1)
            out_bbox = outputs["pred_boxes"].flatten(0, 1)
            out_mask = torch.sigmoid(outputs['pred_masks'].permute(0, 2, 1)).flatten(0, 1)
            _, mask_len = out_mask.shape

            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

            tgt_lens = [len(t["labels"]) for t in targets]
            frame_gts = []
            for t in targets:
                t_gts = t['frame_labels']
                t_len = t['frame_labels'].shape[0]
                pad_labels = torch.tensor([0]*(mask_len - t_len)).to(t['frame_labels'].device)
                t_gts = torch.cat([t['frame_labels'], pad_labels])
                frame_gts.append(t_gts)
            expand_frame_gts = []
            for ft, l in zip(frame_gts, tgt_lens):
                expand_frame_gts += [ft] * l
            expand_frame_gts = torch.stack(expand_frame_gts)
            tgt_mask = (expand_frame_gts == tgt_ids.unsqueeze(-1)).to(torch.float32)



            # Compute the classification cost.
            # alpha = 0.25
            alpha = self.cost_alpha
            gamma = self.cost_gamma
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cl_to_xy(out_bbox), box_cl_to_xy(tgt_bbox))

            # compute the cross entropy loss between each mask pairs
            cost_mask = pair_wise_sigmoid_cross_entropy_loss(out_mask, tgt_mask)

            # Compute the dice loss betwen each mask pairs
            cost_dice = pair_wise_dice_loss(out_mask, tgt_mask)

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou 
                + self.cost_dice * cost_dice + self.cost_mask * cost_mask

            costs = {'cost_bbox': cost_bbox,
                     'cost_class': cost_class,
                     'cost_giou': cost_giou,
                     'out_bbox': out_bbox[:, 0::2]}

            if verbose:
                print('\n')
                print(self.cost_bbox, cost_bbox.var(dim=0), cost_bbox.max(dim=0)[0] - cost_bbox.min(dim=0)[0])
                print(self.cost_class, cost_class.var(dim=0), cost_class.max(dim=0)[0] - cost_class.min(dim=0)[0])
                print(self.cost_giou, cost_giou.var(dim=0), cost_giou.max(dim=0)[0] - cost_giou.min(dim=0)[0])
                # print(self.cost_caption, cost_caption.var(dim=0), cost_caption.max(dim=0)[0] - cost_caption.min(dim=0)[0])

            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["boxes"]) for v in targets]
            # pdb.set_trace()
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            m2o_rate = 4
            rl_indices = [linear_sum_assignment(torch.cat([c[i]]*m2o_rate, -1)) for i, c in enumerate(C.split(sizes, -1))]
            rl_indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j%sizes[ii], dtype=torch.int64)) for ii,(i, j) in
                       enumerate(rl_indices)]

            indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

            if verbose:
                print('------matching results:')
                print(indices)
                for indice in indices:
                    for i, j in zip(*indice):
                        print(out_bbox[i][0::2], tgt_bbox[j][0::2])
                print('-----topK scores:')
                topk_indices = out_prob.topk(10, dim=0)
                print(topk_indices)
                for i,(v,ids) in enumerate(zip(*topk_indices)):
                    print('top {}'.format(i))
                    s= ''
                    for name,cost in costs.items():
                        s += name + ':{} '.format(cost[ids])
                    print(s)

            return indices, rl_indices


def pair_wise_dice_loss(inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    A pair wise version of the dice loss, see `dice_loss` for usage.

    Args:
        inputs (`torch.Tensor`):
            A tensor representing a mask
        labels (`torch.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).

    Returns:
        `torch.Tensor`: The computed loss between each pairs.
    """
    inputs = inputs.sigmoid().flatten(1)
    numerator = 2 * torch.matmul(inputs, labels.T)
    # using broadcasting to get a [num_queries, NUM_CLASSES] matrix
    denominator = inputs.sum(-1)[:, None] + labels.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def pair_wise_sigmoid_cross_entropy_loss(inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    r"""
    A pair wise version of the cross entropy loss, see `sigmoid_cross_entropy_loss` for usage.

    Args:
        inputs (`torch.Tensor`):
            A tensor representing a mask.
        labels (`torch.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).

    Returns:
        loss (`torch.Tensor`): The computed loss between each pairs.
    """

    height_and_width = inputs.shape[1]

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    cross_entropy_loss_pos = criterion(inputs, torch.ones_like(inputs))
    cross_entropy_loss_neg = criterion(inputs, torch.zeros_like(inputs))

    loss_pos = torch.matmul(cross_entropy_loss_pos / height_and_width, labels.T)
    loss_neg = torch.matmul(cross_entropy_loss_neg / height_and_width, (1 - labels).T)
    loss = loss_pos + loss_neg
    return loss

def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class,
                            cost_bbox=args.set_cost_bbox,
                            cost_giou=args.set_cost_giou,
                            cost_dice=args.set_cost_dice,
                            cost_mask=args.set_cost_mask,
                            cost_alpha = args.cost_alpha,
                            cost_gamma = args.cost_gamma
                            )