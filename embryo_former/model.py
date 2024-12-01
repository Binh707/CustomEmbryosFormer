                # ------------------------------------------------------------------------
# EmbryosFormer
# ------------------------------------------------------------------------
# Modified from PDVC(https://github.com/ttengwang/PDVC)
# ------------------------------------------------------------------------
# Modified from Deformable DETR(https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch
import torch.nn.functional as F
from torch import nn
import math
import copy

from misc.detr_utils import box_ops
from misc.detr_utils.misc import inverse_sigmoid
from .deformable_transformer import build_deforamble_transformer
from .criterion import SetCriterion
from .base_encoder import build_base_encoder
from .transformer_utils import (
    MLP,
    RefineTransformerDecoderLayer, RefineTransformerDecoder
)
from .matcher import build_matcher

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_query_positional_emb(Nq: int=4, dq: int=256, Bs: int=16):
    temperature = 10000
    num_pos_feats = dq
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
    dim_t = temperature ** (2 * (torch.div(dim_t, 2, rounding_mode='floor')) / num_pos_feats)
    pos_emb = torch.ones(Bs, Nq)
    pos_emb = torch.cumsum(pos_emb, dim=1)
    # import pdb; pdb.set_trace()
    pos_emb = pos_emb[:, :, None] / dim_t
    pos_emb = torch.stack((pos_emb[:, :, 0::2].sin(), pos_emb[:, :, 1::2].cos()), dim=3).flatten(2)
    return pos_emb



#=========================================================================================================



class EmbryoFormer(nn.Module):
    """ This is the PDVC module that performs dense video captioning """

    def __init__(self, base_encoder, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, opt=None, translator=None):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            num_classes: number of cell_stage classes.
            num_queries: number of event queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            opt: all configs
        """
        super().__init__()
        self.opt = opt
        self.base_encoder = base_encoder
        self.transformer = transformer

        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        self.class_head = nn.Linear(hidden_dim, num_classes)
        self.count_head = nn.Linear(hidden_dim, opt.max_eseq_length + 1)
        self.bbox_head = MLP(hidden_dim, hidden_dim, 2, 3)
        
        # self.frame_head = nn.Linear(hidden_dim, num_classes)
        # self.bbox_width_softmax = nn.Softmax(dim=1)

        self.num_feature_levels = num_feature_levels
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine

        # initialization
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_head.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_head.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_head.layers[-1].bias.data, 0)

        num_pred = transformer.decoder.num_layers
        
        if with_box_refine:
            self.class_head = _get_clones(self.class_head, num_pred)
            self.count_head = _get_clones(self.count_head, num_pred)
            self.bbox_head = _get_clones(self.bbox_head, num_pred)
            nn.init.constant_(self.bbox_head[0].layers[-1].bias.data[1:], -2)
            self.transformer.decoder.bbox_head = self.bbox_head
        else:
            nn.init.constant_(self.bbox_head.layers[-1].bias.data[1:], -2)
            self.class_head = nn.ModuleList([self.class_head for _ in range(num_pred)])
            self.count_head = nn.ModuleList([self.count_head for _ in range(num_pred)])
            self.bbox_head = nn.ModuleList([self.bbox_head for _ in range(num_pred)])
            self.transformer.decoder.bbox_head = None

        self.translator = translator


    def get_filter_rule_for_encoder(self):
        filter_rule = lambda x: 'input_proj' in x \
                                or 'transformer.encoder' in x \
                                or 'transformer.level_embed' in x \
                                or 'base_encoder' in x
        return filter_rule


    def encoder_decoder_parameters(self):
        filter_rule = self.get_filter_rule_for_encoder()
        enc_paras = []
        dec_paras = []
        for name, para in self.named_parameters():
            if filter_rule(name):
                print('enc: {}'.format(name))
                enc_paras.append(para)
            else:
                print('dec: {}'.format(name))
                dec_paras.append(para)
        return enc_paras, dec_paras


    def forward(self, dt, criterion, eval_mode=False):
        vf = dt['video_tensor']  # (N, L, C)
        mask = ~dt['video_mask']  # (N, L)
        duration = dt['video_length'][:, 1]
        N, L, C = vf.shape

        srcs, masks, pos = self.base_encoder(vf, mask, duration)
        src_flatten, temporal_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten = \
            self.transformer.prepare_encoder_inputs(srcs, masks, pos)

        memory = self.transformer.forward_encoder(src_flatten, temporal_shapes, level_start_index, 
                                                valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        srcs_lens = [src.shape[2] for src in srcs]
        splits = torch.split(memory, srcs_lens, dim=1)
        frame_embedds = splits[0]
    
        disable_iterative_refine = False
        query_embed = self.query_embed.weight
        proposals_mask = torch.ones(N, query_embed.shape[0], device=query_embed.device).bool()
        init_reference, tgt, reference_points, query_embed = self.transformer.prepare_decoder_input_query(memory, query_embed)

        hs, inter_references = self.transformer.forward_decoder(
                                tgt, reference_points, memory, temporal_shapes,
                                level_start_index, valid_ratios, query_embed,
                                mask_flatten, proposals_mask
                            )

        others = {
            'memory': memory,
            'frame_embeddings': frame_embedds,
            'mask_flatten': mask_flatten,
            'spatial_shapes': temporal_shapes,
            'level_start_index': level_start_index,
            'valid_ratios': valid_ratios,
            'proposals_mask': proposals_mask,
        }
        if eval_mode:
            out, loss = self.parallel_prediction_full(dt, criterion, hs, init_reference, inter_references, others,
                                                      disable_iterative_refine)
        else:
            out, loss = self.parallel_prediction_matched(dt, criterion, hs, init_reference, inter_references, others,
                                                         disable_iterative_refine)

        return out, loss


    def predict_event_num(self, counter, hs_lid):
        hs_lid_pool = torch.max(hs_lid, dim=1, keepdim=False)[0]  # [bs, feat_dim]
        outputs_class0 = counter(hs_lid_pool)
        return outputs_class0


    def parallel_prediction_full(self, dt, criterion, hs, init_reference, inter_references, others,
                                 disable_iterative_refine): 
        outputs_classes = []
        outputs_classes0 = []
        outputs_coords = []

        num_pred = hs.shape[0]
        for l_id in range(hs.shape[0]):
            if l_id == 0:
                reference = init_reference
            else:
                reference = inter_references[l_id - 1]  # [decoder_layer, batch, query_num, ...]
            hs_lid = hs[l_id]
            outputs_class = self.class_head[l_id](hs_lid)  # [bs, num_query, N_class]
            output_count = self.predict_event_num(self.count_head[l_id], hs_lid)
            tmp = self.bbox_head[l_id](hs_lid)  # [bs, num_query, 2]

            if disable_iterative_refine:
                outputs_coord = reference
            else:
                reference = inverse_sigmoid(reference)
                if reference.shape[-1] == 2:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 1
                    tmp[..., :2] += reference
                outputs_coord = tmp.sigmoid()  # [bs, num_query, 2]

            outputs_classes.append(outputs_class)
            outputs_classes0.append(output_count)
            outputs_coords.append(outputs_coord)

        outputs_class = torch.stack(outputs_classes)  # [decoder_layer, bs, num_query, N_class]
        output_count = torch.stack(outputs_classes0)
        outputs_coord = torch.stack(outputs_coords)  # [decoder_layer, bs, num_query, 2]

        all_out = {'pred_logits': outputs_class,
                   'pred_count': output_count,
                   'pred_boxes': outputs_coord,
                   # 'caption_probs': outputs_cap_probs,
                   # 'seq': outputs_cap_seqs
                   }
        out = {k: v[-1] for k, v in all_out.items()}

        if self.aux_loss:
            ks, vs = list(zip(*(all_out.items())))
            out['aux_outputs'] = [{ks[i]: vs[i][j] for i in range(len(ks))} for j in range(num_pred - 1)]
            loss, last_indices, aux_indices = criterion(out, dt['video_target'])
        else:
            loss, last_indices = criterion(out, dt['video_target'])
        return out, loss



    def parallel_prediction_matched(self, dt, criterion, hs, init_reference, inter_references, others,
                                    disable_iterative_refine):
        outputs_classes = []
        outputs_counts = []
        outputs_coords = []
        outputs_maps = []

        num_pred = hs.shape[0]
        for l_id in range(num_pred):
            hs_lid = hs[l_id]
            reference = init_reference if l_id == 0 else inter_references[
                l_id - 1]  # [decoder_layer, batch, query_num, ...]
            outputs_class = self.class_head[l_id](hs_lid)  # [bs, num_query, N_class]
            outputs_count = self.predict_event_num(self.count_head[l_id], hs_lid)
            tmp = self.bbox_head[l_id](hs_lid)  # [bs, num_query, 2]
            outputs_map = torch.einsum('bsf,bqf->bsq', others['frame_embeddings'], hs_lid)

            if disable_iterative_refine:
                outputs_coord = reference
            else:
                reference = inverse_sigmoid(reference)
                if reference.shape[-1] == 2:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 1
                    tmp[..., :1] += reference
                outputs_coord = tmp.sigmoid()  # [bs, num_query, 2]

            outputs_classes.append(outputs_class)
            outputs_counts.append(outputs_count)
            outputs_coords.append(outputs_coord)
            outputs_maps.append(outputs_map)

        outputs_class = torch.stack(outputs_classes)  # [decoder_layer, bs, num_query, N_class]
        outputs_count = torch.stack(outputs_counts)
        outputs_coord = torch.stack(outputs_coords)  # [decoder_layer, bs, num_query, 2]
        outputs_map = torch.stack(outputs_maps)

        all_out = {
            'pred_logits': outputs_class,
            'pred_count': outputs_count,
            'pred_boxes': outputs_coord,
            'pred_maps': outputs_map,
            }
        out = {k: v[-1] for k, v in all_out.items()}

        if self.aux_loss:
            ks, vs = list(zip(*(all_out.items())))
            out['aux_outputs'] = [{ks[i]: vs[i][j] for i in range(len(ks))} for j in range(num_pred - 1)]
            loss, last_indices, aux_indices = criterion(out, dt['video_target'])
        else:
            loss, last_indices = criterion(out, dt['video_target'])

        return out, loss


#=========================================================================================================



class PostProcess(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size] containing the size of each video of the batch
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        N, N_q, N_class = out_logits.shape
        assert len(out_logits) == len(target_sizes)

        # prob = out_logits.sigmoid()        
        prob = F.softmax(out_logits, dim=-1)

        # topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), N_q, dim=1)
        # scores = topk_values
        # topk_boxes = topk_indexes // out_logits.shape[2]
        # labels = topk_indexes % out_logits.shape[2]
        # boxes = box_ops.box_cl_to_xy(out_bbox)
        # raw_boxes = copy.deepcopy(boxes)
        # boxes[boxes < 0] = 0
        # boxes[boxes > 1] = 1
        # boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 2))
        # scale_fct = torch.stack([target_sizes, target_sizes], dim=1)
        # boxes = boxes * scale_fct[:, None, :]
        # eseq_lens = outputs['pred_count'].argmax(dim=-1).clamp(min=1)

        query_cls_preds = torch.argmax(prob, dim=-1, keepdim=False)

        ex_query_indices = torch.nonzero(query_cls_preds==0, as_tuple=False)
        b_idx, q_idx = ex_query_indices[:,0], ex_query_indices[:,1]

        out_bbox[b_idx, q_idx, :] = torch.tensor([0.0, 0.0]).to(self.opt.device)
        centers, widths = out_bbox[:,:,0], out_bbox[:,:,1]
        centers, indices = torch.sort(centers, dim=-1, descending=False)
        widths = torch.gather(widths, dim=-1, index=indices)

        abs_w = torch.abs(widths)
        abs_sum = torch.sum(abs_w, dim=-1, keepdim=True)
        widths = abs_w / (abs_sum + 1e-8)
        widths = widths * target_sizes[:, None]
        # widths = F.softmax(widths, dim=-1) * target_sizes[:, None]

        return torch.gather(query_cls_preds, dim=-1, index=indices), widths.to(torch.int64)


#=========================================================================================================



def build(args):
    device = torch.device(args.device)
    base_encoder = build_base_encoder(args)
    transformer = build_deforamble_transformer(args)

    model = EmbryoFormer(
        base_encoder,
        transformer,
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        opt=args
    )
    weight_dict = args.weight_dict
    matcher = build_matcher(args)

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = args.losses
    aux_losses = args.aux_losses

    criterion = SetCriterion(args.num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha,
                             focal_gamma=args.focal_gamma, opt=args)

    criterion.to(device)
    postprocessors = {'bbox': PostProcess(args)}
    return model, criterion, postprocessors

