import math
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.ops import nms
import numpy as np
from typing import List

# from config import config
from backbone.resnet50 import ResNet50
from backbone.fpn import FPN
from det_oprs.bbox_opr import bbox_transform_inv_opr, box_overlap_opr, paired_box_overlap_opr
from det_oprs.loss_opr import iou_loss, sigmoid_focal_loss
from det_oprs.fcos_utils import *
from det_oprs.utils import get_padded_tensor

class Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.resnet50 = ResNet50(config.backbone_freeze_at, False)
        self.FPN = FPN(self.resnet50, 3, 7)
        self.F_Head = Fcos_Head(config.num_classes-1)

        # fcos settings
        self.ctr_radius = config.center_radius
        self.size_of_interest = config.size_of_interest
        self.reg_loss_type = config.reg_loss_type
        self.box_quality_type = config.box_quality_type
        self.focal_alpha = config.focal_loss_alpha
        self.focal_gamma = config.focal_loss_gamma

    @torch.enable_grad()
    def forward(self, image, im_info, gt_boxes=None):
        config = self.config
        # pre-processing the data
        image = (image - torch.tensor(config.image_mean[None, :, None, None]).type_as(image)) / (
                torch.tensor(config.image_std[None, :, None, None]).type_as(image))
        image = get_padded_tensor(image, 64)
        fpn_fms, res_fms = self.FPN(image)
        # resnet50_fms p3-p5 (stride: 8,16,32)
        res_fms = res_fms[::-1]
        # fpn_fms p3-p7 (stride: 8,16,32,64,128)
        fpn_fms = fpn_fms[::-1]
        pred_cls_list, pred_reg_list, pred_qly_list = self.F_Head(fpn_fms)

        pred_cls = flatten_outputs(pred_cls_list) # B,M,C
        pred_reg = flatten_outputs(pred_reg_list) # B,M,4
        pred_qly = flatten_outputs(pred_qly_list) # B,M,1

        strides = [8,16,32,64,128]          
        grids = compute_grids(fpn_fms, strides)

        features = {
            'fpn': fpn_fms,
            'resnet_p3': res_fms[0],
            'resnet_p4': res_fms[1],
            'resnet_p5': res_fms[2]
        }

        if self.training:
            loss_dict = self.fcos_criteria(
                    pred_cls, pred_reg, pred_qly, \
                    grids, strides, gt_boxes, im_info)
            return loss_dict
        else:
            # do inference
            # stride: 8,16,32,64,128
            pred_bbox, pred_levels = self.inference(
                grids, strides, pred_cls, pred_reg, pred_qly, features, im_info)
            return pred_bbox, features, pred_levels

    def fcos_criteria(self, pred_cls, pred_reg, pred_qly, 
        grids, list_strides, gt_boxes, im_info):
        config = self.config
        # [B,C,H,W]_l --> B,M,C   
        shapes_per_level = [(x.shape[2], x.shape[3]) for x in pred_reg_list]
        L = len(grids)
        num_loc_list = [len(loc) for loc in grids]
        strides = torch.cat([gt_boxes.new_ones(num_loc_list[l]) * list_strides[l] \
            for l in range(L)]).float() # M
        reg_size_ranges = torch.cat([gt_boxes.new_tensor(self.size_of_interest[l]).float().view(
            1, 2).expand(num_loc_list[l], 2) for l in range(L)]) # M x 2
        grids = torch.cat(grids, dim=0) # M x 2
        
        M = grids.shape[0]
        C = config.num_classes-1
        B = len(im_info)
        num_pos = 0
        loss_reg = 0
        loss_cls = 0
        loss_quality = 0
        
        for i in range(B):
            if config.quality_assign:
                boxes, is_in_pos, reg_target, pred_boxes = fcos_target_in_r(
                    config, grids, strides, reg_size_ranges, gt_boxes[i], self.ctr_radius,\
                    cls_prob=pred_cls[i].clone().detach(), reg_pred=pred_reg[i].clone().detach())
            else:
                boxes, is_in_pos, reg_target = fcos_target_in_r(
                    config, grids, strides, reg_size_ranges, gt_boxes[i], self.ctr_radius)
            
            if is_in_pos is not None:
                box_classes = (boxes[:,4]-1).long() # N, start from 0, not include bg 
                pos_ind, pos_obj = is_in_pos.nonzero(as_tuple=True) 
                num_pos += len(pos_ind) 
                                
                # regression loss
                reg_pos_target = reg_target[pos_ind, pos_obj] / strides[pos_ind].unsqueeze(1) # NumPos x 4
                ious, gious = iou_loss(pred_reg[i][pos_ind], reg_pos_target)
                if self.reg_loss_type == 'iou':
                    loss_reg += (-torch.log(ious)).sum()
                elif self.reg_loss_type == 'linear_iou':
                    loss_reg += (1 - ious).sum()
                elif self.reg_loss_type == 'giou':
                    loss_reg += (1 - gious).sum()
                else:
                    raise NotImplementedError

                # class-agnostic heatmap (ctrness or iou) loss
                if self.box_quality_type == "ctrness":
                    left_right = reg_pos_target[:, [0, 2]]
                    top_bottom = reg_pos_target[:, [1, 3]]
                    quality = torch.sqrt((left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                                 (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]))
                elif self.box_quality_type == "iou":
                    quality = ious                   
                else:
                    raise NotImplementedError
                loss_quality += F.binary_cross_entropy_with_logits(
                    input=pred_qly[i][pos_ind,0],
                    target=quality,
                    reduction='sum')

            # classification positive/negtive map
            cls_target = grids.new_zeros(M, C)
            if boxes is not None:
                pos_cls = box_classes[pos_obj]
                cls_target[pos_ind, pos_cls] = 1.
            loss_cls += sigmoid_focal_loss(
                pred_cls[i],
                cls_target,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
                reduction="sum"
                )

        loss_normalizer = 1.0 / max(num_pos, 1.)
        loss_reg = 2 * loss_reg * loss_normalizer
        loss_cls = loss_cls * loss_normalizer
        loss_quality = 0.5 * loss_quality * loss_normalizer
        
        loss_dict = {}
        if num_pos > 0:
            loss_dict['fcos_reg'] = loss_reg
            loss_dict['fcos_quality'] = loss_quality
        loss_dict['fcos_cls'] = loss_cls
        return loss_dict


    def inference(
        self, grids, strides, pred_cls, pred_reg, pred_qly, features, im_info):
        config = self.config
        pred_cls = pred_cls.sigmoid()
        pred_qly = pred_qly.sigmoid()
        L = len(grids)
        num_loc_list = [len(loc) for loc in grids]
        strides = torch.cat([grids[0].new_ones(num_loc_list[l],1) * strides[l] \
            for l in range(L)]).float() # M x 1
        levels = torch.cat([grids[0].new_ones(num_loc_list[l]) * l \
            for l in range(L)]).long() # M
        grids = torch.cat(grids, dim=0) # M x 2
        M = grids.shape[0]
        C = config.num_classes-1
        bid = 0 # now only inference one image once
        
        boxlists, box_locs = self.pred_box(\
            grids, pred_cls[bid], pred_reg[bid]*strides, pred_qly[bid])
        box_levels = levels[box_locs]
        
        # NMS
        # keep = ml_nms(boxlists, config.test_nms, config.max_boxes_of_image)
        # boxlists = boxlists[keep]
        # box_levels = box_levels[keep]
        return boxlists, box_levels


    def pred_box(self, grids, pred_cls, pred_reg, pred_qly):
        '''
            pred_cls: M,C
            pred_reg: M,4
            pred_qly: M,1
            B = 1
        '''
        config = self.config

        candidate_inds = pred_cls > config.pred_cls_threshold
        pre_nms_top_n = candidate_inds.sum() 
        pre_nms_topk = config.pre_nms_topk
        pre_nms_top_n = pre_nms_top_n.clamp(max=pre_nms_topk)

        pred_cls = (pred_cls * pred_qly).sqrt()
        box_scores = pred_cls[candidate_inds]
        box_locs, box_tags = candidate_inds.nonzero(as_tuple=True)

        box_regs = pred_reg[box_locs]
        box_grids = grids[box_locs]
        box_probs = pred_cls[box_locs]

        if candidate_inds.sum().item() > pre_nms_top_n.item():
            box_scores, topk_inds = box_scores.topk(pre_nms_top_n, sorted=False)
            box_tags = box_tags[topk_inds]
            box_regs = box_regs[topk_inds]
            box_grids = box_grids[topk_inds]
            box_locs = box_locs[topk_inds]
            box_probs = box_probs[topk_inds]

        detections = torch.stack([
            box_grids[:,0] - box_regs[:,0],
            box_grids[:,1] - box_regs[:,1],
            box_grids[:,0] + box_regs[:,2],
            box_grids[:,1] + box_regs[:,3],
        ], dim=1)
        # # avoid invalid boxes
        # detections[:, 2] = torch.max(detections[:, 2], detections[:, 0] + 0.01)
        # detections[:, 3] = torch.max(detections[:, 3], detections[:, 1] + 0.01)

        if config.drise_output_format:
            boxlist = torch.cat([detections, 
                box_scores.unsqueeze(1), 
                box_tags.unsqueeze(1)+1,
                F.normalize(box_probs)], axis=1).detach()
        else:
            boxlist = torch.cat([detections, 
                box_scores.unsqueeze(1), 
                box_tags.unsqueeze(1)+1,
                box_regs], axis=1)
        return boxlist, box_locs


def ml_nms(boxlist, nms_th, max_num):
    boxes = boxlist[:,:4]
    scores = boxlist[:,4]
    labels = boxlist[:,5]
    result_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    for id in torch.jit.annotate(List[int], torch.unique(labels).cpu().tolist()):
        mask = (labels == id).nonzero(as_tuple=False).view(-1)       
        keep = nms(boxes[mask], scores[mask], nms_th)
        result_mask[mask[keep]] = True
    keep = result_mask.nonzero(as_tuple=False).view(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    if len(keep) > max_num:
        keep = keep[:max_num]
    return keep


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

class Fcos_Head(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        num_convs = 4
        in_channels = 256
        cls_subnet = []
        bbox_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            cls_subnet.append(nn.GroupNorm(32, in_channels))
            cls_subnet.append(nn.ReLU(inplace=True))
            bbox_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            bbox_subnet.append(nn.GroupNorm(32, in_channels))
            bbox_subnet.append(nn.ReLU(inplace=True))
        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        # predictor
        self.cls_score = nn.Conv2d(in_channels, num_classes,
            kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, 4,
            kernel_size=3, stride=1, padding=1)
        self.quality = nn.Conv2d(in_channels, 1,
            kernel_size=3, stride=1, padding=1)
        self.scales = nn.ModuleList(
            [Scale(init_value=1.0) for _ in range(5)])

        # Initialization
        for modules in [self.cls_subnet, self.bbox_subnet, self.cls_score, self.bbox_pred, self.quality]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)
        prior_prob = 0.01
        # Use prior in model initialization to improve stability
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls_score.bias, bias_value)
        torch.nn.init.constant_(self.quality.bias, bias_value)
        torch.nn.init.constant_(self.bbox_pred.bias, 8.)


    def forward(self, features):
        pred_cls = []
        pred_reg = []
        pred_qly = []

        for l, feature in enumerate(features):
            pred_cls.append(self.cls_score(self.cls_subnet(feature))) 
            pred_reg.append(F.relu(self.scales[l](\
                self.bbox_pred(self.bbox_subnet(feature)))))
            pred_qly.append(self.quality(self.bbox_subnet(feature)))
        return pred_cls, pred_reg, pred_qly

  





