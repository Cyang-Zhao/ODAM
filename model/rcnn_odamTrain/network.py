import math

import torch
from torch import nn
from torch import autograd
import torch.nn.functional as F
from torchvision import transforms
from torchvision.ops import roi_align
import numpy as np

from backbone.resnet50 import ResNet50
from backbone.fpn import FPN
from module.rpn import RPN
from layers.pooler import assign_boxes_to_levels, roi_pooler
from det_oprs.bbox_opr import bbox_transform_inv_opr, box_overlap_opr, paired_box_overlap_opr
from det_oprs.fpn_roi_target import fpn_roi_target
from det_oprs.loss_opr import softmax_loss, smooth_l1_loss
from det_oprs.utils import get_padded_tensor

INF = 100000000

class Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.resnet50 = ResNet50(config.backbone_freeze_at, False)
        self.FPN = FPN(self.resnet50, 2, 6)
        self.RPN = RPN(config)
        self.RCNN = RCNN(config)

    def forward(self, image, im_info, gt_boxes=None):
        config = self.config
        image = (image - torch.tensor(config.image_mean[None, :, None, None]).type_as(image)) / (
                torch.tensor(config.image_std[None, :, None, None]).type_as(image))
        image = get_padded_tensor(image, 64)
        if self.training:
            return self._forward_train(image, im_info, gt_boxes)
        else:
            return self._forward_test(image, im_info)

    def _forward_train(self, image, im_info, gt_boxes):
        config = self.config
        loss_dict = {}
        fpn_fms, _ = self.FPN(image)
        # fpn_fms stride: 64,32,16,8,4, p6->p2
        rpn_rois, loss_dict_rpn = self.RPN(fpn_fms, im_info, gt_boxes)

        # top_k=1: for each rpn_roi, only assign the gt object which fits it best 
        rcnn_rois, rcnn_labels, rcnn_bbox_targets, rcnn_gts = fpn_roi_target(
                config, rpn_rois, im_info, gt_boxes, top_k=1)
        loss_dict_rcnn = self.RCNN(fpn_fms, rcnn_rois,
                rcnn_labels, rcnn_bbox_targets, rcnn_gts)
        loss_dict.update(loss_dict_rpn)
        loss_dict.update(loss_dict_rcnn)
        return loss_dict

    def _forward_test(self, image, im_info):
        fpn_fms, _ = self.FPN(image)
        rpn_rois = self.RPN(fpn_fms, im_info)
        pred_bbox = self.RCNN(fpn_fms, rpn_rois)
        return pred_bbox.detach()

class RCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # roi head
        self.fc1 = nn.Linear(256*7*7, 1024)
        self.fc2 = nn.Linear(1024, 1024)

        for l in [self.fc1, self.fc2]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.constant_(l.bias, 0)
        # box predictor
        self.pred_cls = nn.Linear(1024, config.num_classes)
        self.pred_delta = nn.Linear(1024, config.num_classes * 4)
        for l in [self.pred_cls]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
        for l in [self.pred_delta]:
            nn.init.normal_(l.weight, std=0.001)
            nn.init.constant_(l.bias, 0)

    @torch.enable_grad()
    def forward(self, fpn_fms, rcnn_rois, labels=None, bbox_targets=None, assigned_gts=None):
        config = self.config
        bbox_stds, bbox_means = config.bbox_normalize_stds, config.bbox_normalize_means
        # input p2-p5
        fpn_fms = fpn_fms[1:][::-1]
        stride = [4, 8, 16, 32]
        assert len(fpn_fms) == len(stride)
        max_level = int(math.log2(stride[-1]))
        min_level = int(math.log2(stride[0]))
        assert (len(stride) == max_level - min_level + 1)        
        level_assignments = assign_boxes_to_levels(rcnn_rois, min_level, max_level, 224, 4)
        pool_features = fpn_fms[0].new_zeros((len(rcnn_rois), fpn_fms[0].shape[1], 7, 7))
        for level, (fm_level, scale_level) in enumerate(zip(fpn_fms, stride)):
            inds = torch.nonzero(level_assignments == level, as_tuple=False).squeeze(1)
            rois_level = rcnn_rois[inds]
            pool_features[inds] = roi_align(fm_level, rois_level, (7,7), spatial_scale=1.0/scale_level,
                    sampling_ratio=-1, aligned=True)  
        # pool_features = roi_pooler(fpn_fms, rcnn_rois, stride, (7, 7), "ROIAlignV2")

        flatten_feature = torch.flatten(pool_features, start_dim=1)
        flatten_feature = F.relu_(self.fc1(flatten_feature))     
        flatten_feature = F.relu_(self.fc2(flatten_feature))
        pred_cls = self.pred_cls(flatten_feature)
        pred_delta = self.pred_delta(flatten_feature)

        if self.training:
            labels = labels.long().flatten()  # class
            fg_masks = labels > 0
            valid_masks = labels >= 0
            fg_gt_classes = labels[fg_masks]

            # loss for regression
            # multi class
            pred_delta = pred_delta.reshape(-1, config.num_classes, 4)
            pred_delta = pred_delta[fg_masks, fg_gt_classes, :]         
            localization_loss = smooth_l1_loss(
                pred_delta,
                bbox_targets[fg_masks],
                config.rcnn_smooth_l1_beta)
            
            pred_bbox = restore_bbox(rcnn_rois[fg_masks], pred_delta, \
                bbox_stds, bbox_means, True)
            
            # loss for iou prediction
            gt_bbox = restore_bbox(rcnn_rois[fg_masks], bbox_targets[fg_masks], \
                    bbox_stds, bbox_means, True)
            pred_gt_ious = paired_box_overlap_opr(pred_bbox, gt_bbox)          

            # get pool grads for fg samples
            fg_inds = fg_masks.nonzero(as_tuple=True)[0]
            pool_grads = self.get_gradient(pred_cls, pool_features) # N, C-1, 256, 7,7
            pool_dams = F.relu_((pool_grads[fg_inds, fg_gt_classes-1,:,:,:] * \
                pool_features[fg_masks]).sum(1)) # Num_pred,7,7

            dam_size = fpn_fms[2].size()[-2:] # image_size // 16
            rois_fg = rcnn_rois[fg_masks, 1:5]
            bids = rcnn_rois[fg_masks, 0].long()
            level_assignments_fg = level_assignments[fg_masks]

            pred_dams = get_dams(\
                pool_dams, bids, rois_fg, fpn_fms, stride, level_assignments_fg, dam_size)

            assigned_gts_fg = assigned_gts[fg_masks]            
            loss_rcnn_match = match_loss(pred_dams, assigned_gts_fg, pred_bbox, pred_gt_ious)           

            # loss for classification
            objectness_loss = softmax_loss(pred_cls, labels)
            objectness_loss = objectness_loss * valid_masks
            normalizer = 1.0 / valid_masks.sum().item()
            loss_rcnn_loc = localization_loss.sum() * normalizer
            loss_rcnn_cls = objectness_loss.sum() * normalizer
            loss_rcnn_match = 0.2 * loss_rcnn_match

            loss_dict = {}
            loss_dict['loss_rcnn_loc'] = loss_rcnn_loc
            loss_dict['loss_rcnn_cls'] = loss_rcnn_cls
            if loss_rcnn_match > 0:
                loss_dict['loss_rcnn_match'] = loss_rcnn_match

            return loss_dict
        else:
            pool_grads = self.get_gradient(pred_cls, pool_features)
            class_num = pred_cls.shape[-1] - 1
            level_assignments = level_assignments.repeat(1, class_num).reshape(-1)
            tag = torch.arange(class_num).type_as(pred_cls)+1
            tag = tag.repeat(pred_cls.shape[0], 1).reshape(-1,1)
            pred_scores = F.softmax(pred_cls, dim=-1)[:, 1:].reshape(-1, 1)
            pred_delta = pred_delta[:, 4:].reshape(-1, 4)
            base_rois = rcnn_rois.repeat(1, class_num).reshape(-1, 5)
            keep = pred_scores[:, 0] > config.pred_cls_threshold
            pred_scores, pred_delta, base_rois, tag, level_assignments = \
                pred_scores[keep],pred_delta[keep],base_rois[keep],tag[keep],level_assignments[keep]
            pool_grads = pool_grads.reshape(-1,256,7,7)[keep]
  
            bids = base_rois[:, 0].long()
            base_rois = base_rois[:, 1:5]
            pred_bbox = restore_bbox(base_rois, pred_delta, bbox_stds, bbox_means, True)
            
            # get pool grads
            pred_index = torch.arange(pred_cls.shape[0]).type_as(tag).repeat(1, class_num).reshape(-1)
            pred_index = pred_index[keep].long()
            # pool_grads = self.get_pool_gradient(tag, pred_index) # Num_pred,256,7,7
            pool_dams = F.relu_((pool_grads * pool_features[pred_index]).sum(1)) # Num_pred,7,7
            
            # get dam maps
            dam_size = fpn_fms[1].size()[-2:] # image_size // 8
            pred_dams = get_dams(pool_dams, bids, base_rois, fpn_fms, stride, level_assignments, dam_size)

            dam_size = pred_dams.new_tensor(dam_size).repeat(len(pred_scores),1)
            pred_bbox = torch.cat([pred_bbox, pred_scores, tag, pred_dams, dam_size], axis=1)
            return pred_bbox

    def get_gradient(self, pred, pool_features):
        grads = []
        with torch.enable_grad():
            for c in range(1, self.config.num_classes):
                grad_mask = pred.new_zeros(pred.shape)
                grad_mask[:, c] = 1.0
                grad = torch.autograd.grad(
                    pred, 
                    pool_features, 
                    grad_outputs=grad_mask, 
                    retain_graph=True)[0]
                grads.append(grad)

        return torch.stack(grads, dim=1)  # N, C-1, 256, 7,7

def get_dams(pool_maps, bids, rois, fpn_fms, stride, level_assignments, dam_size):
    resize = transforms.Resize(dam_size)
    pred_dams = pool_maps.new_zeros(len(pool_maps), dam_size[0]*dam_size[1])
    # project dam to locations on original feature map
    for bid in bids.unique():
        inds = torch.nonzero(bids == bid, as_tuple=True)[0]
        for level, (fm_level, scale_level) in enumerate(zip(fpn_fms, stride)):
            inds_level = inds[level_assignments[inds]==level]
            if len(inds_level)>0:
                dam_maps = roi_align_inv(\
                    pool_maps[inds_level], rois[inds_level], 1.0/scale_level, fm_level.size()[-2:])    
                dam_maps = resize(dam_maps)                           
                 # Num_pred,dam_size
                dam_maps = F.normalize(dam_maps.reshape(len(inds_level), -1))
                pred_dams[inds_level,:] = dam_maps
    return pred_dams

def restore_bbox(rois, deltas, stds, means, unnormalize=True):
    if unnormalize:
        std_opr = torch.tensor(stds[None, :]).type_as(deltas)
        mean_opr = torch.tensor(means[None, :]).type_as(deltas)
        deltas = deltas * std_opr
        deltas = deltas + mean_opr
    pred_bbox = bbox_transform_inv_opr(rois, deltas)
    return pred_bbox

def roi_align_inv(pool_dams, rois, scale, map_size):
    '''
        pool_dams: N,7,7
        rois: N,4 (x1,y1,x2,y2)
        scale: 1.0/scale_level
        map_size: the feature map size before roi_align
    '''
    N, h_pool, w_pool = pool_dams.shape
    rois = rois * scale
    rois[:,0::2] = rois[:,0::2].clamp(min=0., max=map_size[1]-1)
    rois[:,1::2] = rois[:,1::2].clamp(min=0., max=map_size[0]-1)
    rois_x_low, rois_y_low = rois[:,0].floor(), rois[:,1].floor()
    rois_x_high, rois_y_high = rois[:,2].ceil(), rois[:,3].ceil()
    rois_w_max = (rois_x_high - rois_x_low).max().int()+1
    rois_h_max = (rois_y_high - rois_y_low).max().int()+1
    M = rois_w_max * rois_h_max

    shift_y, shift_x = torch.meshgrid(
            torch.arange(0, rois_h_max, 1, dtype=torch.float32, device=rois.device), 
            torch.arange(0, rois_w_max, 1, dtype=torch.float32, device=rois.device))
    rois_grids = torch.stack((shift_x.reshape(-1), shift_y.reshape(-1)), dim=1) # W_max*H_max, 2

    rois_start_locs = torch.stack((rois_x_low, rois_y_low), dim=1) # N, 2
    rois_grids = rois_grids.repeat(N,1,1) + rois_start_locs.reshape(N, 1, 2) # N, W_max*H_max, 2

    grids_on_pool = ((rois_grids - rois[:,:2].reshape(N,1,2)) / \
                (rois[:,2:]-rois[:,:2]).reshape(N,1,2)) * \
                rois.new_tensor([w_pool-1, h_pool-1]).reshape(1,1,2) # N, W_max*H_max, 2 

    grids_x_low, grids_x_high = grids_on_pool[:,:,0].floor(), grids_on_pool[:,:,0].ceil()
    grids_y_low, grids_y_high = grids_on_pool[:,:,1].floor(), grids_on_pool[:,:,1].ceil()

    x_l_valid = (grids_x_low>=0) * (grids_x_low<w_pool)
    y_l_valid = (grids_y_low>=0) * (grids_y_low<h_pool) 
    x_h_valid = (grids_x_high>=0) * (grids_x_high<w_pool)
    y_h_valid = (grids_y_high>=0) * (grids_y_high<h_pool)

    ids, valid_inds = (x_l_valid * y_l_valid * x_h_valid * y_h_valid).nonzero(as_tuple=True)   
    pool_dams = pool_dams.reshape(N, -1) # N, 49

    x_weight = grids_on_pool[:,:,0] - grids_x_low
    y_weight = grids_on_pool[:,:,1] - grids_y_low

    # the dam value on top left corner
    tl_values = pool_dams.new_zeros(N, M)
    # ids, valid_inds = (x_l_valid * y_l_valid).nonzero(as_tuple=True)
    tl_locs = (grids_y_low[ids, valid_inds] * w_pool + grids_x_low[ids, valid_inds]).long()  # locations on pool_dam map
    tl_values[ids, valid_inds] = pool_dams[ids, tl_locs]
    roi_dam_values = tl_values * (1-x_weight) * (1-y_weight)
    del tl_values
    
    # the dam value on top right corner
    tr_values = pool_dams.new_zeros(N, M)
    # ids, valid_inds = (x_h_valid * y_l_valid).nonzero(as_tuple=True)
    tr_locs = (grids_y_low[ids, valid_inds] * w_pool + grids_x_high[ids, valid_inds]).long()  # locations on pool_dam map
    tr_values[ids, valid_inds] = pool_dams[ids, tr_locs]
    roi_dam_values += tr_values * x_weight * (1-y_weight)
    del tr_values

    # the dam value on bottom left corner
    bl_values = pool_dams.new_zeros(N, M)
    # ids, valid_inds = (x_l_valid * y_h_valid).nonzero(as_tuple=True)
    bl_locs = (grids_y_high[ids, valid_inds] * w_pool + grids_x_low[ids, valid_inds]).long()  # locations on pool_dam map
    bl_values[ids, valid_inds] = pool_dams[ids, bl_locs]
    roi_dam_values += bl_values * y_weight * (1-x_weight)
    del bl_values

    # the dam value on bottom right corner
    br_values = pool_dams.new_zeros(N, M)
    # ids, valid_inds = (x_h_valid * y_h_valid).nonzero(as_tuple=True)
    br_locs = (grids_y_high[ids, valid_inds] * w_pool + grids_x_high[ids, valid_inds]).long()  # locations on pool_dam map
    br_values[ids, valid_inds] = pool_dams[ids, br_locs]
    roi_dam_values += br_values * x_weight * y_weight
    del br_values

    # put the values back to map size
    indices = (rois_grids[:,:,1] * map_size[1] + rois_grids[:,:,0]).long() # N,M
    ids, valid_inds = (indices < map_size[0]*map_size[1]).nonzero(as_tuple=True)
    dam_maps = pool_dams.new_zeros(N, map_size[0]*map_size[1])
    dam_maps[ids, indices[ids, valid_inds]] = roi_dam_values[ids, valid_inds] # N, map_size
    return dam_maps.reshape(N, map_size[0], map_size[1])

def match_loss(dams, objs, pred_bbox, pred_gt_iou):
    M, C = dams.shape
    Num_gt = objs.max()+1
    # find the best regression for each object
    ious = pred_gt_iou.new_zeros(M, Num_gt)
    ious[range(M), objs] = pred_gt_iou
    max_iou, max_position = ious.max(dim=0)
    max_position = max_position[max_iou>0]
    max_iou = max_iou[max_iou>0]

    pred_paired_iou = box_overlap_opr(pred_bbox, pred_bbox)
    overlap_mask = pred_paired_iou > 0
    pos_mask = (objs.view(M,1).expand(M,M) == \
                objs.view(1,M).expand(M,M)).float()
    neg_mask = overlap_mask * (1 - pos_mask) # not the same object but overlapped
    pos_pair1, pos_pair2 = pos_mask[max_position].nonzero(as_tuple=True)
    neg_pair1, neg_pair2 = neg_mask[max_position].nonzero(as_tuple=True)    


    # BCE
    if len(max_position) > 0:
        max_iou_dams = dams[max_position]
        pos_sims = (max_iou_dams[pos_pair1]*dams[pos_pair2]).sum(-1).clamp(min=1e-4, max=1-1e-4) 
        neg_sims = (max_iou_dams[neg_pair1]*dams[neg_pair2]).sum(-1).clamp(min=1e-4, max=1-1e-4) 
        
        pos_weights = pred_gt_iou[pos_pair2] / max_iou[pos_pair1]
        # loss_pos = (pos_weights*(-pos_sims.log())).sum() / max(1.,pos_sims.numel())
        loss = ((pos_weights*(-pos_sims.log())).sum()-(1-neg_sims).log().sum()) / \
            max(1.,pos_sims.numel()+neg_sims.numel())
    else:
        loss = 0.
    return loss         












