import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torchvision.ops import roi_align

from backbone.resnet50 import ResNet50
from backbone.fpn import FPN
from module.rpn import RPN
from layers.pooler import assign_boxes_to_levels, roi_pooler
from det_oprs.bbox_opr import bbox_transform_inv_opr
from det_oprs.fpn_roi_target import fpn_roi_target
from det_oprs.loss_opr import softmax_loss, smooth_l1_loss
from det_oprs.utils import get_padded_tensor

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
        rcnn_rois, rcnn_labels, rcnn_bbox_targets, _ = fpn_roi_target(
                config, rpn_rois, im_info, gt_boxes, top_k=1)

        loss_dict_rcnn = self.RCNN(fpn_fms, rcnn_rois,
                rcnn_labels, rcnn_bbox_targets)
        loss_dict.update(loss_dict_rpn)
        loss_dict.update(loss_dict_rcnn)
        return loss_dict

    def _forward_test(self, image, im_info):
        fpn_fms, res_fms = self.FPN(image)
        ## resnet50_fms p3-p5 (stride: 8,16,32)
        res_fms = res_fms[::-1]
        features = {
            'fpn': fpn_fms[1:][::-1],  # fpn_fms p2-p5 (stride: 4, 8, 16, 32)
            'resnet_p3': res_fms[0],
            'resnet_p4': res_fms[1],
            'resnet_p5': res_fms[2]
        }        
        rpn_rois = self.RPN(fpn_fms, im_info)
        pred_bbox, pred_levels = self.RCNN(fpn_fms, rpn_rois)
        return pred_bbox, features, pred_levels

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

    def forward(self, fpn_fms, rcnn_rois, labels=None, bbox_targets=None):
        config = self.config
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
            # loss for regression
            labels = labels.long().flatten()
            fg_masks = labels > 0
            valid_masks = labels >= 0
            # multi class
            pred_delta = pred_delta.reshape(-1, config.num_classes, 4)
            fg_gt_classes = labels[fg_masks]
            pred_delta = pred_delta[fg_masks, fg_gt_classes, :]
            localization_loss = smooth_l1_loss(
                pred_delta,
                bbox_targets[fg_masks],
                config.rcnn_smooth_l1_beta)
            # loss for classification
            objectness_loss = softmax_loss(pred_cls, labels)
            objectness_loss = objectness_loss * valid_masks
            normalizer = 1.0 / valid_masks.sum().item()
            loss_rcnn_loc = localization_loss.sum() * normalizer
            loss_rcnn_cls = objectness_loss.sum() * normalizer
            loss_dict = {}
            loss_dict['loss_rcnn_loc'] = loss_rcnn_loc
            loss_dict['loss_rcnn_cls'] = loss_rcnn_cls
            return loss_dict
        else:
            class_num = pred_cls.shape[-1] - 1
            tag = torch.arange(class_num).type_as(pred_cls)+1
            tag = tag.repeat(pred_cls.shape[0], 1).reshape(-1,1)
            pred_probs = F.softmax(pred_cls, dim=-1)
            pred_scores = pred_probs[:, 1:].reshape(-1, 1)
            pred_delta = pred_delta[:, 4:].reshape(-1, 4)
            base_rois = rcnn_rois[:, 1:5].repeat(1, class_num).reshape(-1, 4)
            levels = level_assignments.reshape(-1, 1).repeat(1, class_num).reshape(-1, 1)
            pred_bbox = restore_bbox(base_rois, pred_delta, \
                config.bbox_normalize_stds, config.bbox_normalize_means, True)
            if config.drise_output_format:
                pred_probs = pred_probs.unsqueeze(1).repeat(1,class_num,1).reshape(-1, config.num_classes)
                pred_bbox = torch.cat([pred_bbox, pred_scores, tag, pred_probs], axis=1)
            else:
                pred_bbox = torch.cat([pred_bbox, pred_scores, tag, pred_delta], axis=1)
            return pred_bbox, levels

def restore_bbox(rois, deltas, stds, means, unnormalize=True):
    if unnormalize:
        std_opr = torch.tensor(stds[None, :]).type_as(deltas)
        mean_opr = torch.tensor(means[None, :]).type_as(deltas)
        deltas = deltas * std_opr
        deltas = deltas + mean_opr
    pred_bbox = bbox_transform_inv_opr(rois, deltas)
    return pred_bbox
