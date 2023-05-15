import math
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

from backbone.resnet50 import ResNet50
from backbone.fpn import FPN

from det_oprs.bbox_opr import bbox_transform_inv_opr, box_overlap_opr, paired_box_overlap_opr
from det_oprs.loss_opr import focal_loss, smooth_l1_loss, iou_loss, sigmoid_focal_loss
from det_oprs.fcos_utils import *
from det_oprs.utils import get_padded_tensor

eps = torch.finfo(torch.float32).eps

class Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.resnet50 = ResNet50(config.backbone_freeze_at, False)
        self.FPN = FPN(self.resnet50, 3, 7)
        self.F_Head = Fcos_Head(config)

        # fcos settings
        self.ctr_radius = config.center_radius
        self.size_of_interest = config.size_of_interest
        self.reg_loss_type = config.reg_loss_type
        self.box_quality_type = config.box_quality_type
        self.focal_alpha = config.focal_loss_alpha
        self.focal_gamma = config.focal_loss_gamma

        # cam settings
        self.rpt_r = config.grad_recept_radius
        self.rpt_n = 2*self.rpt_r + 1
        self.offset_grids = make_offset_grids(self.rpt_n)

    def forward(self, image, im_info, gt_boxes=None):
        config = self.config
        # pre-processing the data
        image = (image - torch.tensor(config.image_mean[None, :, None, None]).type_as(image)) / (
                torch.tensor(config.image_std[None, :, None, None]).type_as(image))
        image = get_padded_tensor(image, 64)
        fpn_fms, _ = self.FPN(image)
        fpn_fms = fpn_fms[::-1]
        pred_cls_list, pred_reg_list, pred_qly_list = self.F_Head(fpn_fms)

        strides = [8,16,32,64,128]          
        grids = compute_grids(fpn_fms, strides)

        if self.training:
            loss_dict = self.fcos_criteria(
                    pred_cls_list, pred_reg_list, pred_qly_list, \
                    grids, strides, gt_boxes, im_info)
            return loss_dict
        else:
            # do inference
            # stride: 8,16,32,64,128
            pred_bbox = self.per_layer_inference(
                    grids, strides, pred_cls_list, pred_reg_list, pred_qly_list, im_info)
            return pred_bbox.detach()

    def fcos_criteria(self, pred_cls_list, pred_reg_list, pred_qly_list, 
        grids, list_strides, gt_boxes, im_info):
        config = self.config
        # [B,C,H,W]_l --> B,M,C   
        shapes_per_level = [(x.shape[2], x.shape[3]) for x in pred_reg_list]
        pred_cls = flatten_outputs(pred_cls_list) # B,M,C
        pred_reg = flatten_outputs(pred_reg_list) # B,M,4
        pred_qly = flatten_outputs(pred_qly_list) # B,M,1
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
        loss_match = 0
        
        for i in range(B):
            if config.quality_assign:
                boxes, is_in_pos, reg_target, pred_boxes = fcos_target_in_r(
                    config, grids, strides, reg_size_ranges, gt_boxes[i], self.ctr_radius,\
                    cls_prob=pred_cls[i].clone().detach(), reg_pred=pred_reg[i].clone().detach())
            else:
                boxes, is_in_pos, reg_target, _ = fcos_target_in_r(
                    config, grids, strides, reg_size_ranges, gt_boxes[i], self.ctr_radius)
            
            if is_in_pos is not None:
                box_classes = (boxes[:,4]-1).long() # N, start from 0, not include bg 
                pos_ind, pos_obj = is_in_pos.nonzero(as_tuple=True) 
                num_pos += len(pos_ind) 

                # cam match loss
                if not config.quality_assign:
                    pred_boxes = get_all_predicted_box(\
                        grids, pred_reg[i].clone().detach(), strides)
                pos_pred_gt_ious = paired_box_overlap_opr(pred_boxes[pos_ind], boxes[pos_obj,:4])
                
                sample_iou, order = pos_pred_gt_ious.sort(descending=True)
                if len(pos_ind) > config.max_train_cam_sample:
                    order = order[:config.max_train_cam_sample]
                    sample_iou = sample_iou[:config.max_train_cam_sample]
                sample_ind, sample_obj = pos_ind[order], pos_obj[order]
                sample_locs, sample_cls, sample_preds = \
                        grids[sample_ind], box_classes[sample_obj], pred_boxes[sample_ind]

                # calculate cams per level
                cam_size = im_info[i,:2] // 16
                cam_resize = transforms.Resize([int(cam_size[0]), int(cam_size[1])])
                start_loc = 0
                locs_obj_level, locs_pred_level, locs_iou_level, locs_cam_level = [],[],[],[]
                for level, num in enumerate(num_loc_list):
                    in_curr_level = (sample_ind >= start_loc) * (sample_ind < start_loc+num)
                    if in_curr_level.sum()>0:
                        cam = self.cam_generator(\
                            i, cam_resize, level, 
                            sample_locs[in_curr_level]//list_strides[level], 
                            sample_cls[in_curr_level], 
                            shapes_per_level[level]) 
                        locs_obj_level.append(sample_obj[in_curr_level]) 
                        locs_pred_level.append(sample_preds[in_curr_level])   
                        locs_iou_level.append(sample_iou[in_curr_level])  
                        locs_cam_level.append(cam)
                    start_loc += num
                        
                
                locs_objs = torch.cat(locs_obj_level, dim=0)
                locs_preds = torch.cat(locs_pred_level, dim=0)
                locs_ious = torch.cat(locs_iou_level, dim=0)
                locs_cams = torch.cat(locs_cam_level, dim=0)

                loss_match += self.match_loss(\
                        locs_cams, locs_objs, locs_preds, locs_ious, len(box_classes))

                
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
        loss_match = 0.5 * loss_match
        
        loss_dict = {}

        loss_dict['fcos_reg'] = loss_reg
        loss_dict['fcos_quality'] = loss_quality
        loss_dict['fcos_cls'] = loss_cls
        if loss_match>0:
            loss_dict['fcos_match'] = loss_match
        return loss_dict

    def match_loss(self, cams, objs, preds, ious, N):
        M, C = cams.shape
        # find the highest-IoU prediction for each object
        ious_pred_gt = ious.new_zeros(M, N)
        ious_pred_gt[range(M), objs] = ious
        max_iou, max_position = ious.max(dim=0)
        max_position = max_position[max_iou>0]
        ious_pred_pairs = box_overlap_opr(preds, preds)
        overlap_mask = ious_pred_pairs > 0
        pos_mask = (objs.view(M,1).expand(M,M) == \
                    objs.view(1,M).expand(M,M)).float()
        neg_mask = overlap_mask * (1-pos_mask)   # have overlap but not the same obj

        if len(max_position)>0:
            max_iou_cams = cams[max_position]
            pos_pair1, pos_pair2 = pos_mask[max_position].nonzero(as_tuple=True)
            neg_pair1, neg_pair2 = neg_mask[max_position].nonzero(as_tuple=True)
            # BCE
            pos_sims = (max_iou_cams[pos_pair1]*cams[pos_pair2]).sum(-1).clamp(min=1e-4, max=1-1e-4) 
            neg_sims = (max_iou_cams[neg_pair1]*cams[neg_pair2]).sum(-1).clamp(min=1e-4, max=1-1e-4) 
            loss = (-pos_sims.log().sum()-(1-neg_sims).log().sum())/max(1.,pos_sims.numel()+neg_sims.numel())
        else:
            loss = 0.
        return loss


    def per_layer_inference(
        self, grids, strides, pred_cls_list, pred_reg_list, pred_qly_list, im_info):
        pred_cls_list = [x.sigmoid() for x in pred_cls_list]
        pred_qly_list = [x.sigmoid() for x in pred_qly_list] 
        boxlists = []
        cam_size = im_info[0,:2] // 16
        cam_resize = transforms.Resize([int(cam_size[0]), int(cam_size[1])])        

        for l, s in enumerate(strides):
            boxlist_level = self.pred_single_level(\
                l, s, grids[l], pred_cls_list[l], pred_reg_list[l]*s, pred_qly_list[l], cam_resize)
            if len(boxlist_level) > 0:
                boxlists.append(boxlist_level)
        boxlists = torch.cat(boxlists, dim=0)
        cam_size = cam_size.repeat(len(boxlists),1)
        return torch.cat([boxlists, cam_size], dim=1)

    def pred_single_level(self, level, stride, grids, pred_cls, pred_reg, pred_qly, cam_resize):
        config = self.config
        B,C,H,W = pred_cls.shape
        pred_cls = pred_cls.permute(0,2,3,1).reshape(-1,C)    # HxW,C one image per inference
        pred_reg = pred_reg.permute(0,2,3,1).reshape(-1,4)
        pred_qly = pred_qly.permute(0,2,3,1).reshape(-1,1)

        candidate_inds = pred_cls > config.pred_cls_threshold
        pre_nms_top_n = candidate_inds.sum() 
        pre_nms_topk = config.pre_nms_topk
        pre_nms_top_n = pre_nms_top_n.clamp(max=pre_nms_topk)

        pred_cls = (pred_cls * pred_qly).sqrt()
        box_scores = pred_cls[candidate_inds]
        box_locs, box_tags = candidate_inds.nonzero(as_tuple=True)

        box_regs = pred_reg[box_locs]
        box_grids = grids[box_locs]

        if candidate_inds.sum().item() > pre_nms_top_n.item():
            box_scores, topk_inds = box_scores.topk(pre_nms_top_n, sorted=False)
            box_tags = box_tags[topk_inds]
            box_regs = box_regs[topk_inds]
            box_grids = box_grids[topk_inds]

        detections = torch.stack([
            box_grids[:,0] - box_regs[:,0],
            box_grids[:,1] - box_regs[:,1],
            box_grids[:,0] + box_regs[:,2],
            box_grids[:,1] + box_regs[:,3],
        ], dim=1)
        # avoid invalid boxes
        detections[:, 2] = torch.max(detections[:, 2], detections[:, 0] + 0.01)
        detections[:, 3] = torch.max(detections[:, 3], detections[:, 1] + 0.01)

        # get cam map
        if len(detections) > 0:
            cams = self.cam_generator(0, cam_resize, level, box_grids//stride, box_tags, (H,W))
        else:
            cams = detections.new_tensor([]) 
        boxlist = torch.cat([detections, box_scores.unsqueeze(1), box_tags.unsqueeze(1), cams], axis=1)
        return boxlist

    def cam_generator(self, bid, cam_resize, level, locs, tags, map_size):
        '''
            bid: image id
            cam_resize: resize to final cam map size
            locs: pixel location grids on this level feature map
            tags: class of predictions
        '''       
        config = self.config
        n_locs = locs.shape[0]
        # turn the locs on HxW to locs on (H+10)x(W+10)
        locs = locs.view(n_locs,1,2) + \
            self.offset_grids.to(locs.device).view(1,self.rpt_n**2,2) # n_locs,121,2

        inps = locs.new_zeros([n_locs,config.num_classes-1,self.rpt_n,self.rpt_n])
        inps[range(n_locs),tags,self.rpt_r,self.rpt_r] = 1.0
        cam_values = self.F_Head.get_cam_patch(bid, level, inps, locs) # n_locs, 121

        # build up the original map size
        H, W = map_size
        cam_inds = (locs[:,:,1] * (W+10) + locs[:,:,0]).long()
        cam_maps = cam_values.new_zeros(n_locs, (H+10)*(W+10))
        cam_maps = cam_maps.scatter_(\
            dim=-1, index=cam_inds, src=cam_values).reshape(n_locs,(H+10),(W+10))
        cam_maps = cam_resize(cam_maps[:,5:H+5,5:W+5])
        cam_maps = F.normalize(cam_maps.reshape(n_locs, -1))
        return cam_maps      


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

class Fcos_Head(nn.Module):
    def __init__(self, config):
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
        self.cls_score = nn.Conv2d(in_channels, config.num_classes-1,
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

        # camgrad settings
        self.rpt_r = config.grad_recept_radius
        self.pad = (self.rpt_r,self.rpt_r,self.rpt_r,self.rpt_r)
        self.in_channels = in_channels

    def forward(self, features):
        pred_cls = []
        pred_reg = []
        pred_qly = []
        self.kernel_list = []
        self.gnweight_list = []
        self.relu_masks = [[] for i in range(len(features))]
        self.final_kernel = None
        self.inp_features = features

        for l, feature in enumerate(features):
            pred_cls.append(self.forward_proc(
                self.forward_proc(feature, self.cls_subnet, l, False),
                self.cls_score, l, True)) 
            pred_reg.append(F.relu(self.scales[l](\
                self.bbox_pred(self.bbox_subnet(feature)))))
            pred_qly.append(self.quality(self.bbox_subnet(feature)))

            self.relu_masks[l] = torch.stack(self.relu_masks[l][::-1], dim=0)
            self.relu_masks[l] = self.relu_masks[l].permute(1,0,2,3,4) # B,4,C,H,W
        

        self.kernel_list = self.kernel_list[::-1]
        self.gnweight_list = self.gnweight_list[::-1]
        return pred_cls, pred_reg, pred_qly

    def forward_proc(self, x, module, level, fin_conv=False):
        for name, l in module.named_modules():
            if isinstance(l, nn.Conv2d):
                x = l(x)
                if level==0: # kernel weights are the same in each level
                    if fin_conv:
                        self.final_kernel = l.weight
                    else:
                        self.kernel_list.append(l.weight)
            if isinstance(l, nn.GroupNorm):
                x = l(x)
                if level==0:            
                    self.gnweight_list.append(l.weight.view(1,self.in_channels,1,1))
            if isinstance(l, nn.ReLU):
                x = l(x)
                relu_mask = torch.gt(x, 0.0).float()
                self.relu_masks[level].append(relu_mask)
        return x   

    def gradient_cnn(self, inps, relu_masks):
        # final Conv2d
        fin_w = torch.flip(self.final_kernel.permute(1,0,2,3), [2,3])
        grad = F.conv2d(inps, fin_w, stride=1, padding=(1,1))  # M,256,11,11
        for w, gn_w, r_m in zip(*[self.kernel_list, self.gnweight_list, relu_masks]):
            # Relu r_m: [M, 256, 11, 11]
            grad = grad * r_m
            # GN gn_weights: [1,256,1,1]
            grad = grad * gn_w
            # Conv            
            w = torch.flip(w.permute(1,0,2,3), [2,3])
            grad = F.conv2d(grad, w, stride=1, padding=(1,1))  # M,256,11,11
        return grad

    def get_cam_patch(self, bid, level, inps, pos_locs):
        '''
            level: level
            bid: which image
            pos_inds: Mx121
        '''
        # features and relu masks need to be cropped based on ROIs
        rpt_n = 2*self.rpt_r + 1     
        per_feat = self.inp_features[level][bid]  # C, H, W
        per_relu_mask = self.relu_masks[level][bid] # 4layers 4,C,H,W     

        per_feat = F.pad(per_feat, self.pad, "constant", 0)
        per_relu_mask = F.pad(per_relu_mask, self.pad, "constant", 0) 

        C,H,W = per_feat.shape
        pos_inds = (pos_locs[:,:,1]*W+pos_locs[:,:,0]).long()
        M,N = pos_inds.shape       
        # pick each 11x11 ROI from features and relu masks
        pos_inds = pos_inds.unsqueeze(1).expand(M,C,N)
        feats = per_feat.unsqueeze(0).reshape(1,C,H*W).expand(M,C,H*W)
        feats = feats.gather(-1, pos_inds)
        feats = feats.reshape(M,C,rpt_n,rpt_n)
        
        masks = []
        for mask in per_relu_mask:
            mask = mask.unsqueeze(0).reshape(1,C,H*W).expand(M,C,H*W)   
            mask = mask.gather(-1, pos_inds)
            masks.append(mask.reshape(M,C,rpt_n,rpt_n))            
        masks = torch.stack(masks, dim=0)  # 4, M, 256, 11, 11 
        grad = self.gradient_cnn(inps, masks)

        # combine two grads
        cam = (feats * grad.detach()).sum(1)  
        cam = F.relu_(cam)  # M, 11, 11
        return cam.reshape(M, -1)

  





