import os
import sys
import math
import argparse
import json
import cv2

import numpy as np
from tqdm import tqdm
import torch
from torchvision import transforms
from torchvision.ops import nms
from typing import List

sys.path.insert(0, '../lib')
sys.path.insert(0, '../model')
from data.Coco import COCODataset
from utils import misc_utils, pytorch_nms_utils
from det_oprs import bbox_opr

def eval_all(args, config, network):
    # model_path
    saveDir = os.path.join('../model', args.model_dir, config.model_dir)
    model_file = os.path.join(saveDir, 
            'dump-{}.pth'.format(args.resume_weights))   
    print(model_file)
    assert os.path.exists(model_file)
    # get devices
    str_devices = args.devices
    devices = misc_utils.device_parser(str_devices) 
	# load data
    img_folder = config.eval_folder
    source = config.eval_source
    coco = COCODataset(config, img_folder, source, is_train=False)      	
    print(coco.__len__())
    len_dataset = coco.__len__()	
    config.drise_output_format = True
    figdir = './visualization/drise/'
    if not os.path.exists(figdir):
        os.makedirs(figdir)
    inference(config, network, model_file, coco, 0, len_dataset, figdir)
    
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

def inference(config, network, model_file, dataset, start, end, figdir):
    torch.set_default_tensor_type('torch.FloatTensor')
    # init model
    net = network(config)
    net.cuda()
    net = net.eval()
    check_point = torch.load(model_file)
    net.load_state_dict(check_point['state_dict'], strict=False)
    # init data
    dataset.ids = dataset.ids[start:end]
    data_iter = torch.utils.data.DataLoader(dataset=dataset, shuffle=False)
        	
	# inference
    coco_id2id = config.coco_id2id

    max_img_size = [config.eval_image_max_size+10, config.eval_image_max_size+10]
    max_masks = generate_max_masks(max_img_size)
    pbar = tqdm(total=end-start, ncols=50)
    for (image, im_info, ID, anno, image_ori) in data_iter:
        image_ori = image_ori[0].numpy()
        image_ori = cv2.cvtColor(image_ori,cv2.COLOR_RGB2BGR)

        pred_boxes, _, _ = net(image.cuda(), im_info.cuda())
        scale_y, scale_x = im_info[0, 2:4]

        # NMS
        keep = ml_nms(pred_boxes, 0.5, 100) #config.test_nms, config.max_boxes_of_image
        pred_boxes = pred_boxes[keep]

        # select high confidence ones
        keep = pred_boxes[:, 4] > 0.1 #config.pred_cls_threshold
        pred_boxes = pred_boxes[keep]
       
        pred_boxes_scale = pred_boxes[:,:6].clone()
        pred_boxes_scale[:, 0:4:2] /= scale_x
        pred_boxes_scale[:, 1:4:2] /= scale_y     

        ori_imgsize = im_info[0, -2:]
        h,w = int(ori_imgsize[0]), int(ori_imgsize[1])
        resize = transforms.Resize([h,w]) 

        # get gts
        anno = [obj for obj in anno if obj["iscrowd"] == 0]
        gt_boxes = [obj["bbox"] for obj in anno]
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)
        gt_boxes[:,2:4] += gt_boxes[:,0:2]
        gt_classes = [coco_id2id[int(obj["category_id"])] for obj in anno]        
        gt_classes = torch.tensor(gt_classes) 

        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            continue

        pred_boxes_scale = pred_boxes_scale.cpu()
        overlaps = bbox_opr.box_overlap_opr(pred_boxes_scale[:,:4], gt_boxes)
        class_mask = pred_boxes_scale[:,5].unsqueeze(1) == gt_classes.unsqueeze(0)
        overlaps = overlaps*class_mask

        gt_inds = []
        box_inds = []
        for j in range(min(len(pred_boxes_scale), len(gt_boxes))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)
            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)            
            if gt_ovr <= 0.9:  # the best covered gt with iou<=0.5
                break
            else:
                # find the proposal box that covers the best covered gt box
                box_ind = argmax_overlaps[gt_ind]
                # mark the proposal box and the gt box as used
                overlaps[box_ind, :] = -1
                overlaps[:, gt_ind] = -1
                gt_inds.append(gt_ind)
                box_inds.append(box_ind)
        
        box_inds = torch.tensor(box_inds).long()
        if len(box_inds)>0:
            pred_boxes = pred_boxes[box_inds]
            print('num of preds:', len(pred_boxes))
            drises = d_rise(max_masks, net, image, im_info, pred_boxes)  # 1, num_pred, image_size
            print('drise shape:', drises.shape)
            drises = resize(drises)
        for k, gt_ind in enumerate(gt_inds):
            drise_map = drises[k]
            drise_map = (drise_map - drise_map.min()) / (drise_map.max()-drise_map.min())

            np.save(figdir+'{}_{}_{}.npy'.format(int(ID), int(gt_ind), 'npy'), drise_map)
            gt_box = gt_boxes[gt_ind]  
            pred_box = pred_boxes_scale[box_inds[k]]
            tmp_img = cv2.rectangle(image_ori.copy(), \
                (int(gt_box[0]), int(gt_box[1])), (int(gt_box[2]), int(gt_box[3])), (255,0,0), 2)
            cv2.imwrite(figdir+'{}_{}_{}.jpg'.format(int(ID), int(gt_ind), 'object'), tmp_img)
            color = cv2.applyColorMap((drise_map.numpy()*255).astype(np.uint8), cv2.COLORMAP_JET)
            c_ret = np.clip(image_ori * (1 - 0.5) + color * 0.5, 0, 255).astype(np.uint8)
            c_ret = cv2.rectangle(c_ret, \
                (int(pred_box[0]), int(pred_box[1])), (int(pred_box[2]), int(pred_box[3])), (0,0,255), 2)
            cv2.imwrite(figdir+'{}_{}_{}.jpg'.format(int(ID), int(gt_ind), 'drise'), c_ret)

        pbar.update(1)

    

def d_rise(max_masks, net, image, im_info, preds):
    N = len(max_masks)
    image_size = image.size()[-2:]
    max_img_size = max_masks.size()[-2:]
    delta_h, delta_w = int(max_img_size[0]-image_size[0]), int(max_img_size[1]-image_size[1]) 
    d_hs = torch.randint(0, max(delta_h,1), size=(N,))
    d_ws = torch.randint(0, max(delta_w,1), size=(N,))         

    drise = image.new_zeros((len(preds), *image.size()[-2:]))
    # image = image.cuda()
    for i in tqdm(range(N)):
        d_h, d_w = d_hs[i], d_ws[i]
        mask = max_masks[i][d_h:d_h + image_size[0], d_w:d_w + image_size[1]].unsqueeze(0)

        masked_img = image * mask.unsqueeze(0)
        masked_preds,_,_ = net(masked_img.cuda(), im_info.cuda())  # box, score, tag, probs

        pair_ious = bbox_opr.box_overlap_opr(masked_preds[:,:4], preds[:,:4]) # num_masked_pred x num_pred
        prob_cosine = (masked_preds[:,6:].unsqueeze(1) * preds[:,6:].unsqueeze(0)).sum(-1) # num_masked_pred x num_pred
        
        if len(masked_preds)>0:
            w_mask = (pair_ious * prob_cosine).max(0)[0]  # num_pred
            drise += w_mask[:,None, None].cpu() * mask
    return drise	    

def generate_max_masks(max_img_size):
    N = 5000
    s = 16
    p = 0.5
    # resize to max_h, max_w
    resize = transforms.Resize((max_img_size[0],max_img_size[1]))
    grid = (torch.rand(size=(N, s, s)) < p).float()   
    grid = resize(grid)
    return grid



def run_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', '-md', default=None, required=True, type=str)
    parser.add_argument('--resume_weights', '-r', default=None, required=True, type=str)
    parser.add_argument('--devices', '-d', default='0', type=str)
    os.environ['NCCL_IB_DISABLE'] = '1'
    args = parser.parse_args()
    # import libs
    model_root_dir = os.path.join('../model/', args.model_dir)
    sys.path.insert(0, model_root_dir)
    from config_coco import config
    from network import Network
    eval_all(args, config, Network)

if __name__ == '__main__':
    run_test()

