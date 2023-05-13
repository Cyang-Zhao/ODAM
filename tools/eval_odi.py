import os
import sys
import math
import argparse
import json

import numpy as np
from tqdm import tqdm
import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.multiprocessing import Queue, Process

sys.path.insert(0, '../lib')
sys.path.insert(0, '../model')
from data.Coco import COCODataset
from utils import misc_utils, pytorch_nms_utils
from det_oprs import bbox_opr

import cv2


eps = torch.finfo(torch.float32).eps

def eval_all(config, objs_list, map_path):
	# load data
    img_folder = config.eval_folder
    source = config.eval_source
    coco = COCODataset(config, img_folder, source, is_train=False)      	
    print(coco.__len__())
    len_dataset = coco.__len__()	
    inference(config,  coco, 0, len_dataset, objs_list, map_path)


def get_map(img_id, gt_id, map_path):
    heatmap = np.load(os.path.join(map_path, '{}_{}_npy.npy'.format(img_id, gt_id)))
    return torch.tensor(heatmap)


def inference(config, dataset, start, end, objs_list, map_path):
    # init data
    dataset.ids = dataset.ids[start:end];
    data_iter = torch.utils.data.DataLoader(dataset=dataset, shuffle=False)
        	
	# inference
    coco_id2id = config.coco_id2id

    Ntp = 0
    port_other_inbox = 0  # box, energy in one other obj/ energy in all objects
    port_other_inmask = 0 # mask, energy in one other obj/ energy in all objects

    port_box_div_obj = []
    port_mask_div_obj = []

    ious = []

    pbar = tqdm(total=end-start, ncols=50)
    for (image, im_info, ID, anno, image_ori) in data_iter:
        if int(ID) in objs_list.keys():
            image_ori = image_ori[0].numpy()
            image_ori = cv2.cvtColor(image_ori,cv2.COLOR_RGB2BGR) 
            h,w = image_ori.shape[:2]

            # get gts
            anno = [obj for obj in anno if obj["iscrowd"] == 0]
            gt_boxes = [obj["bbox"] for obj in anno]
            gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)
            gt_boxes[:,2:4] += gt_boxes[:,0:2]
            gt_classes = [coco_id2id[int(obj["category_id"])] for obj in anno]        
            gt_classes = torch.tensor(gt_classes) 
            gt_masks = [obj["segmentation"] for obj in anno]

            objects = objs_list[int(ID)]
            for gt_id in objects:
                heatmap = get_map(int(ID), gt_id, map_path)  
                if heatmap is None:
                    continue
                if (heatmap.shape[0] != h) or (heatmap.shape[1] != w):
                    continue

                Ntp += 1
                total_box_mask = heatmap.new_zeros(image_ori.shape[:2])
                other_box_mask = heatmap.new_zeros(image_ori.shape[:2])
                total_mask = heatmap.new_zeros(image_ori.shape[:2])
                other_mask = heatmap.new_zeros(image_ori.shape[:2])
                tmp_single_port_inbox = []
                tmp_single_port_inmask = []
                for ind in range(len(gt_boxes)):
                    # box
                    temp_gt_box_mask = heatmap.new_zeros(heatmap.shape)
                    gt_box = gt_boxes[ind]
                    x1, y1 = int(max(0, gt_box[0].floor())), int(max(0, gt_box[1].floor()))
                    x2, y2 = int(min(w, gt_box[2].ceil())), int(min(h, gt_box[3].ceil()))
                    temp_gt_box_mask[y1:y2, x1:x2] = 1.
                    if ind != gt_id:
                        other_box_mask += temp_gt_box_mask
                        ious.append(bbox_opr.paired_box_overlap_opr_diou(gt_boxes[gt_id].unsqueeze(0), gt_box.unsqueeze(0)))
                        tmp_single_port_inbox.append((heatmap*temp_gt_box_mask).sum())
                    total_box_mask += temp_gt_box_mask
                    
                    # mask
                    temp_gt_mask = heatmap.new_zeros(heatmap.shape)
                    gt_poly = gt_masks[ind]
                    for epoly in gt_poly:
                        tmp = np.zeros(image_ori.shape[:2])
                        epoly = np.array(epoly).reshape(-1,2).astype(np.int32)              
                        cv2.fillPoly(tmp, [epoly], 1.)
                        temp_gt_mask += torch.tensor(tmp) 
                    temp_gt_mask = (temp_gt_mask>0).float()
                    if ind != gt_id:
                        other_mask += temp_gt_mask
                        tmp_single_port_inmask.append((heatmap*temp_gt_mask).sum()) 
                    total_mask += temp_gt_mask
                    

                heatmap_sum_inbox = (heatmap * (total_box_mask>0).float()).sum()+1e-10
                heatmap_sum_inmask = (heatmap * (total_mask>0).float()).sum()+1e-10  

                port_other_inbox += (heatmap*(other_box_mask>0).float()).sum()/heatmap_sum_inbox       
                port_other_inmask += (heatmap*(other_mask>0).float()).sum()/heatmap_sum_inmask 

                port_box_div_obj += list(torch.tensor(tmp_single_port_inbox)/heatmap_sum_inbox)
                port_mask_div_obj += list(torch.tensor(tmp_single_port_inmask)/heatmap_sum_inmask)
        
        pbar.update(1)

    print('Ntp:', Ntp)
    print('box odi divided by energy in box:', port_other_inbox/Ntp)
    print('mask odi divided by energy in mask:', port_other_inmask/Ntp)

    method_tag = map_path.split('/')[-1]
    with open('./odi_diou/{}_box_objs.txt'.format(method_tag), 'w') as f:
        for k in range(len(ious)):
            f.write('{} {}\n'.format(port_box_div_obj[k], ious[k].item()))


    with open('./odi_diou/{}_mask_objs.txt'.format(method_tag), 'w') as f:
        for k in range(len(ious)):
            f.write('{} {}\n'.format(port_mask_div_obj[k], ious[k].item()))


def get_list(maps_path):
    dirs = list(maps_path.values())
    files = [os.listdir(d) for d in dirs]

    objs = {}
    flag = True
    for f in files[0]:
        if os.path.splitext(f)[1] == '.npy':
            for file in files[1:]:
                if f not in file:
                    flag = False
            if flag:            
                img_id, gt_id, end = f.split('_')
                img_id, gt_id = int(img_id), int(gt_id)

                if img_id in objs.keys():
                    objs[img_id].append(int(gt_id))
                else:
                    objs[img_id] = []
                    objs[img_id].append(int(gt_id))
        flag = True
    return objs

def run_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', '-md', default=None, required=True, type=str)
    parser.add_argument('--test_method', '-t', default=None, required=True, type=str)
    parser.add_argument('--devices', '-d', default='0', type=str)
    os.environ['NCCL_IB_DISABLE'] = '1'
    args = parser.parse_args()

    drise_path = '/qnap/home_archive/chenyzhao9/exp/visual/drise'
    odam_path = '/qnap/home_archive/chenyzhao9/exp/visual/odam'
    gradcam_path = '/qnap/home_archive/chenyzhao9/exp/visual/gradcam'
    gradcamplus_path = '/qnap/home_archive/chenyzhao9/exp/visual/gradcam++'
    

    maps_path = {
            'odam': odam_path,
            'drise': drise_path,
            # 'gradcam': gradcam_path,
            # 'gradcam++':gradcamplus_path
        }

    objs_list = get_list(maps_path)


    # import libs
    model_root_dir = os.path.join('../model/', args.model_dir)
    sys.path.insert(0, model_root_dir)
    from config_coco import config


    eval_all(config, objs_list, maps_path[args.test_method])
if __name__ == '__main__':
    run_test()

