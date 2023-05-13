import os
import sys
import math
import argparse
import json

import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

sys.path.insert(0, '../lib')
sys.path.insert(0, '../model')
from data.Coco import COCODataset

import cv2


eps = torch.finfo(torch.float32).eps

def eval_all(config, objs_list, map_path):
	# load data
    img_folder = config.eval_folder
    source = config.eval_source
    coco = COCODataset(config, img_folder, source, is_train=False)      	
    print(coco.__len__())
    len_dataset = coco.__len__()	
    inference(config, coco, 0, len_dataset, objs_list, map_path)


def get_map(img_id, gt_id, map_path):
    heatmap = np.load(os.path.join(map_path, '{}_{}_npy.npy'.format(img_id, gt_id)))
    return heatmap


def inference(config, dataset, start, end, objs_list, map_path):
    torch.set_default_tensor_type('torch.FloatTensor')
    # init data
    dataset.ids = dataset.ids[start:end]
    data_iter = torch.utils.data.DataLoader(dataset=dataset, shuffle=False)
        	
	# inference
    coco_id2id = config.coco_id2id

    threshold = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    iou = []

    pbar = tqdm(total=end-start, ncols=50)
    for (image, im_info, ID, anno, image_ori) in data_iter:
        if int(ID) in objs_list.keys():
            image_ori = image_ori[0].numpy()
            image_ori = cv2.cvtColor(image_ori,cv2.COLOR_RGB2BGR) 
            h,w,_ = image_ori.shape
            # get gts
            anno = [obj for obj in anno if obj["iscrowd"] == 0]
            gt_boxes = [obj["bbox"] for obj in anno]
            gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)
            gt_boxes[:,2:4] += gt_boxes[:,0:2]
            gt_classes = [coco_id2id[int(obj["category_id"])] for obj in anno]        
            gt_classes = torch.tensor(gt_classes) 
            gt_masks = [obj["segmentation"] for obj in anno]

            objects = objs_list[int(ID)]

            for i, gt_id in enumerate(objects):
                heatmap = get_map(int(ID), gt_id, map_path)
                if heatmap is None:
                    continue
                if (heatmap.shape[0] != h) or (heatmap.shape[1] != w):
                    continue 
  
                heatmap = get_map(int(ID), gt_id, map_path)  
                
                # heatmap = phi[i].cpu().numpy()
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
                gt_poly = gt_masks[gt_id]
                # generate binary ground truth mask
                gt_mask = np.zeros(image_ori.shape[:2])
                for epoly in gt_poly:
                    tmp = np.zeros(image_ori.shape[:2])
                    epoly = np.array(epoly).reshape(-1,2).astype(np.int32)              
                    cv2.fillPoly(tmp, [epoly], 1.)
                    gt_mask += tmp
                gt_mask = (gt_mask>0).astype(np.float32)

                iou_tmp = []
                
                for thr in threshold:
                    tmp = np.where(heatmap>thr, 1., 0.)
                    iou_tmp.append((gt_mask*tmp).sum() / ((gt_mask+tmp)>0).sum())
                iou.append(iou_tmp)
        
        pbar.update(1)
    
    iou = np.array(iou)
    iou = np.mean(iou, axis=0)
    print('map path', map_path)
    print('iou:', iou)


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
            # 'gradcam++':gradcamplus_path,
        }

    objs_list = get_list(maps_path)

    # import libs
    model_root_dir = os.path.join('../model/', args.model_dir)
    sys.path.insert(0, model_root_dir)
    from config_coco import config

    eval_all(config, objs_list, maps_path[args.test_method])

if __name__ == '__main__':
    run_test()

