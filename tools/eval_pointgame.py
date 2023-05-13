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
    inference(coco, 0, len_dataset, objs_list, map_path)


def get_map(img_id, gt_id, map_path):
    heatmap = np.load(os.path.join(map_path, '{}_{}_npy.npy'.format(img_id, gt_id)))
    return torch.tensor(heatmap)

def inference(dataset, start, end, objs_list, map_path):
    torch.set_default_tensor_type('torch.FloatTensor')

    # init data
    dataset.ids = dataset.ids[start:end]
    data_iter = torch.utils.data.DataLoader(dataset=dataset, shuffle=False)
        	
	# inference
    coco_id2id = dataset.coco_id2id

    num_tp = 0
    isinbox = 0
    isinmask = 0
    energy_box_prop = 0
    energy_mask_prop = 0
    compactness = 0

    pbar = tqdm(total=end-start, ncols=50)
    for (image, im_info, ID, anno, image_ori) in data_iter:
        if int(ID) in objs_list.keys():  
            image_ori = image_ori[0].numpy()
            image_ori = cv2.cvtColor(image_ori,cv2.COLOR_RGB2BGR)
            h,w = image_ori.shape[:2]

            # ori_imgsize = im_info[0, -2:]
            # h,w = int(ori_imgsize[0]), int(ori_imgsize[1])
            shifts_x, shifts_y = torch.arange(0,w,1.), torch.arange(0,h,1.)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            map_grids = torch.stack((shift_x.reshape(-1), shift_y.reshape(-1)), dim=1)

            # get gts
            anno = [obj for obj in anno if obj["iscrowd"] == 0]
            gt_boxes = [obj["bbox"] for obj in anno]
            gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)
            gt_boxes[:,2:4] += gt_boxes[:,0:2]
            gt_classes = [coco_id2id[int(obj["category_id"])] for obj in anno]        
            gt_classes = torch.tensor(gt_classes) 
            gt_masks = [obj["segmentation"] for obj in anno]


            objects = objs_list[int(ID)]
            for gt_ind in objects:
                heatmap = get_map(int(ID), gt_ind, map_path) 
                if heatmap is None:
                    continue
                if (heatmap.shape[0] != h) or (heatmap.shape[1] != w):
                    continue
                
                max_pos_y, max_pos_x = torch.nonzero(heatmap==1, as_tuple=True)
                
                if len(max_pos_x) > 0:
                    max_pos_y, max_pos_x = max_pos_y[0], max_pos_x[0] 
                    num_tp += 1
                    max_poi = [max_pos_x, max_pos_y]   
                    gt_box = gt_boxes[gt_ind]       
                    # whether in box?
                    if (max_pos_x > gt_box[0]) and (max_pos_x < gt_box[2]) and \
                        (max_pos_y > gt_box[1]) and (max_pos_y < gt_box[3]):
                        isinbox += 1
                        # whether in mask?
                        gt_poly = gt_masks[gt_ind]
                        if isPoiWithinPoly(max_poi, gt_poly):
                            isinmask += 1
                    # energy box
                    x1, y1 = int(max(0, gt_box[0].floor())), int(max(0, gt_box[1].floor()))
                    x2, y2 = int(min(w, gt_box[2].ceil())), int(min(h, gt_box[3].ceil()))
                    
                    gt_box_mask = heatmap.new_zeros(heatmap.shape)
                    gt_box_mask[y1:y2, x1:x2] = 1.
                    tmp_energy = (heatmap*gt_box_mask).sum() / (heatmap.sum()+1e-10)
                    energy_box_prop += tmp_energy

                    # energy mask
                    gt_poly = gt_masks[gt_ind]
                    # generate binary ground truth mask
                    gt_mask = np.zeros(image_ori.shape[:2])
                    for epoly in gt_poly:
                        tmp = np.zeros(image_ori.shape[:2])
                        epoly = np.array(epoly).reshape(-1,2).astype(np.int32)              
                        cv2.fillPoly(tmp, [epoly], 1.)
                        gt_mask += tmp
                    gt_mask = (gt_mask>0).astype(np.float32)
                    tmp_energy = (heatmap*gt_mask).sum() / (heatmap.sum()+1e-10)
                    energy_mask_prop += tmp_energy


                    # compactness
                    weighted_dis = heatmap.reshape(-1)*\
                        ((map_grids[:,0]-max_poi[0])**2 + (map_grids[:,1]-max_poi[1])**2)
                    box_size_norm = ((gt_box[2:] - gt_box[:2])**2).sum() / 4.
                    tmp_compact = torch.sqrt(weighted_dis.sum()/(box_size_norm*heatmap.sum()))
                    compactness += tmp_compact                 

                    print('Ntp:{}, inbox:{:.4f}, inmask:{:.4f}, energy box prop:{:.4f}, energy mask prop:{:.4f}, compactness:{:.4f}'.format(\
                        num_tp, isinbox/num_tp, isinmask/num_tp, energy_box_prop/num_tp, energy_mask_prop/num_tp, compactness/num_tp))

        pbar.update(1)
    
def isPoiWithinPoly(poi,poly):
    # poi = [x,y]
    # poly=[[x1,y1,x2,y2,...,xn,yn,x1,y1],[w1,t1,...wk,tk]] 

    sinsc=0 # num of intersection
    for epoly in poly: 
        epoly = torch.tensor(epoly).reshape(-1,2)
        for i in range(len(epoly)): 
            s_poi=epoly[i]
            if i == len(epoly)-1:
                e_poi = epoly[0]
            else:
                e_poi = epoly[i+1]
            if isRayIntersectsSegment(poi,s_poi,e_poi):
                sinsc += 1 

    return True if sinsc%2==1 else  False

def isRayIntersectsSegment(poi,s_poi,e_poi): #[x,y] start_point end_point
    if s_poi[1]==e_poi[1]: 
        return False
    if s_poi[1]>poi[1] and e_poi[1]>poi[1]: 
        return False
    if s_poi[1]<poi[1] and e_poi[1]<poi[1]: 
        return False
    if s_poi[1]==poi[1] and e_poi[1]>poi[1]: 
        return False
    if e_poi[1]==poi[1] and s_poi[1]>poi[1]: 
        return False
    if s_poi[0]<poi[0] and e_poi[1]<poi[1]: 
        return False

    xseg=e_poi[0]-(e_poi[0]-s_poi[0])*(e_poi[1]-poi[1])/(e_poi[1]-s_poi[1]) 
    if xseg<poi[0]: 
        return False
    return True 


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
    
    N = 0
    for img in objs:
        N += len(objs[img])
    print('number of imgs:', len(objs.keys()))
    print('number of objects:', N)
        

    return objs


def run_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', '-md', default=None, required=True, type=str)
    parser.add_argument('--test_method', '-t', default=None, required=True, type=str)
    os.environ['NCCL_IB_DISABLE'] = '1'
    args = parser.parse_args()

    drise_path = './visualization/drise'
    odam_path = './visualization/odam'
    gradcam_path = './visualization/gradcam'
    gradcamplus_path = './visualization/gradcam++'
    

    maps_path = {
            'drise': drise_path,
            'odam': odam_path,         
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

