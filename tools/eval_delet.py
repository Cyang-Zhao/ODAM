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
from utils import misc_utils, pytorch_nms_utils
from det_oprs import bbox_opr

import cv2


eps = torch.finfo(torch.float32).eps

def eval_all(args, config, network, objs_list, map_path):
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
    inference(config, network, model_file, coco, 0, len_dataset, objs_list, map_path)


def get_map(img_id, gt_id, map_path):
    heatmap = np.load(os.path.join(map_path, '{}_{}_npy.npy'.format(img_id, gt_id)))
    return heatmap

def random_pixel(image, poses):
    # adjust the size of perturbation each step based on the size of object size 
    _,_,h,w = image.shape 
    random_patch = torch.rand(3, len(poses)) * 255.
    xs, ys = zip(*poses)
    image[0,:, ys, xs] = random_patch
    return image


def delet_process(net, image, im_info, heatmap, L, grids, gt_box, gt_class, img_id, gt_id, path):
    order = np.argsort(-heatmap.reshape(-1))

    scale_y, scale_x = im_info[0, 2:4]
    map_size_gt = gt_box.clone()
    map_size_gt[0:4:2] *= scale_x
    map_size_gt[1:4:2] *= scale_y
    area = (map_size_gt[2]-map_size_gt[0]) * (map_size_gt[3]-map_size_gt[1])
    pixel_once = max(1, int(area/100))

    # size = ((map_size_gt[2:]-map_size_gt[:2])**2).sum().sqrt()
    # size_r = max(int(size/10), 1) 

    # order = order[::size_r]
    scores = []
    for step in range(1,L+1):
        image = random_pixel(image, grids[order[(step-1)*pixel_once:step*pixel_once]])
        if step%10 == 0:
            pred_boxes, _, _ = \
                net(image.clone().cuda(), im_info.clone().cuda())
            pred_boxes = pred_boxes[:, :6].detach().cpu()
            pred_boxes[:, 0:4:2] /= scale_x
            pred_boxes[:, 1:4:2] /= scale_y
            
            overlaps = bbox_opr.box_overlap_opr(pred_boxes[:,:4], gt_box.unsqueeze(0)) # M, 1
            class_mask = pred_boxes[:,5] == gt_class
            pred_scores = pred_boxes[:,4] 
            if len(pred_scores) > 0:
                max_score = torch.max(class_mask * pred_scores * (overlaps[:,0]>0.5))
            else:
                max_score = 0.
            scores.append(max_score)
    image = image.squeeze(0).permute(1,2,0).numpy().astype(np.uint8)
    # image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR) 
    cv2.imwrite(path+'{}_{}.jpg'.format(img_id, gt_id), image)
    return scores



def make_grids(h, w):
    shifts_x = torch.arange(
        0, w, 1)
    shifts_y = torch.arange(
        0, h, 1)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    grids = torch.stack((shift_x, shift_y), dim=1)
    return grids

def inference(config, network, model_file, dataset, start, end, objs_list, map_path):
    torch.set_default_tensor_type('torch.FloatTensor')
    # init model
    net = network(config)
    net.cuda()
    net = net.eval()
    check_point = torch.load(model_file)
    net.load_state_dict(check_point['state_dict'], strict=False)
    # init data
    dataset.ids = dataset.ids[start:end];
    data_iter = torch.utils.data.DataLoader(dataset=dataset, shuffle=False)
        	
	# inference
    coco_id2id = dataset.coco_id2id

    # if save image after deletion
    method_tag = map_path.split('/')[-1]
    path = './delet_visual/{}/'.format(method_tag)
    if not os.path.exists(path):
        os.makedirs(path)

    L = 100
    score_cums = []
    pbar = tqdm(total=end-start, ncols=50)
    for (image, im_info, ID, anno, image_ori) in data_iter:
        if int(ID) in objs_list.keys():
            image_ori = image_ori[0].numpy()
            image_ori = cv2.cvtColor(image_ori,cv2.COLOR_RGB2BGR) 
            _, _, H, W = image.shape
            grids = make_grids(H,W)

            ori_imgsize = im_info[0, -2:]
            h,w = int(ori_imgsize[0]), int(ori_imgsize[1])

            # get gts
            anno = [obj for obj in anno if obj["iscrowd"] == 0]
            gt_boxes = [obj["bbox"] for obj in anno]
            gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)
            gt_boxes[:,2:4] += gt_boxes[:,0:2]
            gt_classes = [coco_id2id[int(obj["category_id"])] for obj in anno]        
            gt_classes = torch.tensor(gt_classes) 

            # pred_boxes: x1,y1,x2,y2,score,tag 
            # cls_outputs: outputs of detector cls branch
            # (loc, label): score location on the output tensor
            # levels: boxes are predicted from which level
            pred_boxes, _, _ = \
                net(image.clone().cuda(), im_info.clone().cuda())
            scale_y, scale_x = im_info[0, 2:4]
            pred_boxes = pred_boxes[:, :6].detach().cpu()
            pred_boxes[:, 0:4:2] /= scale_x
            pred_boxes[:, 1:4:2] /= scale_y

            overlaps = bbox_opr.box_overlap_opr(pred_boxes[:,:4], gt_boxes)
            class_mask = pred_boxes[:,5].unsqueeze(1) == gt_classes.unsqueeze(0)   
            pred_scores = pred_boxes[:,4]   
            scores_0 = pred_scores.unsqueeze(1) * class_mask * (overlaps>0.5)

            objects = objs_list[int(ID)]
            for gt_id in objects:
                score_0 = torch.max(scores_0[:,gt_id]) 
                heatmap = get_map(int(ID), gt_id, map_path) 
                if heatmap is None:
                    continue
                if (heatmap.shape[0] != h) or (heatmap.shape[1] != w):
                    continue        

                heatmap = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_CUBIC)
                scores_1toL = delet_process(net, image.clone(), im_info.clone(), \
                    heatmap, L, grids, gt_boxes[gt_id], gt_classes[gt_id], int(ID), gt_id, path)
                
                delet_L = torch.tensor([score_0]+scores_1toL) / score_0
                score_cums.append(delet_L)
        
                delet_score_tmp = torch.mean(torch.stack(score_cums,dim=0), dim=0)   
        
                print(' ')
                print('delet_score_tmp:', delet_score_tmp)
        pbar.update(1)
    
    print('number object counted:', len(score_cums))
    delet_score = torch.mean(torch.stack(score_cums,dim=0), dim=0)   
    
    print('map path', map_path)
    print('delet_score:', delet_score)





def get_list(maps_path):
    dirs = list(maps_path.values())
    print(dirs)
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
    parser.add_argument('--resume_weights', '-r', default=None, required=True, type=str)
    parser.add_argument('--test_method', '-t', default=None, required=True, type=str)
    parser.add_argument('--devices', '-d', default='0', type=str)
    os.environ['NCCL_IB_DISABLE'] = '1'
    args = parser.parse_args()

    drise_path = './visualization/drise'
    odam_path = './visualization/odam'
    gradcam_path = './visualization/gradcam'
    gradcamplus_path = './visualization/gradcam++'

    maps_path = {
            'odam': odam_path,
            'drise': drise_path,
            # 'gradcam': gradcam_path,
            # 'gradcam++':gradcamplus_path,
        }

    objs_list = get_list(maps_path)


    # import libs
    model_root_dir = os.path.join('../model/', args.model_dir)
    print(model_root_dir)
    sys.path.insert(0, model_root_dir)
    from config_coco import config
    from network import Network

    eval_all(args, config, Network, objs_list, maps_path[args.test_method])

if __name__ == '__main__':
    run_test()

