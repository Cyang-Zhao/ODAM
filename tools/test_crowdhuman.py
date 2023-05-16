import os
import sys
import math
import argparse

import numpy as np
from tqdm import tqdm
import torch
from torch.multiprocessing import Queue, Process

sys.path.insert(0, '../lib')
sys.path.insert(0, '../model')
from data.CrowdHuman import CrowdHuman
from utils import misc_utils, pytorch_nms_utils
from evaluate import compute_JI, compute_APMR

def eval_all(args, config, network):
    # model_path
    saveDir = os.path.join('../model', args.model_dir, config.model_dir)
    evalDir = os.path.join('../model', args.model_dir, config.eval_dir)
    misc_utils.ensure_dir(evalDir)
    model_file = os.path.join(saveDir, 
            'dump-{}.pth'.format(args.resume_weights))
    # model_file = os.path.join(saveDir, 
    #          'rcnn_emd_refine_mge.pth')
    print(model_file)
    assert os.path.exists(model_file)
    # get devices
    str_devices = args.devices
    devices = misc_utils.device_parser(str_devices)
    # load data
    crowdhuman = CrowdHuman(config, if_train=False)
    #crowdhuman.records = crowdhuman.records[:10]
    # multiprocessing
    num_devs = len(devices)
    len_dataset = len(crowdhuman)
    print(len_dataset)
    num_image = math.ceil(len_dataset / num_devs)
    result_queue = Queue(500)
    procs = []
    all_results = []
    for i in range(num_devs):
        start = i * num_image
        end = min(start + num_image, len_dataset)
        proc = Process(target=inference, args=(
                config, network, model_file, devices[i], crowdhuman, start, end, result_queue))
        proc.start()
        procs.append(proc)
    pbar = tqdm(total=len_dataset, ncols=50)
    for i in range(len_dataset):
        t = result_queue.get()
        all_results.append(t)
        pbar.update(1)
    pbar.close()
    for p in procs:
        p.join()
    fpath = os.path.join(evalDir, 'dump-{}.json'.format(args.resume_weights))
    misc_utils.save_json_lines(all_results, fpath)
    
    # evaluation
    eval_path = os.path.join(evalDir, 'eval-{}.json'.format(args.resume_weights))
    eval_fid = open(eval_path,'w')
    res_line, JI = compute_JI.evaluation_all(fpath, 'box')
    for line in res_line:
        eval_fid.write(line+'\n')
    AP, Recall, MR = compute_APMR.compute_APMR(fpath, config.eval_source, 'box')
    line = 'AP:{:.4f}, Recall:{:.4f}, MR:{:.4f}, JI:{:.4f}.'.format(AP, Recall, MR, JI)
    print(line)
    eval_fid.write(line+'\n')
    eval_fid.close()

def inference(config, network, model_file, device, dataset, start, end, result_queue):
    torch.set_default_tensor_type('torch.FloatTensor')
    torch.multiprocessing.set_sharing_strategy('file_system')
    # init model
    net = network(config)
    net.cuda(device)
    net = net.eval()
    check_point = torch.load(model_file)
    net.load_state_dict(check_point['state_dict'], strict=False)
    # init data
    dataset.records = dataset.records[start:end];
    data_iter = torch.utils.data.DataLoader(dataset=dataset, shuffle=False)
    nms_type = config.test_nms_method
    print('nms_type:', nms_type)
    # inference
    for (image, gt_boxes, im_info, ID, _) in data_iter:
        pred_boxes = net(image.cuda(device), im_info.cuda(device))
        scale = im_info[0, 2]
        if nms_type == 'normal_nms':
            pred_boxes = pred_boxes[:,:6]
            pred_boxes = pred_boxes.reshape(-1, 6)
            keep = pred_boxes[:, 4] > config.pred_cls_threshold
            pred_boxes = pred_boxes[keep]
            pred_boxes = pytorch_nms_utils.nms(
                pred_boxes, config.test_nms, nms_type='normal_nms')
        elif nms_type == 'soft_nms':
            pred_boxes = pred_boxes[:,:6]
            pred_boxes = pred_boxes.reshape(-1, 6)
            keep = pred_boxes[:, 4] > config.pred_cls_threshold
            pred_boxes = pred_boxes[keep]        
            pred_boxes = pytorch_nms_utils.nms(
                pred_boxes, config.test_nms, nms_type='soft_nms')
        elif nms_type == 'odam_nms':
            pred_boxes = pred_boxes[:,:-2]
            keep = pred_boxes[:, 4] > config.pred_cls_threshold
            pred_boxes = pred_boxes[keep]
            pred_boxes = pytorch_nms_utils.nms(
                pred_boxes, config.test_nms, nms_type='odam_nms',
                high_thr=config.highthr, low_thr=config.lowthr)
            pred_boxes = pred_boxes[:, :6]
        elif nms_type == 'feature_nms':
            keep = pred_boxes[:, 4] > config.pred_cls_threshold
            pred_boxes = pred_boxes[keep]
            pred_boxes = pytorch_nms_utils.nms(
                pred_boxes, config.test_nms, nms_type='feature_nms',
                high_thr=config.highthr, low_thr=config.lowthr)
            pred_boxes = pred_boxes[:, :6]
        else:
            # no nms
            pred_boxes = pred_boxes[:,:6].reshape(-1, 6)
            keep = pred_boxes[:, 4] > config.pred_cls_threshold
            pred_boxes = pred_boxes[keep]

        pred_boxes = pred_boxes.cpu().numpy()
        if pred_boxes.shape[0] > config.detection_per_image:
           order = np.argsort(-pred_boxes[:, 4])
           order = order[:config.detection_per_image]
           pred_boxes = pred_boxes[order]

        # recovery the scale
        pred_boxes[:, :4] /= scale
        pred_boxes[:, 2:4] -= pred_boxes[:, :2]
        gt_boxes = gt_boxes[0].numpy()
        gt_boxes[:, 2:4] -= gt_boxes[:, :2]
        result_dict = dict(ID=ID[0], height=int(im_info[0, -3]), width=int(im_info[0, -2]),
                dtboxes=boxes_dump(pred_boxes), gtboxes=boxes_dump(gt_boxes))
        result_queue.put_nowait(result_dict)

def boxes_dump(boxes):
    if boxes.shape[-1] == 7:
        result = [{'box':[round(i, 1) for i in box[:4]],
                   'score':round(float(box[4]), 5),
                   'tag':int(box[5]),
                   'proposal_num':int(box[6])} for box in boxes]
    elif boxes.shape[-1] == 6:
        result = [{'box':[round(i, 1) for i in box[:4].tolist()],
                   'score':round(float(box[4]), 5),
                   'tag':int(box[5])} for box in boxes]
    elif boxes.shape[-1] == 5:
        result = [{'box':[round(i, 1) for i in box[:4]],
                   'tag':int(box[4])} for box in boxes]
    else:
        raise ValueError('Unknown box dim.')
    return result

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
    from config_crowdhuman import config
    from network import Network
    eval_all(args, config, Network)

if __name__ == '__main__':
    run_test()

