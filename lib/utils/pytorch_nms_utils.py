import torch
from typing import List

def pairwise_iou(boxes1, boxes2):
    """
    Given two lists of boxes of size N and M, compute the IoU
    (intersection over union) between **all** N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoU, sized [N,M].
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1]) # [N]
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1]) # [M]

    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(
        boxes1[:, None, :2], boxes2[:, :2]
    )  # [N,M,2]
    width_height.clamp_(min=0)  # [N,M,2]
    inter = width_height.prod(dim=2)  # [N,M]

    # handle empty boxes
    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou


def nms(boxlist, nms_th, nms_type='normal_nms', high_thr=0.8, low_thr=0.2):
    boxes = boxlist[:,:4]
    scores = boxlist[:,4]
    labels = boxlist[:,5]
    if nms_type in ['odam_nms','feature_nms'] :
        features = boxlist[:,6:]
    result_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    for id in torch.jit.annotate(List[int], torch.unique(labels).cpu().tolist()):
        mask = (labels == id).nonzero(as_tuple=False).view(-1)       
        if nms_type == 'normal_nms':
            keep = normal_nms(boxes[mask], scores[mask], nms_th)
        elif nms_type == 'soft_nms':
            keep = soft_nms(boxes[mask], scores[mask], sigma=0.5, thresh=0.1)
        elif nms_type == 'odam_nms':
            keep = odam_nms(boxes[mask], scores[mask], 
                features[mask], nms_th, high_thr, low_thr)
        elif nms_type == 'feature_nms': 
            keep = feature_nms(boxes[mask], scores[mask], 
                features[mask], high_thr, low_thr)
        else:
            raise ValueError('Unknown NMS method.')
        result_mask[mask[keep]] = True
    keep = result_mask.nonzero(as_tuple=False).view(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    boxlist = boxlist[keep]
    return boxlist

def normal_nms(boxes, scores, nms_th):
    _, order = scores.sort(0, descending=True)   
    
    keep = []
    while order.numel() > 0: 
        if order.numel() == 1:    
            i = order.item()
            keep.append(i)
            break  
        else:
            i = order[0].item()   
            keep.append(i)
        pair_ious = pairwise_iou(boxes[i].unsqueeze(0), boxes[order[1:]])[0]

        reserve_cond = (pair_ious <= nms_th)
        idx = reserve_cond.nonzero(as_tuple=False).squeeze()
        if idx.numel() == 0:
            break
        order = order[idx+1]
    return torch.LongTensor(keep)
        

def soft_nms(boxes, scores, sigma=0.5, thresh=0.05):
    N = scores.shape[0]
    indexes = torch.arange(0,N).view(N,1).to(boxes.device)
    for i in range(N):
        tmp_box = boxes[i].clone()
        tmp_score = scores[i].clone()
        tmp_index = indexes[i].clone()
        pos = i+1
        if i != N-1:
            max_score, max_pos = torch.max(scores[pos:], dim=0)
            if scores[i] < max_score:
                boxes[i] = boxes[max_pos.item()+i+1].clone()
                boxes[max_pos.item()+i+1] = tmp_box
                scores[i] = scores[max_pos.item()+i+1].clone()
                scores[max_pos.item()+i+1] = tmp_score
                indexes[i] = indexes[max_pos.item()+i+1].clone()
                indexes[max_pos.item()+i+1] = tmp_index

            # Ious between ith and boxes after ith
            pair_ious = pairwise_iou(boxes[i].unsqueeze(0), boxes[pos:])[0]

            # Gaussian decay
            weight = torch.exp(-(pair_ious*pair_ious)/sigma)
            scores[pos:] = weight * scores[pos:]
#     select the boxes with score over the thresh
    keep = indexes[scores > thresh].long().cpu()
    return keep

def odam_nms(boxes, scores, features, nms_th, high_thr=0.8, low_thr=0.2):
    features = features.reshape(features.shape[0],-1)
    _, order = scores.sort(0, descending=True)   
    
    keep = []
    while order.numel() > 0: 
        if order.numel() == 1:    
            i = order.item()
            keep.append(i)
            break  
        else:
            i = order[0].item()   
            keep.append(i)
        pair_ious = pairwise_iou(boxes[i].unsqueeze(0), boxes[order[1:]])[0]
        pair_corr = (features[i].unsqueeze(0) * features[order[1:]]).sum(-1)
        
        cond1 = (pair_ious <= nms_th) * (pair_corr < high_thr)
        cond2 = (pair_ious > nms_th) * (pair_corr < low_thr)
        idx = (cond1 + cond2).nonzero(as_tuple=False).squeeze()
        if idx.numel() == 0:
            break
        order = order[idx+1]
    return torch.LongTensor(keep)

def feature_nms(boxes, scores, features, high_thr=0.9, low_thr=0.1):
    features = features.reshape(features.shape[0],-1)
    _, order = scores.sort(0, descending=True)   
    
    keep = []
    while order.numel() > 0: 
        if order.numel() == 1:    
            i = order.item()
            keep.append(i)
            break  
        else:
            i = order[0].item()   
            keep.append(i)
        pair_ious = pairwise_iou(boxes[i].unsqueeze(0), boxes[order[1:]])[0]
        dists = torch.linalg.norm(
            features[i].unsqueeze(0)-features[order[1:]],
            ord=2,
            dim=1)
        
        cond1 = pair_ious < low_thr
        cond2 = (pair_ious < high_thr) * (dists > 1.0)
        idx = (cond1 + cond2).nonzero(as_tuple=False).squeeze()
        if idx.numel() == 0:
            break
        order = order[idx+1]
    return torch.LongTensor(keep)
