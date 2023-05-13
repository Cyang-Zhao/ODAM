import torch
from det_oprs.bbox_opr import box_overlap_opr


INF = 100000000

def compute_grids(features, strides):
    grids = []
    for level, feature in enumerate(features):
        h, w = feature.size()[-2:]
        shifts_x = torch.arange(
            0, w * strides[level], 
            step=strides[level],
            dtype=torch.float32, device=feature.device)
        shifts_y = torch.arange(
            0, h * strides[level], 
            step=strides[level],
            dtype=torch.float32, device=feature.device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        grids_per_level = torch.stack((shift_x, shift_y), dim=1) + \
            strides[level] // 2
        grids.append(grids_per_level)
    return grids

def flatten_outputs(raw_pred):
    # Reshape: (B, F, Hl, Wl) -> (B, Hl, Wl, F) -> (B, sum_l N*Hl*Wl, F)
    B = raw_pred[0].shape[0]
    flatten_pred = torch.cat([x.permute(0,2,3,1).reshape(B, -1, raw_pred[0].shape[1]) \
        for x in raw_pred], dim=1)
    return flatten_pred

def fcos_target_in_r(\
    config, grids, strides, size_ranges, gt_boxes, R=1.5, cls_prob=None, reg_pred=None):
    boxes = gt_boxes[gt_boxes[:,4]>0]  # remove background
    M = grids.shape[0]
    N = boxes.shape[0]
    valid_obj = grids.new_tensor([])

    if N > 0:
        l = grids[:, 0].view(M, 1) - boxes[:, 0].view(1, N) # M x N
        t = grids[:, 1].view(M, 1) - boxes[:, 1].view(1, N) # M x N
        r = boxes[:, 2].view(1, N) - grids[:, 0].view(M, 1) # M x N
        b = boxes[:, 3].view(1, N) - grids[:, 1].view(M, 1) # M x N
        reg_target = torch.stack([l, t, r, b], dim=2) # M x N x 4                               
        is_in_boxes = (reg_target.min(dim=2)[0] > 0) # M x N   
        centers = ((boxes[:, [0, 1]] + boxes[:, [2, 3]]) / 2) # N x 2
        is_center_in_r = get_center_in_r(
                        grids, centers, strides, R)  # M x N
        is_in_pos = is_in_boxes * is_center_in_r # M x N
        # filter too small objects
        valid_obj = is_in_pos.sum(0) > 0
    N = valid_obj.sum()
    if N == 0:
        return None, None, None, None
    else:
        valid_box = boxes[valid_obj]
        areas = (valid_box[:,2:4] - valid_box[:,:2]).prod(dim=-1)
        is_in_pos = is_in_pos[:,valid_obj]
        reg_target = reg_target[:,valid_obj,:]
        all_pred_boxes = None
   
        if config.level_assign:
            # assign objects to levels
            reg_target = reg_target * is_in_pos.view(M,N,1) + \
                INF * (1-is_in_pos.view(M,N,1).float())
            is_cared_in_level = assign_reg_fpn(
                    reg_target, size_ranges, areas.view(1,N).expand(M,N)) 
            is_in_pos = is_in_pos * is_cared_in_level 
            return valid_box, is_in_pos, reg_target, None
        if (cls_prob is not None) and (reg_pred is not None):
            gt_tags = valid_box[:,4].long() - 1
            cls_prob = cls_prob.sigmoid()[:, gt_tags]  
            all_pred_boxes = get_all_predicted_box(grids, reg_pred, strides)
            iou = box_overlap_opr(all_pred_boxes, valid_box[:,:4]) # M x N
            quality = cls_prob ** (1 - config.quality_alpha) * iou ** config.quality_alpha
            quality[~is_in_pos] = -1
            # decide the number of positive
            top_dynk = iou.topk(config.quality_dynk, dim=0)[0].sum(0).abs().clamp(min=1.) 
            is_in_pos = assign_with_quality(quality, top_dynk)
            return valid_box, is_in_pos, reg_target, all_pred_boxes

def get_all_predicted_box(grids, reg_pred, strides):
    reg_pred = reg_pred * strides.view(-1,1)    
    pred_boxes = torch.stack([
            grids[:,0] - reg_pred[:,0],
            grids[:,1] - reg_pred[:,1],
            grids[:,0] + reg_pred[:,2],
            grids[:,1] + reg_pred[:,3]
        ], dim=1)  # M x 4  
    return pred_boxes  

def get_center_in_r(locations, centers, strides, R):
    M, N = locations.shape[0], centers.shape[0]
    locations_expanded = locations.view(M, 1, 2).expand(M, N, 2) # M x N x 2
    centers_expanded = centers.view(1, N, 2).expand(M, N, 2) # M x N x 2
    strides_expanded = strides.view(M, 1, 1).expand(M, N, 2) # M x N
    centers_discret = ((centers_expanded / strides_expanded).int() * \
        strides_expanded).float() + strides_expanded / 2 # M x N x 2
    dist_x = (locations_expanded[:, :, 0] - centers_discret[:, :, 0]).abs()
    dist_y = (locations_expanded[:, :, 1] - centers_discret[:, :, 1]).abs()
    return (dist_x <= strides_expanded[:, :, 0] * R) & \
        (dist_y <= strides_expanded[:, :, 0] * R)

def assign_reg_fpn(reg_targets_per_im, size_ranges, areas):
    '''
    Inputs:
        reg_targets_per_im: M x N x 4
        size_ranges: M x 2
        areas: M x N
    '''
    crit = ((reg_targets_per_im[:, :, :2] + \
        reg_targets_per_im[:, :, 2:])**2).sum(dim=2) ** 0.5 / 2 # M x N
    level_mask = (crit >= size_ranges[:, [0]]) & \
        (crit <= size_ranges[:, [1]])
    cared_areas = areas * level_mask + INF * (1-level_mask.float()) # M x N
    is_cared_in_the_level = level_mask.new_zeros(level_mask.shape)
    _, min_inds = cared_areas.min(dim=1)
    is_cared_in_the_level[range(level_mask.shape[0]), min_inds] = 1.
    return is_cared_in_the_level * level_mask

def assign_with_quality(quality, top_dynk):
    # assign the anchor to the object with max quality
    M, N = quality.shape 
    pos_qualities = quality.new_zeros(M,N)
    max_values, assign_objs = quality.max(1)
    pos_qualities[range(M), assign_objs] = max_values

    # pick top-dynk anchor as positives
    sorted_quality, _ = torch.sort(pos_qualities, dim=0, descending=True)
    thresh_for_each_obj = sorted_quality[top_dynk.ceil().long(), range(N)]
    pos_qualities = pos_qualities * (pos_qualities >= thresh_for_each_obj)
    return pos_qualities > 0

def make_offset_grids(size=11):
    shifts_x = torch.arange(0, size,1)
    shifts_y = torch.arange(0, size,1)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    offsets = torch.stack((shift_x, shift_y), dim=1)
    return offsets    