import os
import sys

import numpy as np

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

root_dir = '../../'
add_path(os.path.join(root_dir))
add_path(os.path.join(root_dir, 'lib'))

class COCO:
    root_folder = '/opt/visal/home/chenyzhao9/data/coco'
    train_folder = os.path.join(root_folder, 'train2017')
    eval_folder = os.path.join(root_folder, 'val2017')
    test_folder = os.path.join(root_folder, 'testdev2017')
    train_source = os.path.join(root_folder,'annotations/instances_train2017.json')
    eval_source = os.path.join(root_folder,'annotations/instances_val2017.json')
    test_source = os.path.join(root_folder,'annotations/instances_testdev2017.json')

class Config:
    output_dir = 'coco_model'
    model_dir = os.path.join(output_dir, 'model_dump')
    eval_dir = os.path.join(output_dir, 'eval_dump')
    init_weights = '/opt/visal/home/chenyzhao9/data/model/resnet50_fbaug.pth'

    # ----------data config---------- #
    image_mean = np.array([103.530, 116.280, 123.675])
    image_std = np.array([57.375, 57.120, 58.395])
    to_bgr255 = True
    train_image_min_size = 800
    train_image_max_size = 1333
    train_image_min_size_range = (-1,-1) # -1 means disabled and it will use MIN_SIZE_TRAIN
    # If True, each batch should contain only images for which the aspect ratio
    # is compatible. This groups portrait images together, and landscape images
    # are not batched with portrait images.
    aspect_ratio_grouping = True
    size_divisible = 32
    num_workers = 2
    eval_resize = True
    eval_image_min_size = 800
    eval_image_max_size = 1333
    seed_dataprovider = 3
    train_source = COCO.train_source
    eval_source = COCO.eval_source
    test_source = COCO.test_source    
    train_folder = COCO.train_folder
    eval_folder = COCO.eval_folder
    test_folder = COCO.test_folder

    # ----------train config---------- #
    backbone_freeze_at = 2
    rpn_channel = 256
    
    train_batch_per_gpu = 2
    momentum = 0.9
    weight_decay = 1e-4
    base_lr = 1e-3 * 1.25

    warm_iter = 500
    max_epoch = 12
    lr_decay = [8, 11]
    nr_images_epoch = 118287
    log_dump_interval = 20

    # ----------test config---------- #
    test_nms = 0.5
    test_nms_method = 'normal_nms'
    detection_per_image = 100
    visulize_threshold = 0.3
    pred_cls_threshold = 0.01
    drise_output_format = False


    # ----------model config---------- #
    batch_filter_box_size = 0
    nr_box_dim = 5
    ignore_label = -1
    max_boxes_of_image = 100

    # ----------rois generator config---------- #
    anchor_base_size = 32
    anchor_base_scale = [1]
    anchor_aspect_ratios = [0.5, 1.0, 2.0]
    num_cell_anchors = len(anchor_aspect_ratios)
    anchor_within_border = False

    rpn_min_box_size = 2
    rpn_nms_threshold = 0.7
    train_prev_nms_top_n = 12000
    train_post_nms_top_n = 2000
    test_prev_nms_top_n = 6000
    test_post_nms_top_n = 1000

    # ----------binding&training config---------- #
    rpn_smooth_l1_beta = 1
    rcnn_smooth_l1_beta = 1

    num_sample_anchors = 256
    positive_anchor_ratio = 0.5
    rpn_positive_overlap = 0.7
    rpn_negative_overlap = 0.3
    rpn_bbox_normalize_targets = False

    num_rois = 512
    fg_ratio = 0.25
    fg_threshold = 0.5
    bg_threshold_high = 0.5
    bg_threshold_low = 0.0
    rcnn_bbox_normalize_targets = True
    bbox_normalize_means = np.array([0, 0, 0, 0])
    bbox_normalize_stds = np.array([0.1, 0.1, 0.2, 0.2])

config = Config()

