import os
import sys

import numpy as np

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

root_dir = '../../'
add_path(os.path.join(root_dir))
add_path(os.path.join(root_dir, 'lib'))

class Crowd_human:
    class_names = ['background', 'person']
    num_classes = len(class_names)
    root_folder = '../data/crowdhuman'
    image_folder = os.path.join(root_folder, 'images')
    train_source = os.path.join(root_folder,'annotations/annotation_train.odgt')
    eval_source = os.path.join(root_folder,'annotations/annotation_val.odgt')

class Config:
    output_dir = 'outputs'
    model_dir = output_dir
    # model_dir = os.path.join(output_dir, 'model_dump')
    eval_dir = os.path.join(output_dir, 'eval_dump')
    init_weights = '../data/model/resnet50_fbaug.pth'

    # ----------data config---------- #
    image_mean = np.array([103.530, 116.280, 123.675])
    image_std = np.array([57.375, 57.120, 58.395])
    train_image_short_size = 800
    train_image_max_size = 1400
    eval_resize = True
    eval_image_short_size = 800
    eval_image_max_size = 1400
    seed_dataprovider = 3
    train_source = Crowd_human.train_source
    eval_source = Crowd_human.eval_source
    image_folder = Crowd_human.image_folder
    class_names = Crowd_human.class_names
    num_classes = Crowd_human.num_classes
    class_names2id = dict(list(zip(class_names, list(range(num_classes)))))
    gt_boxes_name = 'fbox'

    # ----------train config---------- #
    backbone_freeze_at = 2
    train_batch_per_gpu = 2
    momentum = 0.9
    weight_decay = 1e-4
    base_lr = 6.25e-4
    focal_loss_alpha = 0.25
    focal_loss_gamma = 2

    warm_iter = 800
    max_epoch = 30
    lr_decay = [24, 27]
    # max_epoch = 50
    # lr_decay = [33, 43]
    nr_images_epoch = 15000
    log_dump_interval = 20

    # ----------test config---------- #
    test_nms = 0.5
    # test_nms_method = 'normal_nms'
    test_nms_method = 'odam_nms'
    # test_nms_method = 'soft_nms'
    visulize_threshold = 0.3
    pred_cls_threshold = 0.05
    highthr = 0.8
    lowthr = 0.2
    detection_per_image = 500

    # ----------dataset config---------- #
    nr_box_dim = 5
    max_boxes_of_image = 500

    # --------fcos config-------- #
    center_radius = 1.5
    level_assign = False
    size_of_interest = [[0, 64], [64, 128], [128, 256], [256, 512], [512, 10000000]]
    quality_assign = True
    quality_alpha = 0.8
    quality_dynk = 5

    reg_loss_type = 'giou'
    box_quality_type = 'iou'
    pre_nms_topk = 500

    # --------camfcos config-------- #
    grad_recept_radius = 5
    max_train_cam_sample = 300


    # ----------binding&training config---------- #
    smooth_l1_beta = 0.1
    negative_thresh = 0.4
    positive_thresh = 0.5
    allow_low_quality = True

config = Config()
