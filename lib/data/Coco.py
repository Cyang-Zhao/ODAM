import os
import cv2
import torch
import torchvision
import numpy as np

from pycocotools.coco import COCO
from utils import misc_utils
from data.structures import BoxList
from data.transforms import build_transforms

def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image=10
    if _count_visible_keypoints(anno) >= 10:
        return True
    return False


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(self, config, img_folder, source, is_train):
        super(COCODataset, self).__init__(img_folder, source)
        self.training = is_train
        self.ids = sorted(self.ids)
        if is_train:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids  

        cat_ids = sorted(self.coco.getCatIds())   # 1~90
        cats = self.coco.loadCats(cat_ids)
        class_names = [c["name"] for c in sorted(cats, key=lambda x: x["id"])] # names  
        cat_ids.insert(0, 0)
        class_names.insert(0, 'background')  # add 0: background

        config.class_names = class_names
        config.num_classes = len(cat_ids)
        config.id2coco_id = {i: v for i, v in enumerate(cat_ids)}
        config.coco_id2id = {v: i for i, v in enumerate(cat_ids)} 
        self.coco_id2id = config.coco_id2id 
        self.nid_to_img_id = {k: v for k, v in enumerate(self.ids)}  
        self.img_id_to_nid = {v: k for k, v in enumerate(self.ids)} 

        # self.records = misc_utils.load_coco_lines(self.coco, self.training)
        self.config = config
        self._transforms = build_transforms(config, self.training)

    def __getitem__(self, index):
        return self.load_record(index)

    def __getitem_from_ID__(self, ID):
        index = self.img_id_to_nid[ID]
        return self.load_record(index)

    def __len__(self):
        return len(self.ids)

    def load_record(self, index):
        image, anno = super(COCODataset, self).__getitem__(index)
        # image
        # image_path = os.path.join(self.img_folder, record[0]['file_name'])        
        # image = misc_utils.load_img(image_path)        
        image_w, image_h = image.size

        if self.training:
            # ground_truth
            anno = [obj for obj in anno if obj["iscrowd"] == 0]
            boxes = [obj["bbox"] for obj in anno]
            target = BoxList(boxes, image.size, mode="xywh").convert("xyxy")
            classes = [obj["category_id"] for obj in anno]
            classes = [self.coco_id2id[c] for c in classes]
            classes = torch.tensor(classes) 
            target.add_field("labels", classes)
            target = target.clip_to_image(remove_empty=True)
            image, target = self._transforms(image, target)
            gtboxes = target.bbox
            classes = target.get_field("labels").unsqueeze(1)
            gtboxes = torch.cat([gtboxes, classes], dim=1)
                    
            # im_info
            nr_gtboxes = gtboxes.shape[0]
            t_height, t_width = image.shape[1:]
            scale_y, scale_x = t_height/image_h, t_width/image_w
            im_info = torch.tensor([0, 0, scale_y, scale_x, image_h, image_w, nr_gtboxes])
            return image, gtboxes, im_info
        else:
            # image
            image_ori = np.array(image.copy())
            image, _ = self._transforms(image, target=None)  
            t_height, t_width = image.shape[1:]         
            scale_y, scale_x = t_height/image_h, t_width/image_w
            im_info = torch.tensor([t_height, t_width, scale_y, scale_x, image_h, image_w])
            return image, im_info, self.nid_to_img_id[index], anno, image_ori

    def get_img_info(self, index):
        img_id = self.nid_to_img_id[index]
        img_data = self.coco.imgs[img_id]
        return img_data

    def collator(self, batch):
        transposed_batch = list(zip(*batch))
        images = transposed_batch[0]
        gtboxes = transposed_batch[1]
        im_info = transposed_batch[2]

        size_divisible = self.config.size_divisible
        max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))

        # TODO Ideally, just remove this and let me model handle arbitrary
        # input sizs
        if size_divisible > 0:
            import math

            stride = size_divisible
            max_size = list(max_size)
            max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
            max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
            max_size = tuple(max_size)

        batch_shape = (len(images),) + max_size
        batched_imgs = torch.ones(*batch_shape).type_as(images[0])
        # batched_imgs = images[0].new(*batch_shape).zero_()
        batched_imgs = batched_imgs * \
            torch.tensor(self.config.image_mean[None, :, None, None]).type_as(batched_imgs)
        for img, pad_img in zip(images, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        ground_truth = []
        for it in gtboxes:
            gt_padded = np.zeros((self.config.max_boxes_of_image, self.config.nr_box_dim))
            max_box = min(self.config.max_boxes_of_image, len(it))
            gt_padded[:max_box] = it[:max_box]
            ground_truth.append(gt_padded)
        ground_truth = torch.tensor(ground_truth).float()
        im_info = torch.stack(im_info, dim=0)
        im_info[:, 0] = batched_imgs.shape[2]
        im_info[:, 1] = batched_imgs.shape[3]

        return batched_imgs, ground_truth, im_info

#     def merge_batch(self, data):
#         # image
#         images = [it[0].numpy() for it in data]
#         gt_boxes = [it[1] for it in data]
#         im_info = np.array([it[2] for it in data])
#         batch_height = np.max(im_info[:, 3])
#         batch_width = np.max(im_info[:, 4])
#         padded_images = [pad_image(
#                 im, batch_height, batch_width, self.config.image_mean) for im in images]
        
#         t_height, t_width, scale = target_size(
#                 batch_height, batch_width, self.short_size, self.max_size)
#         # INTER_CUBIC, INTER_LINEAR, INTER_NEAREST, INTER_AREA, INTER_LANCZOS4
#         resized_images = np.array([cv2.resize(
#                 im, (t_width, t_height), interpolation=cv2.INTER_LINEAR) for im in padded_images])
#         resized_images = resized_images.transpose(0, 3, 1, 2)
#         images = torch.tensor(resized_images).float()
#         # ground_truth
#         ground_truth = []
#         for it in gt_boxes:
#             gt_padded = np.zeros((self.config.max_boxes_of_image, self.config.nr_box_dim))
#             it[:, 0:4] *= scale
#             max_box = min(self.config.max_boxes_of_image, len(it))
#             gt_padded[:max_box] = it[:max_box]
#             ground_truth.append(gt_padded)
#         ground_truth = torch.tensor(ground_truth).float()
#         # im_info
#         im_info[:, 0] = t_height
#         im_info[:, 1] = t_width
#         im_info[:, 2] = scale
#         im_info = torch.tensor(im_info)
#         if max(im_info[:, -1] < 2):
#             return None, None, None
#         else:
#             return images, ground_truth, im_info

# def pad_image(img, height, width, mean_value):
#     o_h, o_w, _ = img.shape
#     margins = np.zeros(2, np.int32)
#     assert o_h <= height
#     margins[0] = height - o_h
#     img = cv2.copyMakeBorder(
#         img, 0, margins[0], 0, 0, cv2.BORDER_CONSTANT, value=0)
#     img[o_h:, :, :] = mean_value
#     assert o_w <= width
#     margins[1] = width - o_w
#     img = cv2.copyMakeBorder(
#         img, 0, 0, 0, margins[1], cv2.BORDER_CONSTANT, value=0)
#     img[:, o_w:, :] = mean_value
#     return img



# def target_size(height, width, short_size, max_size):
#     im_size_min = np.min([height, width])
#     im_size_max = np.max([height, width])
#     scale = (short_size + 0.0) / im_size_min
#     if scale * im_size_max > max_size:
#         scale = (max_size + 0.0) / im_size_max
#     t_height, t_width = int(round(height * scale)), int(
#         round(width * scale))
#     return t_height, t_width, scale

# def flip_boxes(boxes, im_w):
#     flip_boxes = boxes.copy()
#     for i in range(flip_boxes.shape[0]):
#         flip_boxes[i, 0] = im_w - boxes[i, 2] - 1
#         flip_boxes[i, 2] = im_w - boxes[i, 0] - 1
#     return flip_boxes


