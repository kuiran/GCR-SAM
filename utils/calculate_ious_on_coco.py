# test pred bbox and mask iou on coco
import json
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import numpy as np
from mmdet.structures.bbox import bbox_overlaps
import torch
import argparse
import os


def bbox_cxcywh_to_xyxy(bbox):
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    cx, cy, w, h = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.cat(bbox_new, dim=-1)

def polygon_to_bitmap(polygons, height, width):
    """Convert masks from the form of polygons to bitmaps.

    Args:
        polygons (list[ndarray]): masks in polygon representation
        height (int): mask height
        width (int): mask width

    Return:
        ndarray: the converted masks in bitmap representation
    """
    rles = maskUtils.frPyObjects(polygons, height, width)
    rle = maskUtils.merge(rles)
    bitmap_mask = maskUtils.decode(rle)
    return bitmap_mask

def cal_mask_iou(masks1, masks2):
    mask_ious = []
    for mask1, mask2 in zip(masks1, masks2):
        area1 = mask1.sum()
        area2 = mask2.sum()
        inter = ((mask1+mask2)==2).sum()
        mask_iou = inter / (area1+area2-inter)
        mask_ious.append(mask_iou)
    return np.array(mask_ious)

def cal_iou(img_id, instance_bboxes_pred, instance_bboxes_gt, instance_masks_pred, instance_masks_gt, weight, height, with_mask):
    flag = False
    if img_id == 361919:
        print('stop!')
    if instance_bboxes_pred.size(0) != instance_bboxes_gt.size(0):
        instance_bboxes_gt = instance_bboxes_gt[:instance_bboxes_pred.size(0), :]
        flag = True
    # if flag:
    #     a = instance_bboxes_pred.size(0)
    #     b = instance_bboxes_gt.size(0)
    #     print(a, b, img_id)
    bbox_ious = bbox_overlaps(instance_bboxes_pred, instance_bboxes_gt, is_aligned=True)
    if with_mask:
        instance_masks_pred_bitmap = [maskUtils.decode(instance_mask_pred) for instance_mask_pred in instance_masks_pred]
        instance_masks_gt_bitmap = []
        for instance_mask_gt in instance_masks_gt:
            if type(instance_mask_gt) == list:
                instance_masks_gt_bitmap.append(polygon_to_bitmap(instance_mask_gt, height, weight))
            else:
                instance_masks_gt_bitmap.append(maskUtils.decode(instance_mask_gt))
        # instance_masks_gt_bitmap = [polygon_to_bitmap(instance_mask_gt, height, weight) for instance_mask_gt in instance_masks_gt]
        if len(instance_masks_pred) != len(instance_masks_gt):
            instance_masks_gt[:len(instance_masks_pred)]
            flag = True
        mask_ious = cal_mask_iou(instance_masks_pred_bitmap, instance_masks_gt_bitmap)
    else:
        mask_ious = None
    return bbox_ious, mask_ious, flag

def calculate_ious_on_coco(
        bbox_pred_json_path: str,
        mask_pred_json_path: str,
        save_path: str,
        gt_json_path: str,
):
    with open(bbox_pred_json_path) as f:
        bbox_pred_json = json.load(f)
    f.close()

    with_mask = False
    if os.path.exists(mask_pred_json_path):
        with_mask = True
        with open(mask_pred_json_path) as f:
            mask_pred_json = json.load(f)
        f.close()
    else:
        mask_pred_json = bbox_pred_json

    coco = COCO(gt_json_path)

    img_ids = set()
    for bp in bbox_pred_json:
        img_ids.add(bp['image_id'])

    img_ids = list(img_ids)

    instance_ious = {}
    for img_id in img_ids:
        if img_id == 361919:
            print("stop!")
        # print(f'image_id: {img_id}')
        instance_iou = {}
        instance_bboxes_pred = []
        instance_masks_pred = []
        for bbox_pred, mask_pred in zip(bbox_pred_json, mask_pred_json):
            if bbox_pred['image_id'] == img_id and mask_pred['image_id'] == img_id:
                instance_bboxes_pred.append(bbox_pred['bbox'])
                if with_mask:
                    instance_masks_pred.append(mask_pred['segmentation'])
        instance_bboxes_pred = bbox_cxcywh_to_xyxy(torch.tensor(instance_bboxes_pred))
        
        gt_ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=False)
        gt_anns = coco.loadAnns(gt_ann_ids)
        instance_bboxes_gt = []
        instance_masks_gt = []

        img_meta = coco.loadImgs(img_id)

        weight = img_meta[0]['width']
        height = img_meta[0]['height']

        for gt_ann in gt_anns:
            instance_bboxes_gt.append(gt_ann['true_bbox'])
            if with_mask:
                instance_masks_gt.append(gt_ann['segmentation'])

        instance_bboxes_gt = bbox_cxcywh_to_xyxy(torch.tensor(instance_bboxes_gt))

        box_ious, mask_ious, flag = cal_iou(img_id, instance_bboxes_pred, instance_bboxes_gt, instance_masks_pred, instance_masks_gt, weight, height, with_mask)
        # instance_iou['image_id'] = img_id
        instance_iou['bbox_ious'] = box_ious.tolist()
        instance_iou['average_bbox_ious'] = box_ious.mean().tolist()
        if with_mask:
            instance_iou['mask_ious'] = mask_ious.tolist()
            instance_iou['average_mask_ious'] = mask_ious.mean()

        instance_ious[img_id] = instance_iou
        if flag:
            print(f'image: {img_id} lose some instances!')
        # print('completed!')

    bbox_iou = []
    mask_iou = []
    for key in instance_ious:
        bbox_iou.append(instance_ious[key]['average_bbox_ious'])
        if with_mask:
            mask_iou.append(instance_ious[key]['average_mask_ious'])

    bbox_iou = np.array(bbox_iou)
    print(f'average bbox iou: {bbox_iou.mean()}')
    instance_ious['average_bbox_iou'] = bbox_iou.mean()
    if with_mask:
        mask_iou = np.array(mask_iou)
        print(f'average mask iou: {mask_iou.mean()}')
        instance_ious['average_mask_iou'] = mask_iou.mean()

    with open(save_path, 'w') as f:
        json.dump(instance_ious, f)

    print('iou record completed!')


def main():
    parse = argparse.ArgumentParser(
        description='calculate ious of gcr+sam'
    )
    parse.add_argument('prefix', help='Get input path_prefix')
    parse.add_argument('suffix', help='Get input path suffix', default='results')
    parse.add_argument('--gt_json_path', 
                        default="data/coco/annotations/instances_val2017_add-points_size-range0.5.json", 
                        help='Coco format gt json file path')

    args = parse.parse_args()
    
    box_suffix = f'{args.suffix}.bbox.json'
    mask_suffix = f'{args.suffix}.segm.json'

    bbox_pred_json_path = os.path.join(args.prefix, box_suffix)
    mask_pred_json_path = os.path.join(args.prefix, mask_suffix)
    save_path = os.path.join(args.prefix, 'results.iou1.debug.json')

    calculate_ious_on_coco(bbox_pred_json_path, mask_pred_json_path, save_path, args.gt_json_path)


if __name__ == '__main__':
    main()