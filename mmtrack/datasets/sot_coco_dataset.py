# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List

import numpy as np
from mmengine.dataset import force_full_init
from mmengine.fileio.file_client import FileClient
from pycocotools.coco import COCO

from mmtrack.registry import DATASETS
from .base_sot_dataset import BaseSOTDataset

import random

coco_id_name_map={1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
                   6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
                   11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
                   16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
                   22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
                   28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
                   35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
                   40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
                   44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
                   51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
                   56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
                   61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
                   70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
                   77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
                   82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
                   88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}


@DATASETS.register_module()
class SOTCocoDataset(BaseSOTDataset):
    """Coco dataset of single object tracking.

    The dataset only support training mode.
    """

    def __init__(self, with_class=False, *args, **kwargs):
        """Initialization of SOT dataset class."""
        super().__init__(*args, **kwargs)
        self.with_class = with_class

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``.
        Each instance is viewed as a video.

        Returns:
            list[dict]: The length of the list is the number of valid object
                annotations. The inner dict contains annotation ID in coco
                API.
        """
        file_client = FileClient.infer_client(uri=self.ann_file)
        with file_client.get_local_path(self.ann_file) as local_file:
            self.coco = COCO(local_file)
        ann_list = list(self.coco.anns.keys())
        data_infos = [
            dict(ann_id=ann) for ann in ann_list
            if self.coco.anns[ann].get('iscrowd', 0) == 0
        ]
        return data_infos

    def get_bboxes_from_video(self, video_idx: int) -> np.ndarray:
        """Get bbox annotation about one instance in an image.

        Args:
            video_idx (int): The index of video.

        Returns:
            ndarray: In [1, 4] shape. The bbox is in (x, y, w, h) format.
        """
        ann_id = self.get_data_info(video_idx)['ann_id']
        anno = self.coco.anns[ann_id]
        bboxes = np.array(anno['bbox'], dtype=np.float32).reshape(-1, 4)
        return bboxes

    def get_img_infos_from_video(self, video_idx: int) -> dict:
        """Get the image information about one instance in a image.

        Args:
            video_idx (int): The index of video.

        Returns:
            dict: {
                    'video_id': int,
                    'frame_ids': np.ndarray,
                    'img_paths': list[str],
                    'video_length': int
                  }
        """
        ann_id = self.get_data_info(video_idx)['ann_id']
        imgs = self.coco.loadImgs([self.coco.anns[ann_id]['image_id']])
        img_names = [
            osp.join(self.data_prefix['img_path'], img['file_name'])
            for img in imgs
        ]
        frame_ids = np.arange(self.get_len_per_video(video_idx))
        if self.with_class:
            img_infos = dict(
                video_id=video_idx,
                frame_ids=frame_ids,
                img_paths=img_names,
                video_length=1,
                class_name=coco_id_name_map[self.coco.anns[ann_id]['category_id']])
        else:
            img_infos = dict(
                video_id=video_idx,
                frame_ids=frame_ids,
                img_paths=img_names,
                video_length=1)
        return img_infos

    @force_full_init
    def get_len_per_video(self, video_idx: int) -> int:
        """Get the number of frames in a video. Here, it returns 1 since Coco
        is a image dataset.

        Args:
            video_idx (int): The index of video. Each video_idx denotes an
                instance.

        Returns:
            int: The length of video.
        """
        return 1

    def prepare_train_data(self, video_idx: int) -> dict:
        """Get training data sampled from some videos. We firstly sample two
        videos from the dataset and then parse the data information in the
        subsequent pipeline. The first operation in the training pipeline must
        be frames sampling.

        Args:
            video_idx (int): The index of video.

        Returns:
            dict: Training data pairs, triplets or groups.
        """
        video_idxes = random.choices(list(range(len(self))), k=2)
        pair_video_infos = []
        for video_idx in video_idxes:
            ann_infos = self.get_ann_infos_from_video(video_idx)
            img_infos = self.get_img_infos_from_video(video_idx)
            video_infos = dict(**img_infos, **ann_infos)
            pair_video_infos.append(video_infos)
        
        results = self.pipeline(pair_video_infos)
        return results
