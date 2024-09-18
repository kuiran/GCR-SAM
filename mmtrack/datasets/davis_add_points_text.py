from mmdet.datasets.coco import CocoDataset
from mmtrack.registry import DATASETS
from typing import List, Union
import os.path as osp
from .coco_add_points_text import Coco_add_points_text1


@DATASETS.register_module()
class Davis_add_points_text(Coco_add_points_text1):
    """Dataset for COCO."""

    METAINFO = {
        'classes':
        ('person', 'bicycle', 'car', 'motorcycle', 'black swan', 'camel', 'cow', 
         'dog', 'sheep', 'gold fish',  'horse', 'pig', 'kite surfing', 'bag',
         'gun', 'box', 'cell phone', 'surfboard', 'rope', 'parachute cord'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
         (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
         (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
         (0, 82, 0), (120, 166, 157)]
    }