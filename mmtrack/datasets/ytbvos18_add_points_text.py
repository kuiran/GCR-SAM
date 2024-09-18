from mmdet.datasets.coco import CocoDataset
from mmtrack.registry import DATASETS
from typing import List, Union
import os.path as osp
from .coco_add_points_text import Coco_add_points_text1


@DATASETS.register_module()
class Ytbvos_add_points_text(Coco_add_points_text1):
    """Dataset for COCO."""

    METAINFO = {
        'classes':
        ('bear', 'tree', 'gorilla', 'backpack', 'person', 'horse', 
        'car', 'eagle', 'bird', 'bicyle', 'lizard', 'skateboard', 
        'paraglider', 'knife', 'box', 'motorcycle', 'umbrella', 'kangaroo', 
        'camel', 'frisbee', 'sea lion', 'ball', 'flag', 'bus', 'hand', 
        'hat', 'lion', 'rabbit', 'train', 'cup', 'fire truck', 'dog', 
        'spider', 'snake', 'cow', 'goose', 'cat', 'deer', 'leopard', 
        'panda', 'mirror', 'duck', 'raccoon', 'trumpet', 'microphone', 
        'cabinet', 'monkey', 'bottle', 'fish', 'claw', 'tennis racket', 
        'tissue', 'shark', 'frog', 'turtle', 'truck', 'airplane', 'ring', 
        'tiger', 'snail', 'butterfly', 'giraffe', 'leg', 'bracelet', 
        'boat', 'mouse', 'kangaroo', 'toilet', 'shelf', 'elephant', 'jellyfish', 
        'trash bin', 'squirrel', 'vegetable', 'bag', 'glasses', 'stopcock', 
        'basket', 'towel', 'license plate', 'duck', 'picture', 'oar', 'shoe', 
        'pool', 'apple', 'camera', 'chain', 'penguin', 'curtain', 'sheep', 
        'scooter', 'cartoon', 'soap', 'barricade', 'shovel', 'toys', 'basin', 
        'watch', 'pole', 'hedgehog', 'guitar', 'pillow', 'crocodile', 'cat teaser stick', 
        'dolphin', 'carpet', 'board', 'scarf', 'telegraph pole', 'mantis', 'monitor', 
        'saddle', 'fox', 'peacock'),
            # palette is a list of color tuples, which is used for visualization.
        'palette':
        None
    }