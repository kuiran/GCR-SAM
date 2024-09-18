# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import time
from typing import List

from mmtrack.registry import DATASETS
from .base_sot_dataset import BaseSOTDataset

import random
import numpy as np


@DATASETS.register_module()
class LaSOTDataset(BaseSOTDataset):
    """LaSOT dataset of single object tracking.

    The dataset can both support training and testing mode.
    """

    def __init__(self, replace_first_frame_ann=False, first_frame_ann_path=None, with_class=False, *args, **kwargs):
        """Initialization of SOT dataset class."""
        super(LaSOTDataset, self).__init__(*args, **kwargs)
        self.with_class = with_class
        self.replace_first_frame_ann = replace_first_frame_ann
        self.first_frame_ann_path = first_frame_ann_path

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``.

        Returns:
            list[dict]: The length of the list is the number of videos. The
                inner dict is in the following format:
                    {
                        'video_path': the video path
                        'ann_path': the annotation path
                        'start_frame_id': the starting frame number contained
                            in the image name
                        'end_frame_id': the ending frame number contained in
                            the image name
                        'framename_template': the template of image name
                    }
        """
        print('Loading LaSOT dataset...')
        start_time = time.time()
        data_infos = []
        data_infos_str = self._loadtxt(
            self.ann_file, return_ndarray=False).split('\n')
        # the first line of annotation file is a dataset comment.
        for line in data_infos_str[1:]:
            # compatible with different OS.
            line = line.strip().replace('/', os.sep).split(',')
            data_info = dict(
                video_path=line[0],
                ann_path=line[1],
                start_frame_id=int(line[2]),
                end_frame_id=int(line[3]),
                framename_template='%08d.jpg')
            data_infos.append(data_info)
        print(f'LaSOT dataset loaded! ({time.time()-start_time:.2f} s)')
        return data_infos

    def get_visibility_from_video(self, video_idx: int) -> dict:
        """Get the visible information of instance in a video.

        Args:
            video_idx (int): The index of video.

        Returns:
            dict: The visibilities of each object in the video.
        """
        video_path = osp.dirname(self.get_data_info(video_idx)['video_path'])
        full_occlusion_file = osp.join(self.data_prefix['img_path'],
                                       video_path, 'full_occlusion.txt')
        out_of_view_file = osp.join(self.data_prefix['img_path'], video_path,
                                    'out_of_view.txt')
        full_occlusion = self._loadtxt(
            full_occlusion_file, dtype=bool, delimiter=',')
        out_of_view = self._loadtxt(
            out_of_view_file, dtype=bool, delimiter=',')
        visible = ~(full_occlusion | out_of_view)
        return dict(visible=visible)

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
            if self.with_class:
                video_infos['class_name'] = video_infos['img_paths'][0].split('/')[3]
            pair_video_infos.append(video_infos)
        
        results = self.pipeline(pair_video_infos)
        return results
    
    def get_replace_first_frame_bbox(self, video_ind):
        """
        Args:
            video_ind: int
        Return:
            bbox: ndarray: (1, 4)
        """
        bboxes = self._loadtxt(self.first_frame_ann_path, dtype=float, delimiter=',')
        # bboxes[:, 2:] += bboxes[:, :2]
        return bboxes[video_ind]
    
    def prepare_test_data(self, video_idx: int, frame_idx: int) -> dict:
        """Get testing data of one frame. We parse one video, get one frame
        from it and pass the frame information to the pipeline.

        Args:
            video_idx (int): The index of video.
            frame_idx (int): The index of frame.

        Returns:
            dict: Testing data of one frame.
        """
        # Avoid reloading the files of video information
        # repeatedly in all frames of one video.
        if self.test_memo.get('video_idx', None) != video_idx:
            self.test_memo.video_idx = video_idx
            ann_infos = self.get_ann_infos_from_video(video_idx)
            # if self.replace_first_frame_ann:
            # ann_infos['bboxes'][0] = self.get_replace_first_frame_bbox(video_idx)
            if self.replace_first_frame_ann:
                ann_infos['bboxes'][0] = self.get_replace_first_frame_bbox(video_idx)
            
            img_infos = self.get_img_infos_from_video(video_idx)
            self.test_memo.video_infos = dict(**img_infos, **ann_infos)
        assert 'video_idx' in self.test_memo and 'video_infos'\
            in self.test_memo

        results = {}
        results['img_path'] = self.test_memo.video_infos['img_paths'][
            frame_idx]
        results['frame_id'] = frame_idx
        results['video_id'] = video_idx
        results['video_length'] = self.test_memo.video_infos['video_length']

        results['instances'] = []
        instance = {}
        instance['bbox'] = self.test_memo.video_infos['bboxes'][frame_idx]
        instance['visible'] = self.test_memo.video_infos['visible'][frame_idx]
        instance['bbox_label'] = np.array([0], dtype=np.int32)
        results['instances'].append(instance)

        if self.with_class:
            results['class_name'] = results['img_path'].split('/')[3]

        results = self.pipeline(results)
        return results
