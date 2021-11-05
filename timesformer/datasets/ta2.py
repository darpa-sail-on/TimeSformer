# Source: https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/mean.py

import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy
import cv2

import logging
from timesformer.datasets import decoder as decoder
from timesformer.datasets import utils as utils
from timesformer.datasets import video_container as container
from timesformer.datasets.build import DATASET_REGISTRY


logger = logging.getLogger(__name__)


@DATASET_REGISTRY.register()
class TimesformerEval(data.Dataset):
    def __init__(self, cfg, video_paths):
        self._num_clips = (
            cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
        )
        self.video_paths = video_paths
        self.cfg = cfg

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        min_scale, max_scale, crop_size = (
            [self.cfg.DATA.TEST_CROP_SIZE] * 3
            if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
            else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2
            + [self.cfg.DATA.TEST_CROP_SIZE]
        )
        sampling_rate = utils.get_random_sampling_rate(
            self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
            self.cfg.DATA.SAMPLING_RATE,
        )
        self.temporal_sample_index = (
                i // self.cfg.TEST.NUM_SPATIAL_CROPS
                for i in range(self._num_clips)
        )
        self.spatial_sample_index = (
            (
                i % self.cfg.TEST.NUM_SPATIAL_CROPS
                for i in range(self._num_clips)
            )
            if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
            else 1
        )
        all_frames = []
        for temporal_sample_index, spatial_sample_index in \
                zip(self.temporal_sample_index, self.spatial_sample_index):
            video_container = container.get_video_container(
                self.video_paths[index],
                self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                self.cfg.DATA.DECODING_BACKEND,
            )
            frames = decoder.decode(
                video_container,
                sampling_rate,
                self.cfg.DATA.NUM_FRAMES,
                temporal_sample_index,
                self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                target_fps=self.cfg.DATA.TARGET_FPS,
                backend=self.cfg.DATA.DECODING_BACKEND,
                max_spatial_scale=min_scale,
            )

            # Perform color normalization.
            frames = utils.tensor_normalize(
                frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
            )
            frames = frames.permute(3, 0, 1, 2)
            # Perform data augmentation.
            frames = utils.spatial_sampling(
                frames,
                spatial_idx=spatial_sample_index,
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
                random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            )


            if self.cfg.MODEL.ARCH not in ['vit']:
                frames = utils.pack_pathway_output(self.cfg, frames)
            else:
                # Perform temporal sampling from the fast pathway.
                frames = torch.index_select(
                     frames,
                     1,
                     torch.linspace(
                         0, frames.shape[1] - 1, self.cfg.DATA.NUM_FRAMES

                     ).long(),
                )
            all_frames.append(frames)
        if len(all_frames) == 0:
            import pdb
            pdb.set_trace()
        all_frames = torch.stack(all_frames)
        return all_frames

    def __len__(self):
        return len(self.video_paths)
