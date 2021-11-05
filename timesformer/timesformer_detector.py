import logging
import os
from fvcore.common.config import CfgNode
import pandas as pd
import torch

import timesformer.utils.checkpoint as cu
from timesformer.models import build_model
from timesformer.datasets.ta2 import TimesformerEval
import timesformer.utils.checkpoint as cu
from timesformer.config.defaults import get_cfg


class TimesformerDetector:
    def __init__(self, session_id, test_id, test_type,
                 feature_extractor_params, kl_params, evm_params,
                 characterization_params, dataloader_params):
        """
        Constructor for the activity recognition models

        :param session_id (int): Session id for the test
        :param test_id (int): Test id for the test
        :param test_type (str): Type of test
        :param feature_extractor_params (dict): Parameters for feature extractor
        :param kl_params (dict): Parameter for kl divergence
        :param evm_params (dict): Parameter for evm
        :param dataloader_params (dict): Parameters for dataloader
        """
        self.logger = logging.getLogger(__name__)
        self.session_id = session_id
        self.test_id = test_id
        self.logging_header = f"session: {session_id}, test id: {test_id}"
        self.test_type = test_type
        self.base_cfg = get_cfg()
        self.base_cfg.MODEL.MODEL_NAME = feature_extractor_params["model_name"]
        self.base_cfg.MODEL.ARCH = feature_extractor_params["arch"]
        self.base_cfg.MODEL.NUM_CLASSES = feature_extractor_params["num_classes"]
        self.base_cfg.MODEL.NUM_PERSPECTIVES = feature_extractor_params["num_perspectives"]
        self.base_cfg.MODEL.NUM_LOCATIONS = feature_extractor_params["num_locations"]
        self.base_cfg.MODEL.NUM_RELATIONS = feature_extractor_params["num_relations"]
        self.base_cfg.NUM_GPUS = feature_extractor_params["num_gpus"]
        self.base_cfg.TRAIN.CHECKPOINT_FILE_PATH = \
            feature_extractor_params["checkpoint_file_path"]
        self.feature_extractor = build_model(self.base_cfg)
        self.model = build_model(self.base_cfg)
        cu.load_test_checkpoint(self.base_cfg, self.model)

        # Add dataloader parameters
        self.base_cfg.TEST = CfgNode()
        self.base_cfg.TEST.NUM_ENSEMBLE_VIEWS = dataloader_params["num_ensemble_views"]
        self.base_cfg.TEST.NUM_SPATIAL_CROPS = dataloader_params["num_spatial_crops"]
        self.base_cfg.DATA.TEST_CROP_SIZE = dataloader_params["test_crop_size"]
        self.base_cfg.DATA.TRAIN_JITTER_SCALES = dataloader_params["train_jitter_scales"]
        self.base_cfg.DATA.SAMPLING_RATE = dataloader_params["sampling_rate"]
        self.base_cfg.DATA.NUM_FRAMES = dataloader_params["num_frames"]

        self.batch_size = dataloader_params["batch_size"]
        self.number_workers = dataloader_params["n_threads"]
        self.pin_memory = dataloader_params["pin_memory"]

    @torch.no_grad()
    def feature_extraction(self, dataset_path, dataset_root, round_id=None):
        """
        Extract features from novelty detector.

        :param dataset_path (str): Path to the csv file containing all image paths
        :param dataset_root (str): Path to the root directory relative to which
                                   dataset is stored
        :return A tuple of dictionaries
        """
        relative_fpaths = pd.read_csv(dataset_path, header=None)
        relative_fpaths = relative_fpaths.values.tolist()
        absolute_fpaths = list(map(lambda root, rpath: os.path.join(root, rpath[0]),
                                   [dataset_root]*len(relative_fpaths), relative_fpaths))
        test_dataset = TimesformerEval(self.base_cfg, absolute_fpaths)
        loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=self.batch_size,
                                             shuffle=False,
                                             num_workers=self.number_workers,
                                             pin_memory=self.pin_memory)
        self.logger.info(f"{self.logging_header}: Starting feature extraction")
        feature_dict = {}
        logit_dict = {}
        for video_idx, inputs in enumerate(loader):
            inputs = inputs.cuda()
            for input_tensor in torch.unbind(inputs):
                preds, preds_per, preds_loc, preds_rel, feats = self.model(input_tensor)
                feature_dict[relative_fpaths[video_idx][0]] = feats.detach().cpu().numpy()
                logit_dict[relative_fpaths[video_idx][0]] = {
                        "class_preds": preds.detach().cpu().numpy(),
                        "prespective_preds": preds_per.detach().cpu().numpy(),
                        "location_preds": preds_loc.detach().cpu().numpy(),
                        "relation_preds": preds_rel.detach().cpu().numpy()
                    }

        self.logger.info(f"{self.logging_header}: Finished feature extraction")
        return feature_dict, logit_dict
