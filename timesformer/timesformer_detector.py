import copy
import logging
import numpy as np
import os
import pandas as pd
import torch

from fvcore.common.config import CfgNode
from vast.opensetAlgos.extreme_value_machine import ExtremeValueMachine

import timesformer.utils.checkpoint as cu
from timesformer.models import build_model
from timesformer.datasets.ta2 import TimesformerEval
from timesformer.config.defaults import get_cfg
from timesformer.kl_divergence import kl_divergence_based_wd
from timesformer.utils.realign_logits import realign_logits


class TimesformerDetector:
    def __init__(self, session_id, test_id, test_type,
                 feature_extractor_params, kl_params, evm_params,
                 characterization_params, dataloader_params,
                 detection_threshold):
        """
        Constructor for the activity recognition models

        :param session_id (int): Session id for the test
        :param test_id (int): Test id for the test
        :param test_type (str): Type of test
        :param feature_extractor_params (dict): Parameters for feature extractor
        :param kl_params (dict): Parameter for kl divergence
        :param evm_params (dict): Parameter for evm
        :param dataloader_params (dict): Parameters for dataloader
        :param detection_threshold (float): The threshold for binary detection
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
        self.model.eval()

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

        # Add KL parameters
        self.kl_threshold = kl_params["KL_threshold"] * kl_params["threshold_scale"]
        kl_decay_rate = kl_params["decay_rate"]
        num_rounds = kl_params["num_rounds"]
        self.sliding_window = []
        self.past_window = []
        self.window_size = kl_params["window_size"]
        self.sigma_train = kl_params["sigma_train"]
        self.mu_train = kl_params["mu_train"]
        self.acc = 0.0
        self.has_world_changed = False
        self.kl_threshold_decay = (num_rounds  * kl_decay_rate)/float(num_rounds)

        self.detection_threshold = detection_threshold
        self.evm = ExtremeValueMachine.load(evm_params["model_path"],
                                            device=torch.device("cuda:0"))
        torch.manual_seed(0)
        np.random.seed(0)
        self.logger.info(f"{self.logging_header}: Initialization complete")

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
                preds, preds_per, preds_loc, preds_rel, feats = \
                        self.model(input_tensor)
                feature_dict[relative_fpaths[video_idx][0]] = feats.detach().cpu().numpy()
                logit_dict[relative_fpaths[video_idx][0]] = {
                        "class_preds": preds.detach().cpu().numpy(),
                        "prespective_preds": preds_per.detach().cpu().numpy(),
                        "location_preds": preds_loc.detach().cpu().numpy(),
                        "relation_preds": preds_rel.detach().cpu().numpy()
                    }
        self.logger.info(f"{self.logging_header}: Finished feature extraction")
        return feature_dict, logit_dict

    def world_detection(self, feature_dict, logit_dict, round_id=None):
        """
        Detect Change in World

        :param feature_dict (dict): Dictionary containing features
        :param logit_dict   (dict): Dictionary containing logits
        :param round_id     (int): Integer identifier for round

        :return string containing path to csv file with the results
        """
        logging_header = self._add_round_to_header(self.logging_header, round_id)
        self.logger.info(f"{logging_header}: Starting to detect change in world")
        if round_id is None:
            result_path = f"wd_{self.session_id}_{self.test_id}.csv"
        else:
            result_path = f"wd_{self.session_id}_{self.test_id}_{round_id}.csv"
        image_names, FVs = zip(*feature_dict.items())
        if self.has_world_changed:
            image_predictions =  [1.0]*len(image_names)
        else:
            class_map = map(self.evm.known_probs,
                            map(lambda x: torch.Tensor(x).double(), FVs))
            self.class_probabilities = torch.stack(list(class_map), axis=0)
            self.max_probabilities, _ = torch.max(self.class_probabilities, axis=2)
            mean_max_probs = torch.mean(self.max_probabilities, axis=1)
            image_predictions, self.sliding_window = kl_divergence_based_wd(
                    mean_max_probs, self.sliding_window, self.window_size,
                    self.mu_train, self.sigma_train, self.kl_threshold,
                    self.acc)
            self.acc = image_predictions[-1]
            if self.acc > self.detection_threshold:
                self.has_world_changed = True
        df = pd.DataFrame(zip(image_names, image_predictions),
                          columns=['id', 'P_world_changed'])
        self.logger.info(f"{logging_header}: Number of samples in results {df.shape}")
        df.to_csv(result_path, index=False, header=False, float_format='%.4f')
        self.logger.info(f"{logging_header}: Finished with change detection")
        self.kl_threshold =  self.kl_threshold - self.kl_threshold_decay
        return result_path

    def novelty_classification(self, feature_dict, logit_dict, round_id=None):
        logging_header = self._add_round_to_header(self.logging_header, round_id)
        self.logger.info(f"{logging_header}: Starting to classify samples")
        image_names, FVs = zip(*feature_dict.items())
        self.image_names = list(image_names)
        if not hasattr(self, "class_probabilities"):
            class_map = map(self.evm.class_probabilities, FVs)
            self.class_probabilities = torch.stack(list(class_map), axis=0)
            self.max_probabilities, _ = torch.max(self.class_probabilities, axis=2)
        class_logits = []
        for _, logit in logit_dict.items():
            class_logit = torch.Tensor(logit["class_preds"])
            class_logits.append(class_logit)
        logits_tensor = torch.stack(class_logits).cpu().detach()
        realigned_logits = realign_logits(logits_tensor)
        class_probabilities = realign_logits(self.class_probabilities)
        m = 1 - torch.mean(self.max_probabilities, axis=1)
        known_probs = torch.mean(torch.tensor(class_probabilities), axis=1)
        pu = torch.zeros(m.shape)
        logits_tensor = torch.sum(realigned_logits, dim=1)
        softmax_scores = torch.nn.functional.softmax(logits_tensor, dim=1)
        self.logger.info(f"Softmax scores: {torch.argmax(softmax_scores, dim=1)}")
        self.logger.info(f"EVM scores: {torch.argmax(known_probs, dim=1)}")
        self.logger.info(f"Acc: {self.acc}")
        if self.has_world_changed:
            scaled_m = torch.ones(m.shape).double()
            scaled_m[m >= self.detection_threshold] = \
                (m[m >= self.detection_threshold] - 0.001)
            scaled_softmax = torch.einsum("ij,i->ij", softmax_scores, scaled_m)
            all_rows_tensor = torch.cat((scaled_softmax, m.view(-1, 1)), 1)
        else:
            pu = pu.view(-1, 1)
            all_rows_tensor = torch.cat((softmax_scores, pu), 1)
        norm = torch.norm(all_rows_tensor, p=1, dim=1)
        normalized_tensor = all_rows_tensor/norm[:, None]
        df = pd.DataFrame(zip(image_names, *normalized_tensor.t().tolist()))
        if round_id is None:
            result_path = f"ncl_{self.session_id}_{self.test_id}.csv"
        else:
            result_path = f"ncl_{self.session_id}_{self.test_id}_{round_id}.csv"
        df.to_csv(result_path, index = False, header = False, float_format='%.4f')
        self.logger.info(f"{logging_header}: Finished classifying samples")
        return result_path

    def _add_round_to_header(self, logger_header, round_id):
        if round_id is not None:
            logger_header += " Round id: {}".format(round_id)
        return logger_header
