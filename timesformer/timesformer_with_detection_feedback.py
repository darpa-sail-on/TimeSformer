from timesformer.timesformer_rd_detector import TimesformerWithRedlightDetector
from timesformer.utils.realign_logits import realign_logits
from timesformer.finch import FINCH
from vast.opensetAlgos.extreme_value_machine import ExtremeValueMachine

import logging
import os
import torch
import pandas as pd
import numpy as np

class TimesformerWithDetectionFeedback(TimesformerWithRedlightDetector):
    def __init__(self, session_id, test_id, test_type, feature_extractor_params,
                 kl_params, evm_params, dataloader_params, adaptation_params,
                 detection_threshold, feedback_obj):
        """
        Constructor for the activity recognition models

        :param session_id (int): Session id for the test
        :param test_id (int): Test id for the test
        :param test_type (str): Type of test
        :param feature_extractor_params (dict): Parameters for feature extractor
        :param kl_params (dict): Parameter for kl divergence
        :param evm_params (dict): Parameter for evm
        :param dataloader_params (dict): Parameters for dataloader
        :param adaptation_params (dict): Parameters for adaptation
        :param detection_threshold (float): The threshold for binary detection
        :param feedback_obj: An instance used for requesting feedback
        """
        super().__init__(session_id, test_id, test_type,
                         feature_extractor_params, kl_params, evm_params,
                         {}, dataloader_params, detection_threshold)
        self.logger = logging.getLogger(__name__)
        self.red_light_ind = False
        self.novel_features = []
        self.features_from_round = {}
        self.feedback_obj = feedback_obj
        self.adaptation_params = adaptation_params
        self.discover_evm = None
        self.logger.info(f"{self.logging_header}: Initialization complete")


    def novelty_classification(self, feature_dict, logit_dict, round_id=None):
        self.features_from_round = feature_dict
        logging_header = self._add_round_to_header(self.logging_header, round_id)
        if not self.has_world_changed or not self.discover_evm:
            return super().novelty_classification(feature_dict, logit_dict, round_id)
        else:
            image_names, FVs = zip(*feature_dict.items())
            self.image_names = list(image_names)
            if not hasattr(self, "class_probabilities"):
                class_map = map(self.evm.class_probabilities, FVs)
                self.class_probabilities = torch.stack(list(class_map), dim=0)
                self.max_probabilities, _ = torch.max(self.class_probabilities, dim=2)

            # Realign logits
            class_logits = []
            for _, logit in logit_dict.items():
                class_logit = torch.Tensor(logit["class_preds"])
                class_logits.append(class_logit)
            logits_tensor = torch.stack(class_logits).cpu().detach()
            realigned_logits = realign_logits(logits_tensor)
            logits_tensor = torch.sum(realigned_logits, dim=1)
            softmax_scores = torch.nn.functional.softmax(logits_tensor, dim=1)

            # Combine unknown and discovered prob
            discovered_probs = map(self.discover_evm.known_probs,
                                   map(lambda x: torch.Tensor(x).double(), FVs))
            discovered_probs = torch.stack(list(discovered_probs), axis=0)
            discovered_max_probs, _ = torch.max(discovered_probs, axis=2)
            mean_unknown_probs = 1 - torch.mean(self.max_probabilities, axis=1,
                                                keepdim=True)
            mean_discovered_probs = torch.mean(discovered_max_probs, axis=1,
                                               keepdim=True)
            self.logger.info(f"mean unknown prob: {mean_unknown_probs}")
            self.logger.info(f"mean discovered prob: {mean_discovered_probs}")
            unknown_probs, _ = torch.min(torch.cat([mean_unknown_probs,
                                                    mean_discovered_probs],
                                                    axis=1), axis=1)

            # Scale logits based on unknown probs
            scaled_unknown_probs = torch.ones(unknown_probs.shape).double()
            scaled_unknown_probs[unknown_probs >= self.detection_threshold] = \
                (unknown_probs[unknown_probs >= self.detection_threshold] - 0.001)
            scaled_softmax = torch.einsum("ij,i->ij", softmax_scores,
                                          scaled_unknown_probs)
            all_rows_tensor = torch.cat((scaled_softmax,
                                         unknown_probs.view(-1, 1)), 1)

            # Normalize and write scaled results
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

    def novelty_adaptation(self, round_id: int) -> None:
        if not self.has_world_changed:
            return
        logging_header = self._add_round_to_header(self.logging_header, round_id)
        self.logger.info(f"{logging_header}: Starting Adaptation")
        image_names = list(self.features_from_round.keys())
        detection_df = self.feedback_obj.get_feedback(round_id,
                                                      list(range(len(image_names))),
                                                      image_names)
        novel_df = detection_df[detection_df["detection"] > 0]
        n_image_ids = novel_df["id"].tolist()
        novel_features_from_round = list(map(lambda key: self.features_from_round[key],
                                             n_image_ids))
        self.novel_features.extend(novel_features_from_round)
        if len(self.novel_features) < self.adaptation_params["min_samples"]:
            self.logger.info("Insufficient samples for adaptation")
            return
        clusters, num_clusters, _ = FINCH(np.concatenate(self.novel_features),
                                          distance="cosine")
        cluster_labels, cluster_count = np.unique(clusters[:, -1], return_counts=True)
        self.logger.info(f"{logging_header}: Retraining EVM")
        self.discover_evm = ExtremeValueMachine(tail_size=8000,
                                                cover_threshold=0.8,
                                                distance_multiplier=0.4,
                                                labels=cluster_labels,
                                                distance_metric="cosine",
                                                device=torch.device("cuda:0"))
        novel_features_map = map(lambda x: torch.from_numpy(x), self.novel_features)
        feature_tensor = torch.cat(list(novel_features_map), dim=0)
        feature_labels  = torch.from_numpy(clusters[:, -1]).float()
        known_features = torch.load(self.adaptation_params["known_features"])["feats"]
        known_features = torch.cat(known_features)
        self.discover_evm.fit(feature_tensor, feature_labels, extra_negatives=known_features)
        self.logger.info(f"{logging_header}: Finished Retraining EVM")
