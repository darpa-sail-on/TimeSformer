from timesformer.timesformer_detector import TimesformerDetector

import logging
import os
import torch
import pandas as pd

class TimesformerWithRedlightDetector(TimesformerDetector):
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
        super().__init__(session_id, test_id, test_type,
                         feature_extractor_params, kl_params, evm_params,
                         characterization_params, dataloader_params,
                         detection_threshold)
        self.logger = logging.getLogger(__name__)
        self.red_light_ind = False
        self.logger.info(f"{self.logging_header}: Initialization complete")


    def world_detection(self, feature_dict, logit_dict, red_light_video,
                        round_id=None):
        """
        Detect change in world

        :param feature_dict   (dict): A dictionary containing features
        :param logit_dict     (dict): A dictionary containing logits
        :param red_light_video (str): Video where the world change
        :param round_id        (int): Round id

        :return Path to csv file containing results for detecting change in world
        """
        detection_results = super().world_detection(feature_dict, logit_dict, round_id)
        detection_df = pd.read_csv(detection_results, header=None)

        logging_header = self._add_round_to_header(self.logging_header, round_id)
        self.logger.info(f"{logging_header}: Starting to detect change in world")
        if round_id is None:
            result_path = f"wd_{self.session_id}_{self.test_id}.csv"
        else:
            result_path = f"wd_{self.session_id}_{self.test_id}_{round_id}.csv"
        if self.red_light_ind:
            detection_df[1] = 1.0
        else:
            detection_df[1] = 0.0
        detection_df.to_csv(result_path, index=False, header=False, float_format='%.4f')
        # Set red_light found after the round
        if red_light_video in feature_dict.keys():
            self.red_light_ind = True
            self.has_world_changed = True
            self.logger.info("Detected red light image")
        return result_path

    def novelty_classification(self, feature_dict, logit_dict, round_id=None):
        classification_results = super().novelty_classification(feature_dict,
                                                                logit_dict,
                                                                round_id)
        if not self.has_world_changed:
            results_df = pd.read_csv(classification_results, header=None)
            image_names = results_df[0].tolist()
            predictions = torch.Tensor(results_df[list(range(1, 32))].copy().to_numpy())
            predictions[:, 30] = 0.0
            predictions = torch.nn.functional.softmax(predictions, dim=1).detach()
            updated_df = pd.DataFrame(zip(image_names, *predictions.t().tolist()))
            updated_df.to_csv(classification_results,
                              index = False, header = False, float_format='%.4f')
        return classification_results
