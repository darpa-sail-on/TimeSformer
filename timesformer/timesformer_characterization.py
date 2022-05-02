from timesformer.timesformer_rd_detector import TimesformerWithRedlightDetector

import logging
import torch
import pandas as pd
from vast.opensetAlgos.extreme_value_machine import ExtremeValueMachine


class TimesformerWithCharacterization(TimesformerWithRedlightDetector):
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
        self.location_evm = ExtremeValueMachine.load(evm_params["location_model_path"],
                                                     device=torch.device("cuda:0"))
        self.perspective_evm = ExtremeValueMachine.load(evm_params["perspective_model_path"],
                                                        device=torch.device("cuda:0"))
        self.relation_on_evm = ExtremeValueMachine.load(evm_params["relation_on_model_path"],
                                                        device=torch.device("cuda:0"))
        self.relation_what_evm = ExtremeValueMachine.load(evm_params["relation_what_model_path"],
                                                          device=torch.device("cuda:0"))
        self.relation_with_evm = ExtremeValueMachine.load(evm_params["relation_with_model_path"],
                                                          device=torch.device("cuda:0"))
        self.num_action_novelty = 0
        self.num_location_novelty = 0
        self.num_prespective_novelty = 0
        self.num_relation_on_novelty = 0
        self.num_relation_what_novelty = 0
        self.num_relation_with_novelty = 0
        self.min_novel_samples = characterization_params["min_novel_samples"]
        self.logger.info(f"{self.logging_header}: Initialization complete")

    def _get_num_novel_samples(self, evm, feature_vectors):
        class_map = map(evm.known_probs,
                        map(lambda x: torch.Tensor(x).double(), feature_vectors))
        class_probabilities = torch.stack(list(class_map), axis=0)
        max_probabilities, _ = torch.max(class_probabilities, axis=2)
        novel_scores = 1 - torch.mean(max_probabilities, axis=1)
        return torch.nonzero(novel_scores >= self.detection_threshold)

    def novelty_classification(self, feature_dict, logit_dict, round_id=None):
        classification_results = super().novelty_classification(feature_dict,
                                                                logit_dict,
                                                                round_id)

        if self.has_world_changed:
            _ , FVs = zip(*feature_dict.items())
            class_novel_scores = 1 - torch.mean(self.max_probabilities, axis=1)
            self.num_action_novelty += len(torch.nonzero(class_novel_scores >= self.detection_threshold))
            self.num_location_novelty += len(self._get_num_novel_samples(self.location_evm, FVs))
            self.num_prespective_novelty += len(self._get_num_novel_samples(self.perspective_evm, FVs))
            self.num_relation_on_novelty += len(self._get_num_novel_samples(self.relation_on_evm, FVs))
            self.num_relation_what_novelty += len(self._get_num_novel_samples(self.relation_what_evm, FVs))
            self.num_relation_with_novelty += len(self._get_num_novel_samples(self.relation_with_evm, FVs))
            self.logger.info(f"""Action novelty: {self.num_action_novelty} \n
                                 Location novelty: {self.num_location_novelty} \n
                                 Perspective novelty: {self.num_prespective_novelty} \n
                                 Relation on novelty: {self.num_relation_on_novelty} \n
                                 Relation what novelty: {self.num_relation_what_novelty} \n
                                 Relation with novelty: {self.num_relation_with_novelty}""")
        return classification_results

    def novelty_characterization(self) -> str:
        max_novelty_level = "action"
        max_num_novelty = self.num_action_novelty

        if self.has_world_changed:
            for novelty_level, num_novelty in zip(
                    ["location", "perspective", "relation_on", "relation_what", "relation_with"],
                    [self.num_location_novelty, self.num_prespective_novelty,
                     self.num_relation_on_novelty, self.num_relation_what_novelty,
                     self.num_relation_with_novelty]):
                if num_novelty > max_num_novelty:
                    max_num_novelty = num_novelty
                    max_novelty_level = novelty_level
        else:
            max_novelty_level = "No novelty"

        self.logger.info(f"{self.logging_header}: {max_novelty_level} found in the test")
        result_path = f"nc_{self.session_id}_{self.test_id}.csv"


        with open(result_path, "w") as f:
            f.write(max_novelty_level)

        self.logger.info(f"{self.logging_header}: Finished characterization")
        return result_path

