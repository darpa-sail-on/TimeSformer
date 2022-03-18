from sail_on_client.agent.ond_agent import ONDAgent
from argparse import Namespace
from typing import Dict, Any, Tuple
import ubelt as ub
import logging

from timesformer.timesformer_with_detection_feedback import TimesformerWithDetectionFeedback

class TimesformerWithDetectionFeedbackAdapter(ONDAgent):
    def __init__(self,
                 fe_params,
                 kl_params,
                 evm_params,
                 dataloader_params,
                 adaptation_params,
                 detection_threshold) -> None:
        """
        Constructor for timesformer that can take into account detection feedback.

        Returns:
            None
        """
        self.fe_params = fe_params
        self.kl_params = kl_params
        self.evm_params = evm_params
        self.dataloader_params = dataloader_params
        self.adaptation_params = adaptation_params
        self.detection_threshold = detection_threshold
        self.logger = logging.getLogger(__name__)
        ONDAgent.__init__(self)
        self.step_dict = {"Initialize": self.initialize,
                          "FeatureExtraction": self.feature_extraction,
                          "WorldDetection": self.world_detection,
                          "NoveltyClassification": self.novelty_classification,
                          "NoveltyAdaptation": self.novelty_adaptation,
                          "NoveltyCharacterization": self.novelty_characterization}

    def initialize(self, toolset: Dict):
        """
        Initialize method for the algorithm.

        Args:
            toolset: Parameters for initialization

        Returns:
            None
        """
        if "FeedbackInstance" in toolset:
            feedback_obj = toolset["FeedbackInstance"]
        else:
            feedback_obj = None
            self.logger.warn("No feedback object provided")
        self.detector = TimesformerWithDetectionFeedback(toolset['session_id'],
                                                         toolset['test_id'],
                                                         toolset['test_type'],
                                                         self.fe_params,
                                                         self.kl_params,
                                                         self.evm_params,
                                                         self.dataloader_params,
                                                         self.adaptation_params,
                                                         self.detection_threshold,
                                                         feedback_obj)

    def feature_extraction(self, toolset: Dict) -> Tuple:
        """
        Feature extraction for the algorithm.

        Args:
            toolset: Parameters for initialization

        Returns:
            Tuple of features and logit dict
        """
        return self.detector.feature_extraction(toolset['dataset'],
                                                toolset['dataset_root'],
                                                toolset['round_id'])

    def world_detection(self, toolset: Dict) -> str:
        """
        Detect that the world has changed for the algorithm.

        Args:
            toolset: Parameters for initialization

        Returns:
            None
        """
        self.redlight_image = toolset['redlight_image']
        return self.detector.world_detection(toolset['features_dict'],
                                             toolset['logit_dict'],
                                             self.redlight_image,
                                             toolset['round_id'])

    def novelty_classification(self, toolset: Dict) -> str:
        """
        Novelty classification for the algorithm.

        Args:
            toolset: Parameters for initialization

        Returns:
            None
        """
        return self.detector.novelty_classification(toolset['features_dict'],
                                                    toolset['logit_dict'],
                                                    toolset['round_id'])

    def novelty_characterization(self, toolset: Dict) -> str:
        """
        Novelty characterization for the algorithm.

        Args:
            toolset: Parameters for initialization

        Returns:
            None
        """
        pass

    def novelty_adaptation(self, toolset: Dict) -> None:
        """
        Novelty adaptation for the algorithm.

        Args:
            toolset: Parameters for initialization

        Returns:
            None
        """
        return self.detector.novelty_adaptation(toolset['round_id'])


    def execute(self, toolset: Dict, step_descriptor: str) -> Any:
        """
        Overriden execute method for executing different steps in the ond
        protocol

        :param toolset (dict): A dictionary with parameters for the algorithm
        :param step_descriptor (str): A string denoting the algorithm step that
                                      should be executed
        """
        return self.step_dict[step_descriptor](toolset)
