from sail_on_client.agent.ond_agent import ONDAgent
from argparse import Namespace
from typing import Dict, Any, Tuple
import ubelt as ub

from timesformer.adaptive_timesformer_detector import AdaptiveTimesformerDetector
from timesformer.timesformer_adapter import TimesformerAdapter


class AdaptiveTimesformerAdapter(TimesformerAdapter):
    def __init__(self,
                 fe_params,
                 kl_params,
                 evm_params,
                 fine_tune_params,
                 feedback_interpreter_params,
                 #characterization_params,
                 dataloader_params,
                 detection_threshold) -> None:
        """
        Constructor for X3D adapter

        Returns:
            None
        """
        self.fe_params = fe_params
        self.kl_params = kl_params
        self.evm_params = evm_params
        self.dataloader_params = dataloader_params
        self.fine_tune_params = fine_tune_params
        self.feedback_interpreter_params = feedback_interpreter_params
        self.detection_threshold = detection_threshold
        ONDAgent.__init__(self)
        self.step_dict = {"Initialize": self.initialize,
                          "FeatureExtraction": self.feature_extraction,
                          "WorldDetection": self.world_detection,
                          "NoveltyClassification": self.novelty_classification,
                          "NoveltyAdaptation": self.novelty_adaptation,
                          "NoveltyCharacterization": self.novelty_characterization}


    def initialize(self, toolset: Dict) -> None:
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
        self.detector = AdaptiveTimesformerDetector(toolset['session_id'],
                                            toolset['test_id'],
                                            toolset['test_type'],
                                            self.fe_params,
                                            self.kl_params,
                                            self.evm_params,
                                            self.fine_tune_params,
                                            self.feedback_interpreter_params,
                                            self.dataloader_params,
                                            self.detection_threshold,
                                            feedback_obj)


    def novelty_adaptation(self, toolset: Dict) -> None:
        """
        Novelty adaptation for the algorithm.

        Args:
            toolset: Parameters for initialization

        Returns:
            None
        """
        return self.detector.novelty_adaptation(toolset['round_id'])
