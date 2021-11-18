from sail_on_client.agent.ond_agent import ONDAgent
from argparse import Namespace
from typing import Dict, Any, Tuple
import ubelt as ub

from timesformer.timesformer_detector import TimesformerDetector

class TimesformerAdapter(ONDAgent):
    def __init__(self,
                 fe_params,
                 kl_params,
                 evm_params,
                 fine_tune_params,
                 feedback_interpreter_params,
                 characterization_params,
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
        self.fine_tune_params = fine_tune_params,
        self.feedback_interpreter_params = feedback_interpreter_params,
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
        self.detector = TimesformerDetector(toolset['session_id'],
                                            toolset['test_id'],
                                            toolset['test_type'],
                                            self.fe_params,
                                            self.kl_params,
                                            self.evm_params,
                                            self.fine_tune_params,
                                            self.feedback_interpreter_params,
                                            self.dataloader_params,
                                            self.detection_threshold)

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
        return self.detector.world_detection(toolset['features_dict'],
                                             toolset['logit_dict'],
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
        raise NotImplementedError()
        return self.detector.novelty_characterization(
            toolset['dataset'],
            toolset['dataset_root'],
            toolset['round_id'],
        )

    def novelty_adaptation(self, toolset: Dict) -> None:
        """
        Novelty adaptation for the algorithm.

        Args:
            toolset: Parameters for initialization

        Returns:
            None
        """
        raise NotImplementedError()
        return self.detector.novelty_adaptation(
            toolset['dataset'],
            toolset['dataset_root'],
            toolset['round_id'],
        )

    def execute(self, toolset: Dict, step_descriptor: str) -> Any:
        """
        Overriden execute method for executing different steps in the ond
        protocol

        :param toolset (dict): A dictionary with parameters for the algorithm
        :param step_descriptor (str): A string denoting the algorithm step that
                                      should be executed
        """
        return self.step_dict[step_descriptor](toolset)
