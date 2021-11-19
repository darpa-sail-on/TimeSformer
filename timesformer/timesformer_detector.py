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

from arn.models.fine_tune import FineTune, FineTuneFCANN
from arn.models.novelty_detector import WindowedMeanKLDiv
#from arn.models.novelty_recognizer import FINCHRecognizer
from arn.models.feedback import CLIPFeedbackInterpreter
from arn.models.owhar import OWHAPredictorEVM


class TimesformerDetector:
    def __init__(
        self,
        session_id,
        test_id,
        test_type,
        feature_extractor_params,
        kl_params,
        evm_params,
        fine_tune_params,
        feedback_interpreter_params,
        dataloader_params,
        detection_threshold,
    ):
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

        if feedback_interpreter_param:
            self.interpret_activity_feedback = True
            interpreter = CLIPFeedbackInterpreter(
                feedback_interpreter_params['clip_path'],
                feedback_interpreter_params['clip_templates'],
                feedback_interpreter_params['pred_known_map'],
                feedback_interpreter_params['pred_label_encs'],
                feedback_interpreter_params['feedback_known_map'],
                feedback_interpreter_params['feedback_label_encs'],
            )
        else:
            interpreter=None
            self.interpret_activity_feedback = False

        self.feedback_columns = [
            'kinetics_id_1',
            'kinetics_id_2',
            'kinetics_id_3',
            'kinetics_id_4',
            'kinetics_id_5',
        ]

        # Must store the train features and labels for updating fine tuning.
        self.train_features = torch.load(
            feedback_interpreter_params['train_feature_path'],
        )
        self.train_labels = torch.nn.functional.one_hot(
            torch.cat(self.train_features['labels']).type(torch.long)
        )
        self.train_features = torch.cat(self.train_features['feats'])

        # TODO Store the val features and labels for updating fine tuning.
        #   Currently only used to assess the val performance of model.
        #self.val_features = torch.load(
        #    feedback_interpreter_params['val_feature_path'],
        #)
        #self.val_labels = torch.nn.functional.one_hot(
        #    torch.cat(self.val_features['labels']).type(torch.long)
        #)
        #self.val_features = torch.cat(self.val_features['feats'])

        # OWHAR: FineTune, EVM, FINCH, CLIP Feedback Interpreter args
        self.owhar = OWHAPredictorEVM(
            FineTune.load(
                torch.load(fine_tune_params["model_path"]),
                device=torch.device('cuda'),
            ),
            ExtremeValueMachine.load(
                evm_params["model_path"],
                device=torch.device("cuda:0"),
            ),
            WindowedMeanKLDiv(
                detection_threshold=detection_threshold,
                kl_threshold=kl_params["KL_threshold"],
                kl_threshold_decay_rate=kl_params["decay_rate"],
                mean_train=kl_params["mu_train"],
                std_dev_train=kl_params["sigma_train"],
                window_size=kl_params["window_size"],
                num_rounds=kl_params["num_rounds"],
            ),
            interpreter,
        )

        # Fit the OWHAR model on the given data.
        self.owhar.fit_increment(
            self.train_features,
            self.train_labels,
            True,
            #self.val_features,
            #self.val_labels,
            #True,
        )

        # Obtain detection threshold and kl threshold if None provided
        if feedback_interpreter_params['thresh_set_data']:
            if self.owhar.novelty_detector.kl_threshold is not None:
                logging.warning(
                    'kl_threshold was already set, but finding from data',
                )

            test_features = torch.load(
                feedback_interpreter_params['thresh_set_data'],
            )

            # NOTE self.detection_threshold is NOT informed from val, atm
            kl_threshold = self.owhar.novelty_detector.find_kl_threshold(
                test_features['known'],
                test_features['unknown'],
            )
            self.owhar.novelty_detector.kl_threshold = kl_threshold

        # TODO characterization requires an owhar per subtask.

        self.logger.info(f"{self.logging_header}: Initialization complete")

    @property
    def has_world_changed(self):
        return self.owhar.novelty_detector.has_world_changed

    @property
    def acc(self):
        return self.owhar.novelty_detector.accuracy

    @property
    def detection_threshold(self):
        return self.owhar.novelty_detector.detection_threshold

    def set_detection_threshold(self value):
        return self.owhar.novelty_detector.detection_threshold = value

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
        class_map = map(self.owhar.known_probs,
                        map(lambda x: torch.Tensor(x).double(), FVs))
        self.class_probabilities = torch.stack(list(class_map), axis=0)
        self.max_probabilities = torch.max(self.class_probabilities, axis=2)[0]

        if round_id == 0:
            detections = torch.zeros(len(image_names))
        else:
            detections = self.owhar.novelty_detect.detect(
                self.max_probabilities,
                True,
                self.logger,
            )

        df = pd.DataFrame(
            zip(image_names, detections.tolist()),
            columns=['id', 'P_world_changed'],
        )

        # TODO self.has_world_changed = ... Make properties for these
        #   self.temp_world_changed
        #   self. other selfs...

        self.logger.info(f"{logging_header}: Number of samples in results {df.shape}")
        df.to_csv(result_path, index=False, header=False, float_format='%.4f')
        self.logger.info(f"{logging_header}: Finished with change detection")

        return result_path

    def classification(self, owhar, feature_dict, logit_dict, round_id=None):
        """Perform classification with the given owhar predictor."""
        # TODO The subtasks that consist of characterization are all
        # classification tasks. As was done for the HWR, make a
        raise NotImplementedError(
            'This is lowest priority as part of characterization.',
        )

    def novelty_classification(self, feature_dict, logit_dict, round_id=None):
        logging_header = self._add_round_to_header(self.logging_header, round_id)
        self.logger.info(f"{logging_header}: Starting to classify samples")
        image_names, FVs = zip(*feature_dict.items())
        self.image_names = list(image_names)

        # TODO if world_detection not run first, have to run probs throu

        if not hasattr(self, "class_probabilities"):
            class_map = map(self.class_probabilities, FVs)
            self.class_probabilities = torch.stack(list(class_map), axis=0)
            self.max_probabilities, _ = torch.max(self.class_probabilities, axis=2)

        m = 1 - torch.mean(self.max_probabilities, axis=1)
        known_probs = torch.mean(torch.tensor(self.class_probabilities), axis=1)
        pu = torch.zeros(m.shape)
        class_logits = []
        for _, logit in logit_dict.items():
            class_logit = torch.Tensor(logit["class_preds"])
            class_logits.append(class_logit)
        logits_tensor = torch.stack(class_logits).cpu()
        softmax_scores = torch.nn.functional.softmax(logits_tensor, dim=1)
        softmax_scores = torch.mean(softmax_scores, axis=1)
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

    def binary_novelty_feedback_adapt(
        self,
        max_probabilities,
        detection_threshold,
        feedback_df=None
    ):
        """Performs older version of binary novelty feedback adaptation.

        Returns
        -------
        float | float, pandas.DataFrame
            The new detection threshold is returned. The feedback data_frame is
            returned when feedback_df is not given as an argument.
        """
        # TODO Make your variable names more descriptive.
        m = 1 - torch.mean(max_probabilities, axis=1)
        m = m.detach().cpu().numpy()

        num_pred_known = len(m[m <= detection_threshold])
        num_pred_unknown = len(m[m > detection_threshold])

        pred_known = np.zeros(shape=(num_pred_known, 2), dtype=np.float64)
        pred_unknown = np.zeros(shape=(num_pred_unknown, 2), dtype=np.float64)
        known_idx = np.argwhere(m <= detection_threshold).squeeze()
        unknown_idx = np.argwhere(m > detection_threshold).squeeze()

        known_preds = m[m <= detection_threshold]
        unknown_preds = m[m > detection_threshold]

        pred_known[:, 0] = known_idx
        pred_known[:, 1] = known_preds
        pred_unknown[:, 0] = unknown_idx
        pred_unknown[:, 1] = unknown_preds

        pred_known_sorted = pred_known[pred_known[:, 1].argsort()[::-1]]
        pred_unknown_sorted = pred_unknown[pred_unknown[:, 1].argsort()]

        # Get Feedback from batch predictions of novel (unknown) samples
        income_per_batch = self.feedback_obj.income_per_batch
        half_income_per_batch = int(income_per_batch/2)
        unknown_image = pred_unknown_sorted[:half_income_per_batch, 0]
        known_image = pred_known_sorted[:half_income_per_batch, 0]
        image_list = np.concatenate([known_image, unknown_image]).tolist()

        if feedback_df is None:
            data_frame = self.feedback_obj.get_feedback(
                round_id,
                image_list,
                self.image_names,
            )
        else:
            data_frame = feedback_df

        # Get known and unknown labels
        known_labels = data_frame["labels"][:len(known_image)].to_numpy()
        unknown_labels = data_frame["labels"][len(known_image):].to_numpy()

        # Record known and unknown prediction performance
        known_pred_wrong = len(known_labels[known_labels == 88])
        unknown_pred_wrong = len(unknown_labels[unknown_labels != 88])

        self.logger.info(f"Known pred wrong: {known_pred_wrong}")
        self.logger.info(f"Unknown pred wrong: {unknown_pred_wrong}")

        unknown_acc = unknown_pred_wrong/half_income_per_batch
        known_acc = known_pred_wrong/half_income_per_batch

        if self.feedback_weight * unknown_acc > 0.0:
            detection_threshold -= self.feedback_weight*unknown_acc
        if self.feedback_weight * known_acc > 0.0:
            detection_threshold += self.feedback_weight*known_acc

        self.logger.info(f"New detection threshold is {detection_threshold}")
        if feedback_df is None:
            return np.clip(detection_threshold, 0.0, 1.0), data_frame
        return np.clip(detection_threshold, 0.0, 1.0)

    def novelty_adaption(self, round_id):
        """
        Novelty adaptation
        :param round_id: round id in a test
        """
        self.logger.info(f"Starting novelty_adaption: {round_id}")
        if not self.has_world_changed:
            return

        # Adaptation w/o class size update:
        # Update the detection threshold and get feedback
        detect_thresh, feedback_df = self.binary_novelty_feedback_adapt(
            self.max_probabilities,
            self.detection_threshold,
        )

        # Check if should use feedback from classes.
        if self.interpret_activity_feedback:
            # Get the feedback as label text
            label_text = feedback_df[self.feedback_columns].values

            # Interpret the feedback
            feedback_labels = self.owhar.feedback_interpreter.interpret(
                label_text,
            )

            # TODO Combine the train data with the feedback data for update

            # Incremental fits on all prior train and saved feedback
            self.owhar.fit_increment(
                input_samples,
                labels,
                is_feature_repr=True,
                #val_input_samples,
                #val_labels,
                #val_is_feature_repr=True,
            )

        # TODO Handle the saving of results with updated predictor etc...
        self.probabilities
        self.max_probabilities

        # Adaptation given only novelty information after updating the fine
        # tune and the EVM.
        self.set_detection_threshold(self.binary_novelty_feedback_adapt(
            self.max_probabilities,
            self.detection_threshold,
            feedback_df,
        ))

        # TODO adaptation w/ class size update (thus FINCH after deciding novel
        # classes exist and enough samples for them). This won't happen until
        # later difficulties of the DARPA eval.

    def novelty_characterization(self, dataset_id_list, round_id=None):
        logging_header = self._add_round_to_header(self.logging_header, round_id)
        self.logger.info(f"{logging_header}: Starting to characterize samples")
        if round_id is None:
            result_path = os.path.join(self.csv_folder,
                        f"nc_{self.session_id}_{self.test_id}.csv")
        else:
            result_path = os.path.join(self.csv_folder,
                        f"nc_{self.session_id}_{self.test_id}_{round_id}.csv")
        if len(self.novel_dict.values())>0:
            if len(self.novel_dict.values())>=5:
                data =  torch.stack(list(map(lambda x: x[0], self.novel_dict.values())))
                c_all, num_clust, req_c = FINCH(data.cpu().data.numpy())
                cluster_labels = c_all[:,-1]
                N = num_clust[-1] # number of clusters after clustering.
            else:
                N = 1
        else:
            N = 0
            self.logger.warn(f"{logging_header}: self.Novel_dict is empty.")
        M = len(dataset_id_list)
        col1 = ['id']
        col2 = [('U' + str(k+1)) for k in range(N)]
        col = col1 + col2
        df = pd.DataFrame(np.zeros((M, 1+N)), columns=col)
        df['id']  = dataset_id_list
        if len(self.novel_dict.values())>0:
            novel_index = 0
            for k in range(M):
                if dataset_id_list[k] in self.novel_dict.keys():
                    if len(self.novel_dict.values())>=5:
                        c = cluster_labels[novel_index]
                        novel_index = novel_index + 1
                        df.iloc[k,c+1] = 1.0
                    else:
                        df.iloc[k, novel_index] = 1.0
        if round_id is None:
            result_path = os.path.join(self.csv_folder,
                                       "nc_{}_{}.csv".format(self.session_id,
                                                             self.test_id))
        else:
            result_path = os.path.join(self.csv_folder,
                                       "nc_{}_{}_{}.csv".format(self.session_id,
                                                                self.test_id,
                                                                round_id))
        df.to_csv(result_path, index = False, header = False, float_format='%.4f')
        self.logger.info(f"{logging_header}: Finished characterizing samples")
        return result_path
