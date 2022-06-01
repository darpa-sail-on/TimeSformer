import copy
import logging
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from fvcore.common.config import CfgNode

import timesformer.utils.checkpoint as cu
from timesformer.config.defaults import get_cfg
from timesformer.datasets.ta2 import TimesformerEval
from timesformer.models import build_model
from timesformer.timesformer_detector import TimesformerDetector

# For config parsing of arn predictor.
from docstr.cli.cli import docstr_cap
from exputils.data.labels import NominalDataEncoder

logger = logging.getLogger(__name__)


class AdaptiveTimesformerDetector(TimesformerDetector):
    def __init__(
        self,
        session_id,
        test_id,
        test_type,
        feature_extractor_params,
        dataloader_params,
        detection_threshold,
        data_params,
        predictor,
        feedback_obj,
    ):
        """
        Constructor for the activity recognition models

        :param session_id (int): Session id for the test
        :param test_id (int): Test id for the test
        :param test_type (str): Type of test
        :param feature_extractor_params (dict): Parameters for feature extractor
        :param dataloader_params (dict): Parameters for dataloader
        :param detection_threshold (float): The threshold for binary detection
        :param data_params (dict): The params for data pertaining to arn
        :param predictor (dict): The config for arn predictor, parsed w/ docstr
        :param feedback_obj: An instance used for requesting feedback
        """
        # TODO some of this can be resolved with super().init, but needs
        # thought thru. For now all that matters is it will work.
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

        self.feedback_columns = [
            "class1",
            "class2",
            "class3",
            "class4",
            "class5",
        ]

        # Must store the train features and labels for updating fine tuning.
        self.train_features = torch.load(data_params['train_feature_path'])

        CLASS_MAPPING = [15,1,28,6,29,3, 20,22,23,9,7,10,11,13,14,17,18,5,8,19,24,12,16,26,4,21,0,25,2]

        for x in range(len(self.train_features['feats'])):
            limit = int(self.train_features['feats'][x].shape[0]*.60)
            self.train_features['feats'][x] = self.train_features['feats'][x][:limit, :]
            self.train_features['labels'][x] = self.train_features['labels'][x][:limit]

        temp = torch.cat(self.train_features['labels'])
        for x in range(len(temp)):
            temp[x] = CLASS_MAPPING[int(temp[x])]

        self.train_labels = torch.nn.functional.one_hot(
            temp.type(torch.long)
        )#.float()

        self.train_features = torch.cat(self.train_features['feats'])

        # TODO characterization requires an predictor per subtask.
        self.feedback_obj = feedback_obj
        if self.feedback_obj:
            self.feedback_weight = data_params.get('feedback_weight', 1.0)

        self.dtype = getattr(torch, data_params.get('dtype', 'double'))

        # Parse the predictor params using docstr
        self.predictor = docstr_cap(predictor, True, True)
        self.predictor.label_enc = NominalDataEncoder.load(
            data_params['pred_known_map']
        )

        # OWHAR: FineTune, EVM, FINCH, CLIP Feedback Interpreter args
        #classlist = list(interpreter.pred_known_map.encoder)
        #del classlist[27]

        self.interpret_activity_feedback = hasattr(
            self.predictor,
            'feedback_interpreter',
        )

        logging.debug(
            'type(self.train_features) = %s',
            type(self.train_features),
        )
        logging.debug(
            'self.train_features.shape = %s',
            self.train_features.shape,
        )
        logging.debug(
            'type(self.train_labels) = %s',
            type(self.train_labels),
        )
        logging.debug(
            'self.train_labels.shape = %s',
            self.train_labels.shape,
        )

        # Fit the OWHAR fine tune model on the given train data, if not loaded
        self.predictor.fit(torch.utils.data.TensorDataset(
            self.train_features,
            self.train_labels,
        ))

        # Obtain detection threshold and kl threshold if None provided
        if data_params['thresh_set_data']:
            if self.predictor.novelty_detector.kl_threshold is not None:
                logging.warning(
                    'kl_threshold was already set, but finding from data',
                )
            test_features = torch.load(data_params['thresh_set_data'])

            logging.debug(
                "type(self.test_features['known']) = %s",
                type(test_features['known']),
            )
            logging.debug(
                "type(self.test_features['known'].shape) = %s",
                test_features['known'].shape,
            )
            logging.debug(
                "type(self.test_features['unknown']) = %s",
                type(test_features['unknown']),
            )
            logging.debug(
                "type(self.test_features['unknown'].shape) = %s",
                test_features['unknown'].shape,
            )

            # NOTE self.detection_threshold is NOT informed from val, atm
            self.predictor.novelty_detector.kl_threshold = self.find_kl_threshold(
                self.train_features,
                test_features['known'],
                test_features['unknown'],
                dtype=self.dtype,
            )

        logger.info("%s: Initialization complete", self.logging_header)

    def find_kl_threshold(
        self,
        ond_train,
        ond_val,
        ond_unknown,
        evm_batch_size=10000,
        batch_size=100,
        TA1_test_size=1024,
        number_of_evaluation=1,
        n_cpu=4,
        max_percentage_of_early=5.0,
        dtype=None
    ):
        """Copy pasted from kl_finder.py in Kitware's code to preserve
        functionality and meet deadlines.

        Args
        ----
        ond_val : torch.Tensor
            The known feature representations as given by ground truth
        ond_unknown : torch.Tensor
            The unknown feature representations as given by ground truth
        """
        if dtype is None:
            dtype = torch.double
        ond_train = ond_train[~torch.any(ond_train.isnan(), dim=1)]
        ond_val = ond_val[~torch.any(ond_val.isnan(), dim=1)]
        ond_unknown = ond_unknown[~torch.any(ond_unknown.isnan(), dim=1)]

        # TODO may have to add predictor arg to find kl thresholds of sub tasks
        p_train = []
        for i in tqdm(range(0, ond_train.shape[0], evm_batch_size)):
            t1 = self.predictor.known_probs(ond_train[i:i+evm_batch_size].to(dtype))
            p_train.append(t1)
        p_train = torch.cat(p_train).detach().cpu().numpy()

        p_val = self.predictor.known_probs(ond_val.to(dtype))
        p_unknown = self.predictor.known_probs(ond_unknown.to(dtype))
        p_val = p_val.detach().cpu().numpy()
        p_unknown = p_unknown.detach().cpu().numpy()

        def KL_Gaussian(mu, sigma, m, s):
          kl = np.log(s/sigma) + ( ( (sigma**2) + ( (mu-m) **2) ) / ( 2 * (s**2) ) ) - 0.5
          return kl

        mu_p_val = np.mean(p_val)
        sigma_p_val = np.std(p_val)
        mu_p_unknown = np.mean(p_unknown)
        sigma_p_unknown = np.std(p_unknown)

        logging.info("\nstart finding kl threshold")

        n_val = p_val.shape[0]

        def task(n):
            rng = np.random.default_rng(n)
            ind = rng.choice(n_val, size=batch_size, replace=False)
            p_batch = p_val[ind]
            mu_p_batch = np.mean(p_batch)
            sigma_p_batch = np.std(p_batch)
            return KL_Gaussian(
                mu=mu_p_batch,
                sigma=sigma_p_batch,
                m=1.0,
                s=np.sqrt(np.mean((p_train - 1.0)**2)),
            )

        average_of_known_batch = int(TA1_test_size / (batch_size * 2) )

        kl = np.zeros((number_of_evaluation, average_of_known_batch))


        for j in range(average_of_known_batch):
            arr = (number_of_evaluation * j) + np.arange(number_of_evaluation)
            kl[:, j] = task(arr)

        kl_evals = np.amax(kl, axis=1)

        kl_sorted = np.sort(kl_evals, kind='stable')

        min_percentage_not_early = 100.0 - max_percentage_of_early
        index = int(number_of_evaluation * (min_percentage_not_early/ 100)) + 1
        if index >= number_of_evaluation:
            index = -1

        return kl_sorted[index]

    @property
    def has_world_changed(self):
        return self.predictor.novelty_detector.has_world_changed

    @property
    def acc(self):
        return self.predictor.novelty_detector.accuracy

    @property
    def detection_threshold(self):
        return self.predictor.novelty_detector.detection_threshold

    def set_detection_threshold(self, value):
        self.predictor.novelty_detector.detection_threshold = value

    def world_detection(self, feature_dict, logit_dict, round_id=None):
        """
        Detect Change in World

        :param feature_dict (dict): Dictionary containing features
        :param logit_dict   (dict): Dictionary containing logits
        :param round_id     (int): Integer identifier for round

        :return string containing path to csv file with the results
        """
        logging_header = self._add_round_to_header(self.logging_header, round_id)
        logger.info("%s: Starting to detect change in world", logging_header)
        if round_id is None:
            result_path = f"wd_{self.session_id}_{self.test_id}.csv"
        else:
            result_path = f"wd_{self.session_id}_{self.test_id}_{round_id}.csv"

        image_names, FVs = zip(*feature_dict.items())
        class_map = map(self.predictor.known_probs,
                        map(lambda x: torch.Tensor(x).to(self.dtype), FVs))
        self.class_probabilities = torch.stack(list(class_map), axis=0)
        self.max_probabilities = torch.max(self.class_probabilities, axis=2)[0]

        self.round_feature_dict = feature_dict
        for x in self.round_feature_dict:
            self.round_feature_dict[x] = torch.Tensor(self.round_feature_dict[x])
        if round_id == 0:
            detections = torch.zeros(len(image_names))
        else:
            detections = self.predictor.novelty_detector.detect(
                self.max_probabilities,
                True,
                logger,
            )

        df = pd.DataFrame(
            zip(image_names, detections.tolist()),
            columns=['id', 'P_world_changed'],
        )

        logger.info(
            "%s: Number of samples in results %s",
            logging_header,
            df.shape,
        )
        df.to_csv(result_path, index=False, header=False, float_format='%.4f')
        logger.info("%s: Finished with change detection", logging_header)

        return result_path

    def classification(self, predictor, feature_dict, logit_dict, round_id=None):
        """Perform classification with the given predictor predictor."""
        # TODO The subtasks that consist of characterization are all
        # classification tasks. As was done for the HWR, make a
        raise NotImplementedError(
            'This is lowest priority as part of characterization.',
        )

    def novelty_classification(self, feature_dict, logit_dict, round_id=None):
        logging_header = self._add_round_to_header(self.logging_header, round_id)
        logger.info("%s: Starting to classify samples", logging_header)
        image_names, FVs = zip(*feature_dict.items())
        self.image_names = list(image_names)

        # TODO if world_detection not run first, have to run probs throu

        #if not hasattr(self, "class_probabilities"):
        #    class_map = map(self.predictor.known_probs, FVs)
        #    self.class_probabilities = torch.stack(list(class_map), axis=0)
        #    self.max_probabilities, _ = torch.max(self.class_probabilities, axis=2)
        #    self.round_feature_dict = feature_dict

        # TODO add switch to classify from TimeSformer, FineTune, or EVM


        # TODO absolutely Disgusting hack, sorry guys
        FVs = list(FVs)
        for x in range(len(FVs)):
            if not torch.is_tensor(FVs[x]):
                FVs[x] = torch.Tensor(FVs[x][1])
            else: FVs[x] = FVs[x][1]
        # End of disgusting hack, lol
        temp = torch.stack(FVs, axis=0).to(torch.device('cuda:0'))
        fine_tune_preds = self.predictor.fine_tune.predict(
            temp,
        )
        logger.info("Softmax scores: %s", torch.argmax(fine_tune_preds, dim=1))
        logger.info("Acc: %s", self.acc)
        pu = torch.zeros(fine_tune_preds.shape[0],).view(-1, 1).to(torch.device('cuda:0'))
        all_rows_tensor = torch.cat((fine_tune_preds, pu), 1)
        norm = torch.norm(all_rows_tensor, p=1, dim=1)
        normalized_tensor = all_rows_tensor/norm[:, None]

        df = pd.DataFrame(zip(image_names, *normalized_tensor.t().tolist()))
        if round_id is None:
            result_path = f"ncl_{self.session_id}_{self.test_id}.csv"
        else:
            result_path = f"ncl_{self.session_id}_{self.test_id}_{round_id}.csv"
        df.to_csv(result_path, index = False, header = False, float_format='%.4f')
        logger.info("%s: Finished classifying samples", logging_header)
        return result_path

    def _add_round_to_header(self, logger_header, round_id):
        if round_id is not None:
            logger_header += f" Round id: {round_id}"
        return logger_header

    def get_feedback(
        self,
        max_probabilities,
        detection_threshold,
        round_id,
        return_details=False,
    ):
        """Get feedback on those detected as novel and fill with those detected
        as non-novel otherwise.
        """
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

        feedback_df = self.feedback_obj.get_feedback(
            round_id,
            image_list,
            self.image_names,
        )

        if return_details:
            return feedback_df, len(known_image), half_income_per_batch
        return feedback_df

    def binary_novelty_feedback_adapt(
        self,
        max_probabilities,
        detection_threshold,
        round_id,
        feedback_df=None
    ):
        """Performs older version of binary novelty feedback adaptation.

        Returns
        -------
        float | float, pandas.DataFrame
            The new detection threshold is returned. The feedback data_frame is
            returned when feedback_df is not given as an argument.
        """
        data_frame, len_known, half_income_per_batch = self.get_feedback(
            max_probabilities,
            detection_threshold,
            round_id,
            return_details=True,
        )

        # Get known and unknown labels
        known_labels = data_frame["labels"][:len_known].to_numpy()
        unknown_labels = data_frame["labels"][len_known:].to_numpy()

        # Record known and unknown prediction performance
        known_pred_wrong = len(known_labels[known_labels == 88])
        unknown_pred_wrong = len(unknown_labels[unknown_labels != 88])

        logger.info("Known pred wrong: %d", known_pred_wrong)
        logger.info("Unknown pred wrong: %d", unknown_pred_wrong)

        unknown_acc = unknown_pred_wrong/half_income_per_batch
        known_acc = known_pred_wrong/half_income_per_batch

        if self.feedback_weight * unknown_acc > 0.0:
            detection_threshold -= self.feedback_weight*unknown_acc
        if self.feedback_weight * known_acc > 0.0:
            detection_threshold += self.feedback_weight*known_acc

        logger.info("New detection threshold is %f", detection_threshold)
        if feedback_df is None:
            return np.clip(detection_threshold, 0.0, 1.0), data_frame
        return np.clip(detection_threshold, 0.0, 1.0)

    def novelty_adaptation(self, round_id):
        """
        Novelty adaptation
        :param round_id: round id in a test
        """
        logger.info("Starting novelty_adaption: %s", round_id)
        if not self.has_world_changed:
            return

        # Adaptation w/o class size update:
        # Update the detection threshold and get FEEDBACK
        # NOTE rm because no binary novelty feedback given in m24.
        #feedback_df = self.binary_novelty_feedback_adapt(
        #    self.max_probabilities,
        #    self.detection_threshold,
        #    round_id,
        #)[1]

        # Check if should use feedback from classes.
        if not self.interpret_activity_feedback:
            return

        feedback_df = self.get_feedback(
            self.max_probabilities,
            self.detection_threshold,
            round_id,
        )

        # Get the feedback as label text and interpret the feedback
        raw_feedback_labels = feedback_df[self.feedback_columns].values

        feedback_labels = self.predictor.feedback_interpreter.interpret(
            raw_feedback_labels,
        )
        feedback_labels = torch.mean(feedback_labels,1)
        temp = torch.zeros((feedback_labels.shape[0], feedback_labels.shape[1] + 1))
        temp[:, :27] = feedback_labels[:, :27]
        temp[:, 28:] = feedback_labels[:, 27:]
        feedback_labels = temp
        features_arr = []
        for x in feedback_df['id']:
            features_arr.append(torch.Tensor(self.round_feature_dict[x])[1,:])
        # Combine the train data with the feedback data for update


        self.train_features = torch.cat([
            self.train_features,
            torch.stack(features_arr).to(self.train_features.device), # id values are indeices
        ])


        self.train_labels = torch.cat([self.train_labels.to(feedback_labels.device), feedback_labels])

        # Incremental fits on all prior train and saved feedback
        self.predictor.fit(torch.utils.data.TensorDataset(
            self.train_features,
            self.train_labels,
        ))

        # Handle the saving of results with updated predictor etc...
        class_map = map(self.predictor.known_probs, self.round_feature_dict.values())
        self.class_probabilities = torch.stack(list(class_map), axis=0)
        self.max_probabilities, _ = torch.max(self.class_probabilities, axis=2)

        # Adaptation given only novelty information after updating the fine
        # tune and the EVM.
        #self.set_detection_threshold(self.binary_novelty_feedback_adapt(
        #    self.max_probabilities,
        #    self.detection_threshold,
        #    round_id,
        #    feedback_df,
        #))

        # TODO adaptation w/ class size update (thus FINCH after deciding novel
        # classes exist and enough samples for them). This won't happen until
        # later difficulties of the DARPA eval.
