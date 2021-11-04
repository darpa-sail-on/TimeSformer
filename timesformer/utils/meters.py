# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Meters."""

import datetime
import numpy as np
import os
from collections import defaultdict, deque
import torch
from fvcore.common.timer import Timer
from sklearn.metrics import average_precision_score

import timesformer.utils.logging as logging
import timesformer.utils.metrics as metrics
import timesformer.utils.misc as misc

logger = logging.get_logger(__name__)


class TestMeter(object):
    """
    Perform the multi-view ensemble for testing: each video with an unique index
    will be sampled with multiple clips, and the predictions of the clips will
    be aggregated to produce the final prediction for the video.
    The accuracy is calculated with the given ground truth labels.
    """

    def __init__(
        self,
        num_videos,
        num_clips,
        num_cls,
        overall_iters,
        multi_label=False,
        ensemble_method="sum",
    ):
        """
        Construct tensors to store the predictions and labels. Expect to get
        num_clips predictions from each video, and calculate the metrics on
        num_videos videos.
        Args:
            num_videos (int): number of videos to test.
            num_clips (int): number of clips sampled from each video for
                aggregating the final prediction for the video.
            num_cls (int): number of classes for each prediction.
            overall_iters (int): overall iterations for testing.
            multi_label (bool): if True, use map as the metric.
            ensemble_method (str): method to perform the ensemble, options
                include "sum", and "max".
        """

        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.num_clips = num_clips
        self.overall_iters = overall_iters
        self.multi_label = multi_label
        self.ensemble_method = ensemble_method
        # Initialize tensors.
        self.video_preds = torch.zeros((num_videos, num_cls))
        if multi_label:
            self.video_preds -= 1e10

        self.video_labels = (
            torch.zeros((num_videos, num_cls))
            if multi_label
            else torch.zeros((num_videos)).long()
        )
        self.clip_count = torch.zeros((num_videos)).long()
        self.topk_accs = []
        self.stats = {}

        # Reset metric.
        self.reset()

    def reset(self):
        """
        Reset the metric.
        """
        self.clip_count.zero_()
        self.video_preds.zero_()
        if self.multi_label:
            self.video_preds -= 1e10
        self.video_labels.zero_()

    def update_stats(self, preds, labels, clip_ids):
        """
        Collect the predictions from the current batch and perform on-the-flight
        summation as ensemble.
        Args:
            preds (tensor): predictions from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (num_cls).
            labels (tensor): the corresponding labels of the current batch.
                Dimension is N.
            clip_ids (tensor): clip indexes of the current batch, dimension is
                N.
        """
        for ind in range(preds.shape[0]):
            vid_id = int(clip_ids[ind]) // self.num_clips
            if self.video_labels[vid_id].sum() > 0:
                assert torch.equal(
                    self.video_labels[vid_id].type(torch.FloatTensor),
                    labels[ind].type(torch.FloatTensor),
                )
            self.video_labels[vid_id] = labels[ind]
            if self.ensemble_method == "sum":
                self.video_preds[vid_id] += preds[ind]
            elif self.ensemble_method == "max":
                self.video_preds[vid_id] = torch.max(
                    self.video_preds[vid_id], preds[ind]
                )
            else:
                raise NotImplementedError(
                    "Ensemble Method {} is not supported".format(
                        self.ensemble_method
                    )
                )
            self.clip_count[vid_id] += 1

    def log_iter_stats(self, cur_iter):
        """
        Log the stats.
        Args:
            cur_iter (int): the current iteration of testing.
        """
        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "split": "test_iter",
            "cur_iter": "{}".format(cur_iter + 1),
            "eta": eta,
            "time_diff": self.iter_timer.seconds(),
        }
        logging.log_json_stats(stats)

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def finalize_metrics(self, ks=(1, 5)):
        """
        Calculate and log the final ensembled metrics.
        ks (tuple): list of top-k values for topk_accuracies. For example,
            ks = (1, 5) correspods to top-1 and top-5 accuracy.
        """
        if not all(self.clip_count == self.num_clips):
            logger.warning(
                "clip count {} ~= num clips {}".format(
                    ", ".join(
                        [
                            "{}: {}".format(i, k)
                            for i, k in enumerate(self.clip_count.tolist())
                        ]
                    ),
                    self.num_clips,
                )
            )

        self.stats = {"split": "test_final"}
        if self.multi_label:
            map = get_map(
                self.video_preds.cpu().numpy(), self.video_labels.cpu().numpy()
            )
            self.stats["map"] = map
        else:
            num_topks_correct = metrics.topks_correct(
                self.video_preds, self.video_labels, ks
            )
            topks = [
                (x / self.video_preds.size(0)) * 100.0
                for x in num_topks_correct
            ]

            assert len({len(ks), len(topks)}) == 1
            for k, topk in zip(ks, topks):
                self.stats["top{}_acc".format(k)] = "{:.{prec}f}".format(
                    topk, prec=2
                )
        logging.log_json_stats(self.stats)


class ScalarMeter(object):
    """
    A scalar meter uses a deque to track a series of scaler values with a given
    window size. It supports calculating the median and average values of the
    window, and also supports calculating the global average.
    """

    def __init__(self, window_size):
        """
        Args:
            window_size (int): size of the max length of the deque.
        """
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        """
        Reset the deque.
        """
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        """
        Add a new scalar value to the deque.
        """
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        """
        Calculate the current median value of the deque.
        """
        return np.median(self.deque)

    def get_win_avg(self):
        """
        Calculate the current average value of the deque.
        """
        return np.mean(self.deque)

    def get_global_avg(self):
        """
        Calculate the global mean value.
        """
        return self.total / self.count


class TrainMeter(object):
    """
    Measure training stats.
    """

    def __init__(self, epoch_iters, cfg):
        """
        Args:
            epoch_iters (int): the overall number of iterations of one epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.epoch_iters = epoch_iters
        self.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.lr = None
        # Current minibatch errors (smoothed over a window).
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top1_err_per = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top1_err_loc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top1_err_rel = ScalarMeter(cfg.LOG_PERIOD)
        # Number of misclassified examples.
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_top1_mis_per = 0
        self.num_top1_mis_loc = 0
        self.num_top1_mis_rel = 0
        self.num_samples = 0
        self.output_dir = cfg.OUTPUT_DIR
        self.extra_stats = {}
        self.extra_stats_total = {}
        self.log_period = cfg.LOG_PERIOD

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()
        self.loss_total = 0.0
        self.lr = None
        self.mb_top1_err.reset()
        self.mb_top5_err.reset()
        self.mb_top1_err_per.reset()
        self.mb_top1_err_loc.reset()
        self.mb_top1_err_rel.reset()
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_top1_mis_per = 0
        self.num_top1_mis_loc = 0
        self.num_top1_mis_rel = 0
        self.num_samples = 0

        for key in self.extra_stats.keys():
            self.extra_stats[key].reset()
            self.extra_stats_total[key] = 0.0

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats(self, top1_err, top5_err, top1_err_per, top1_err_loc, top1_err_rel, loss, lr, mb_size, stats={}):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            top1_err_per (float): top1 error rate for perspective.
            top1_err_loc (float): top1 error rate for locations.
            top1_err_rel (float): top1 error rate for relations.
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """
        self.loss.add_value(loss)
        self.lr = lr
        self.loss_total += loss * mb_size
        self.num_samples += mb_size

        if not self._cfg.DATA.MULTI_LABEL:
            # Current minibatch stats
            self.mb_top1_err.add_value(top1_err)
            self.mb_top5_err.add_value(top5_err)
            self.mb_top1_err_per.add_value(top1_err_per)
            self.mb_top1_err_loc.add_value(top1_err_loc)
            self.mb_top1_err_rel.add_value(top1_err_rel)
            # Aggregate stats
            self.num_top1_mis += top1_err * mb_size
            self.num_top5_mis += top5_err * mb_size
            self.num_top1_mis_per += top1_err_per * mb_size
            self.num_top1_mis_loc += top1_err_loc * mb_size
            self.num_top1_mis_rel += top1_err_rel * mb_size

        for key in stats.keys():
            if key not in self.extra_stats:
                self.extra_stats[key] = ScalarMeter(self.log_period)
                self.extra_stats_total[key] = 0.0
            self.extra_stats[key].add_value(stats[key])
            self.extra_stats_total[key] += stats[key] * mb_size

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "train_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "dt": self.iter_timer.seconds(),
            "dt_data": self.data_timer.seconds(),
            "dt_net": self.net_timer.seconds(),
            "eta": eta,
            "loss": self.loss.get_win_median(),
            "lr": self.lr,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
        }
        if not self._cfg.DATA.MULTI_LABEL:
            stats["top1_err"] = self.mb_top1_err.get_win_median()
            stats["top5_err"] = self.mb_top5_err.get_win_median()
            stats["top1_err_per"] = self.mb_top1_err_per.get_win_median()
            stats["top1_err_loc"] = self.mb_top1_err_loc.get_win_median()
            stats["top1_err_rel"] = self.mb_top1_err_rel.get_win_median()
        for key in self.extra_stats.keys():
            stats[key] = self.extra_stats_total[key] / self.num_samples
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch + 1) * self.epoch_iters
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "dt": self.iter_timer.seconds(),
            "dt_data": self.data_timer.seconds(),
            "dt_net": self.net_timer.seconds(),
            "eta": eta,
            "lr": self.lr,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
        }
        if not self._cfg.DATA.MULTI_LABEL:
            top1_err = self.num_top1_mis / self.num_samples
            top5_err = self.num_top5_mis / self.num_samples
            avg_loss = self.loss_total / self.num_samples
            top1_err_per = self.num_top1_mis_per / self.num_samples
            top1_err_loc = self.num_top1_mis_loc / self.num_samples
            top1_err_rel = self.num_top1_mis_rel / self.num_samples
            stats["top1_err"] = top1_err
            stats["top5_err"] = top5_err
            stats["top1_err_per"] = top1_err_per
            stats["top1_err_loc"] = top1_err_loc
            stats["top1_err_rel"] = top1_err_rel
            stats["loss"] = avg_loss
        for key in self.extra_stats.keys():
            stats[key] = self.extra_stats_total[key] / self.num_samples
        logging.log_json_stats(stats)


class ValMeter(object):
    """
    Measures validation stats.
    """

    def __init__(self, max_iter, cfg):
        """
        Args:
            max_iter (int): the max number of iteration of the current epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.max_iter = max_iter
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        # Current minibatch errors (smoothed over a window).
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top1_err_per = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top1_err_loc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top1_err_rel = ScalarMeter(cfg.LOG_PERIOD)
        # Min errors (over the full val set).
        self.min_top1_err = 100.0
        self.min_top5_err = 100.0
        self.min_top1_err_per = 100.0
        self.min_top1_err_loc = 100.0
        self.min_top1_err_rel = 100.0
        # Number of misclassified examples.
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_top1_mis_per = 0
        self.num_top1_mis_loc = 0
        self.num_top1_mis_rel = 0
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []
        self.all_preds_per = []
        self.all_perspectives = []
        self.all_preds_loc = []
        self.all_locations = []
        self.all_preds_rel = []
        self.all_relations = []
        self.output_dir = cfg.OUTPUT_DIR
        self.extra_stats = {}
        self.extra_stats_total = {}
        self.log_period = cfg.LOG_PERIOD

    def reset(self):
        """
        Reset the Meter.
        """
        self.iter_timer.reset()
        self.mb_top1_err.reset()
        self.mb_top5_err.reset()
        self.mb_top1_err_per.reset()
        self.mb_top1_err_loc.reset()
        self.mb_top1_err_rel.reset()
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_top1_mis_per = 0
        self.num_top1_mis_loc = 0
        self.num_top1_mis_rel = 0
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []
        self.all_preds_per = []
        self.all_perspectives = []
        self.all_preds_loc = []
        self.all_locations = []
        self.all_preds_rel = []
        self.all_relations = []

        for key in self.extra_stats.keys():
            self.extra_stats[key].reset()
            self.extra_stats_total[key] = 0.0

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats(self, top1_err, top5_err, top1_err_per, top1_err_loc, top1_err_rel, mb_size, stats={}):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            top1_err_per (float): top1 error rate for perspective.
            top1_err_loc (float): top1 error rate for locations.
            top1_err_rel (float): top1 error rate for relations.
            mb_size (int): mini batch size.
        """
        self.mb_top1_err.add_value(top1_err)
        self.mb_top5_err.add_value(top5_err)
        self.mb_top1_err_per.add_value(top1_err_per)
        self.mb_top1_err_loc.add_value(top1_err_loc)
        self.mb_top1_err_rel.add_value(top1_err_rel)
        self.num_top1_mis += top1_err * mb_size
        self.num_top5_mis += top5_err * mb_size
        self.num_top1_mis_per += top1_err_per * mb_size
        self.num_top1_mis_loc += top1_err_loc * mb_size
        self.num_top1_mis_rel += top1_err_rel * mb_size
        self.num_samples += mb_size

        for key in stats.keys():
            if key not in self.extra_stats:
                self.extra_stats[key] = ScalarMeter(self.log_period)
                self.extra_stats_total[key] = 0.0
            self.extra_stats[key].add_value(stats[key])
            self.extra_stats_total[key] += stats[key] * mb_size


    def update_predictions(self, preds, labels, preds_per, perspectives, preds_loc, locations, preds_rel, relations):
        """
        Update predictions and labels.
        Args:
            preds (tensor): model output predictions.
            labels (tensor): labels.
            preds_per (tensor): model output predictions of perspectives.
            perspectives (tensor): labels of perspectives.
            preds_loc (tensor): model output predictions of locations.
            locations (tensor): labels of locations.
            preds_rel (tensor): model output predictions of relations.
            relations (tensor): labels of relations.
        """
        # TODO: merge update_prediction with update_stats.
        self.all_preds.append(preds)
        self.all_labels.append(labels)
        self.all_preds_per.append(preds_per)
        self.all_perspectives.append(perspectives)
        self.all_preds_loc.append(preds_loc)
        self.all_locations.append(locations)
        self.all_preds_rel.append(preds_rel)
        self.all_relations.append(relations)

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.max_iter - cur_iter - 1)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "val_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
        }
        if not self._cfg.DATA.MULTI_LABEL:
            stats["top1_err"] = self.mb_top1_err.get_win_median()
            stats["top5_err"] = self.mb_top5_err.get_win_median()
            stats["top1_err_per"] = self.mb_top1_err_per.get_win_median()
            stats["top1_err_loc"] = self.mb_top1_err_loc.get_win_median()
            stats["top1_err_rel"] = self.mb_top1_err_rel.get_win_median()
        for key in self.extra_stats.keys():
            stats[key] = self.extra_stats[key].get_win_median()
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        stats = {
            "_type": "val_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
        }
        if self._cfg.DATA.MULTI_LABEL:
            stats["map"] = get_map(
                torch.cat(self.all_preds).cpu().numpy(),
                torch.cat(self.all_labels).cpu().numpy(),
            )
        else:
            top1_err = self.num_top1_mis / self.num_samples
            top5_err = self.num_top5_mis / self.num_samples
            top1_err_per = self.num_top1_mis_per / self.num_samples
            top1_err_loc = self.num_top1_mis_loc / self.num_samples
            top1_err_rel = self.num_top1_mis_rel / self.num_samples
            self.min_top1_err = min(self.min_top1_err, top1_err)
            self.min_top5_err = min(self.min_top5_err, top5_err)
            self.min_top1_err_per = min(self.min_top1_err_per, top1_err_per)
            self.min_top1_err_loc = min(self.min_top1_err_loc, top1_err_loc)
            self.min_top1_err_rel = min(self.min_top1_err_rel, top1_err_rel)

            stats["top1_err"] = top1_err
            stats["top5_err"] = top5_err
            stats["top1_err_per"] = top1_err_per
            stats["top1_err_loc"] = top1_err_loc
            stats["top1_err_rel"] = top1_err_rel
            stats["min_top1_err"] = self.min_top1_err
            stats["min_top5_err"] = self.min_top5_err
            stats["min_top1_err_per"] = self.min_top1_err_per
            stats["min_top1_err_loc"] = self.min_top1_err_loc
            stats["min_top1_err_rel"] = self.min_top1_err_rel

        for key in self.extra_stats.keys():
            stats[key] = self.extra_stats_total[key] / self.num_samples

        logging.log_json_stats(stats)


def get_map(preds, labels):
    """
    Compute mAP for multi-label case.
    Args:
        preds (numpy tensor): num_examples x num_classes.
        labels (numpy tensor): num_examples x num_classes.
    Returns:
        mean_ap (int): final mAP score.
    """

    logger.info("Getting mAP for {} examples".format(preds.shape[0]))

    preds = preds[:, ~(np.all(labels == 0, axis=0))]
    labels = labels[:, ~(np.all(labels == 0, axis=0))]
    aps = [0]
    try:
        aps = average_precision_score(labels, preds, average=None)
    except ValueError:
        print(
            "Average precision requires a sufficient number of samples \
            in a batch which are missing in this sample."
        )

    mean_ap = np.mean(aps)
    return mean_ap
