# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified by Dawei Du, Kitware, Inc.
"""Multi-view train a video evm model."""

import numpy as np
import os
import torch
import pkbar
import cv2
from timesformer.utils.parser import load_config, parse_args
import timesformer.utils.checkpoint as cu
import timesformer.utils.distributed as du
import timesformer.utils.logging as logging
from timesformer.datasets import loader
from timesformer.models import build_model
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics.cluster import normalized_mutual_info_score
import evm_based_novelty_detector.MultipleEVM as MEVM
logger = logging.get_logger(__name__)

tail_base = 8000
cover_thre = 0.8
dist_thre = 0.4
number_of_known_classes = 29

@torch.no_grad()
def perform_test(model, train_loader, known_loader, unknown_loader, cfg):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        train_loader, known_loader, unknown_loader (loader): video loader.
        model (model): the pretrained video model to test.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Enable eval mode.
    model.eval()
    # features of train set
    # if there exist saved features from training samples
    if os.path.exists("evm_models/timesformer_train_feats.bin"):
        with open("evm_models/timesformer_train_feats.bin", "rb") as train_data:
            train_feats = torch.load(train_data)    
        evm_feats, evm_labels = train_feats['feats'], train_feats['labels']
        mevm = MEVM.MultipleEVM(tailsize=tail_base, cover_threshold=cover_thre, distance_function='cosine', distance_multiplier=dist_thre, device='cuda')
        mevm.train(evm_feats, evm_labels)
    else:
        n_train = len(train_loader.dataset)
        bar = pkbar.Pbar(name='Extracting training features: ', target=len(train_loader))
        extracted_feats = -1*torch.ones(n_train, 768+5) # feat dim + label dim
        evm_feats, evm_labels, evm_perspectives, evm_locations, evm_relations = [], [], [], [], []
        j = 0
        for cur_iter, (inputs, labels, perspectives, locations, relations, video_idx, meta) in enumerate(train_loader):
            bar.update(cur_iter)
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            # Transfer the data to the current GPU device.
            labels = labels.cuda()
            perspectives = perspectives.cuda()
            locations = locations.cuda()
            relations = torch.stack(relations).t().cuda()
            video_idx = video_idx.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)

            # Perform the forward pass.
            preds, preds_per, preds_loc, preds_rel, feats = model(inputs)
            if torch.sum(torch.sum(feats,1)==0) > 0:
                pdb.set_trace()
            # Gather all the predictions across all the devices to perform ensemble.
            if cfg.NUM_GPUS > 1:
                preds, preds_per, preds_loc, preds_rel, feats, labels, perspectives, locations, relations, video_idx = du.all_gather([preds, preds_per, preds_loc, preds_rel, feats, labels, perspectives, locations, relations, video_idx])
            preds = preds.cpu()
            preds_per = preds_per.cpu()
            preds_loc = preds_loc.cpu()
            preds_rel = preds_rel.cpu()
            feats = feats.cpu()
            labels = labels.cpu()
            perspectives = perspectives.cpu()
            locations = locations.cpu()
            relations = relations.cpu()
            video_idx = video_idx.cpu()

            extracted_feats[j:(j+inputs.size(0)), :5] = torch.cat((labels.unsqueeze(1), perspectives.unsqueeze(1), locations.unsqueeze(1), relations), 1)
            extracted_feats[j:(j+inputs.size(0)), 5:] = feats
            j = j + inputs.size(0)

        for k in range(number_of_known_classes):
            cur_feat = extracted_feats[extracted_feats[:,0]==k]
            evm_labels.append(cur_feat[:,0].detach().clone())
            evm_feats.append(cur_feat[:,5:].detach().clone())

        # train the evm model
        with open("evm_models/timesformer_train_feats.bin", "wb") as output:
            train_feats = {'feats':evm_feats, 'labels':evm_labels}
            torch.save(train_feats, output)
        mevm = MEVM.MultipleEVM(tailsize=tail_base, cover_threshold=cover_thre, distance_function='cosine', distance_multiplier=dist_thre, device='cuda')
        mevm.train(evm_feats, evm_labels)
        mevm.save('evm_models/timesformer_feats_evm.hdf5') 

    # if there exist saved features from testing samples
    n_views = cfg.TEST.NUM_ENSEMBLE_VIEWS*cfg.TEST.NUM_SPATIAL_CROPS
    if os.path.exists("evm_models/timesformer_test_feats.bin"):
        with open("evm_models/timesformer_test_feats.bin", "rb") as test_data:
            test_feats = torch.load(test_data)    
        known_feats, unknown_feats = test_feats['known'], test_feats['unknown']
        mprob, mevm_index = mevm.max_probabilities(known_feats)
        evm_known = [1 - np.mean(mprob[k:k+n_views]) for k in range(0,len(mprob),n_views)] 
        mprob, mevm_index = mevm.max_probabilities(unknown_feats)
        evm_unknown = [1 - np.mean(mprob[k:k+n_views]) for k in range(0,len(mprob),n_views)] 
    else:
        n_known = len(known_loader.dataset)
        bar = pkbar.Pbar(name='Extracting known testing features: ', target=len(known_loader))
        for cur_iter, (inputs, labels, perspectives, locations, relations, video_idx, meta) in enumerate(known_loader):
            bar.update(cur_iter)
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            # Transfer the data to the current GPU device.
            labels = labels.cuda()
            perspectives = perspectives.cuda()
            locations = locations.cuda()
            relations = torch.stack(relations).t().cuda()

            video_idx = video_idx.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)

            # Perform the forward pass.
            preds, preds_per, preds_loc, preds_rel, feats = model(inputs)
            # Gather all the predictions across all the devices to perform ensemble.
            if cfg.NUM_GPUS > 1:
                preds, preds_per, preds_loc, preds_rel, feats, labels, perspectives, locations, relations, video_idx = du.all_gather([preds, preds_per, preds_loc, preds_rel, feats, labels, perspectives, locations, relations, video_idx])
            preds = preds.cpu()
            preds_per = preds_per.cpu()
            preds_loc = preds_loc.cpu()
            preds_rel = preds_rel.cpu()
            feats = feats.cpu()
            labels = labels.cpu()
            perspectives = perspectives.cpu()
            locations = locations.cpu()
            relations = relations.cpu()
            video_idx = video_idx.cpu()
            if cur_iter == 0:
                known_test_feats = feats
            else:
                known_test_feats = torch.cat((known_test_feats, feats), 0)
        
        mprob, mevm_index = mevm.max_probabilities(known_test_feats)
        evm_known = [1 - np.mean(mprob[k:k+n_views]) for k in range(0,len(mprob),n_views)] 


        n_unknown = len(unknown_loader.dataset)
        bar = pkbar.Pbar(name='Extracting unknown testing features: ', target=len(unknown_loader))
        for cur_iter, (inputs, labels, perspectives, locations, relations, video_idx, meta) in enumerate(unknown_loader):
            bar.update(cur_iter)
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            # Transfer the data to the current GPU device.
            labels = labels.cuda()
            perspectives = perspectives.cuda()
            locations = locations.cuda()
            relations = torch.stack(relations).t().cuda()

            video_idx = video_idx.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)

            # Perform the forward pass.
            preds, preds_per, preds_loc, preds_rel, feats = model(inputs)
            # Gather all the predictions across all the devices to perform ensemble.
            if cfg.NUM_GPUS > 1:
                preds, preds_per, preds_loc, preds_rel, feats, labels, perspectives, locations, relations, video_idx = du.all_gather([preds, preds_per, preds_loc, preds_rel, feats, labels, perspectives, locations, relations, video_idx])
            preds = preds.cpu()
            preds_per = preds_per.cpu()
            preds_loc = preds_loc.cpu()
            preds_rel = preds_rel.cpu()
            feats = feats.cpu()
            labels = labels.cpu()
            perspectives = perspectives.cpu()
            locations = locations.cpu()
            relations = relations.cpu()
            video_idx = video_idx.cpu()
            if cur_iter == 0:
                unknown_test_feats = feats
            else:
                unknown_test_feats = torch.cat((unknown_test_feats, feats), 0)
        mprob, mevm_index = mevm.max_probabilities(unknown_test_feats)
        evm_unknown = [1 - np.mean(mprob[k:k+n_views]) for k in range(0,len(mprob),n_views)] 

        # save features of both known and unknown classes in testing set
        with open("evm_models/timesformer_test_feats.bin", "wb") as output:
            test_feats = {'known':known_test_feats, 'unknown':unknown_test_feats}
            torch.save(test_feats, output)

    # Calculate scores
    auc_sc, ap_sc, f1_sc, nmi_sc, fpr, tpr, precision, recall = voc_eval(evm_known, evm_unknown)
    print('known score: {:.4f}/unknown score: {:.4f}'.format(np.mean(evm_known), np.mean(evm_unknown)))
    print('AUC={:.4f}/AP={:.4f}/NMI={:.4f}'.format(auc_sc, ap_sc, nmi_sc))

def voc_eval(known_sc, unknown_sc):
    all_sc = np.concatenate((known_sc,unknown_sc))
    label = np.concatenate((np.zeros(len(known_sc)),np.ones(len(unknown_sc))))
    #calculate the auc
    fpr,tpr,threshold = roc_curve(label, all_sc)
    auc_sc = auc(fpr,tpr)
    #calculate the ap
    precision,recall,thresholds = precision_recall_curve(label, all_sc)
    ap_sc = average_precision_score(label, all_sc)
    f1 = 2*(precision*recall)/(precision+recall)
    f1_sc = max(f1)
    idx = np.where(f1==f1_sc)
    #calculate the NMI
    nmi_sc = normalized_mutual_info_score(label, all_sc)

    return auc_sc, ap_sc, idx, nmi_sc, fpr, tpr, precision, recall

def main():
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    args = parse_args()
    if args.num_shards > 1:
       args.output_dir = str(args.job_dir)
    cfg = load_config(args)

    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    cu.load_test_checkpoint(cfg, model)

    # Create video training and testing loaders.
    train_loader = loader.construct_loader(cfg, "training")
    known_loader = loader.construct_loader(cfg, "known")
    unknown_loader = loader.construct_loader(cfg, "unknown")

    # # Perform multi-view test on the entire dataset.
    perform_test(model, train_loader, known_loader, unknown_loader, cfg)

if __name__ == "__main__":
    main()
