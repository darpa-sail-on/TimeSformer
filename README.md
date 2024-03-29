# TimeSformer

This repository is a fork of [TimeSformer](https://github.com/facebookresearch/TimeSformer).
Please refer to the original repository for replicating their work described in
their [paper](https://arxiv.org/pdf/2102.05095.pdf).

# Prerequisites
1. M24 Activity Videos: Available internally on Veydrus and Tellurak under `/data/datasets/m24-activity/all_vids`
2. [Poetry](https://python-poetry.org/docs/#installation)
3. [Python 3.8](https://www.python.org/downloads/release/python-380/)
4. [CUDA 10.2](https://developer.nvidia.com/cuda-10.2-download-archive)

## Using Poetry
1. Install poetry based on the instructions provided in their [documentation](https://python-poetry.org/docs/#installation).
2. Clone timesformer along with additional dependencies using:
   ```
    git clone git@github.com:darpa-sail-on/TimeSformer.git
    git clone git@github.com/darpa-sail-on/ND-Activity-Recognition-Feeback.git
   ```
   This would create TimeSformer, and ND-Activity-Recognition-Feeback directories in your working directory
3. Create a virtual environment and install the components using the following commands:
   ```
    cd TimeSformer
    poetry install
    poetry run pip install ../ND-Activity-Recognition-Feeback
    poetry shell
   ```

## Using Conda
1. Create a conda virtual environment and activate it:
    ```
    conda create -n timesformer python=3.8 -y
    source activate timesformer
    ```
2. Install the following packages:

    - torchvision: `pip install torchvision` or `conda install torchvision -c pytorch`
    - [fvcore](https://github.com/facebookresearch/fvcore/): `pip install 'git+https://github.com/facebookresearch/fvcore'`
    - simplejson: `pip install simplejson`
    - einops: `pip install einops`
    - timm: `pip install timm`
    - PyAV: `conda install av -c conda-forge`
    - psutil: `pip install psutil`
    - scikit-learn: `pip install scikit-learn`
    - OpenCV: `pip install opencv-python`
    - tensorboard: `pip install tensorboard`
    - sail-on-client: `pip install sail-on-client`
3. Build the TimeSformer codebase by running:
    ```
    git clone git@github.com:darpa-sail-on/TimeSformer.git
    cd TimeSformer
    python pip install .
    ```
4. Install Additional dependencies in your working directory using:
   ```
    git clone git@github.com/darpa-sail-on/ND-Activity-Recognition-Feeback.git
    pip install ../ND-Activity-Recognition-Feeback
   ```
   Note: Working directory in this case is the directory where timesformer is cloned.

# Usage

## Dry Run

1. Download the models (*.pyth and *.bin) available in [google drive](https://drive.google.com/drive/folders/1OG4gSL6luN0dWA_5L74vGAIrVi7HoGgP?usp=sharing).
2. If you are using the files on your machine use the following command
   ```
     HYDRA_FULL_ERROR=1 sail-on-client --config-dir <your working directory>/TimeSformer/configs/ \
                                       --config-name dry_run_local \
                                       test_root=<your working directory>/TimeSformer/data \
                                       protocol.smqtk.config.dataset_root=<root directory for videos from first prerequisites> \
                                       model_root=<root directory where models were downloaded from step 1 and 2> \
                                       algorithms@protocol.smqtk.config.algorithms=[timesformer_base] \
                                       protocol.smqtk.config.test_ids=[OND.0.10001.6438158]
    ```


## M-30 Evaluation

### Feature Extraction

1. Download the models (*.pyth and *.hdf5) available in [google drive](https://drive.google.com/drive/folders/1OG4gSL6luN0dWA_5L74vGAIrVi7HoGgP?usp=sharing).
2. If you are using the evaluation use the following command
   ```
   HYDRA_FULL_ERROR=1 sail-on-client --config-dir <your working directory>/TimeSformer/configs/ \
                                     --config-name feature_extraction_par \
                                     server_url=<url for server> \
                                     protocol.smqtk.config.dataset_root=<root directory for videos> \
                                     model_root=<root directory for models> \
                                     protocol.smqtk.config.feature_dir=<root directory where features are saved> \
                                     algorithms@protocol.smqtk.config.algorithms=[timesformer_base] \
                                     protocol.smqtk.config.test_ids=[<comma seperated list of test ids>]
   ```
3. If you are using the files on your machine use the following command
   ```
   HYDRA_FULL_ERROR=1 sail-on-client --config-dir <your working directory>/TimeSformer/configs/ \
                                     --config-name feature_extraction_local \
                                     test_root=<root directory for tests> \
                                     protocol.smqtk.config.dataset_root=<root directory for videos> \
                                     model_root=<root directory for models> \
                                     protocol.smqtk.config.feature_dir=<root directory where features are saved> \
                                     algorithms@protocol.smqtk.config.algorithms=[timesformer_base] \
                                     protocol.smqtk.config.test_ids=[<comma seperated list of test ids>]
   ```
4. [Optional] To use slurm with the feature extraction use the following command
   ```
    HYDRA_FULL_ERROR=1 sail-on-client --config-dir configs/ \
                                      --config-name feature_extraction_local \
                                      --multirun protocol.smqtk.config.test_ids=["OND.9.99999.0"],["OND.9.99999.1"],["OND.9.99999.2"],["OND.9.99999.3"],["OND.9.99999.4"],["OND.9.99999.5"],["OND.9.99999.6"],["OND.9.99999.7"] \
                                      test_root=/data/datasets/m24-activity-test/feature_extraction_tests \
                                      protocol.smqtk.config.dataset_root=/data/datasets/m24-activity-test/1115_2021 \
                                      model_root=/home/khq.kitware.com/ameya.shringi/models/timesformer-m24 \
                                      protocol.smqtk.config.feature_dir=/home/khq.kitware.com/ameya.shringi/features/timesformer-m24 \
                                      algorithms@protocol.smqtk.config.algorithms=[timesformer_base] \
                                      hydra/launcher=veydrus \
    ```
5. Aggregate the features obtained from feature extraction step using
   ```
    aggregate-features --feature-paths <space seperated path to feature files generated in previous step> \
                       --output-path <path to a directory where aggregated features are saved>/timesformer_features.pkl
   ```


### Given Detection

1. Download the models (*.pyth and *.hdf5) available in [google drive](https://drive.google.com/drive/folders/1OG4gSL6luN0dWA_5L74vGAIrVi7HoGgP?usp=sharing).
2. With the evaluation server use the following command
   ```
     HYDRA_FULL_ERROR=1 sail-on-client --config-dir configs/ \
                                       --config-name given_detection_par \
                                       server_url=<url for server> \
                                       model_root=<root directory where models are stored> \
                                       protocol.smqtk.config.feature_dir=<root directory where features are stored> \
                                       protocol.smqtk.config.dataset_root=<root directory of vidoes> \
                                       algorithms@protocol.smqtk.config.algorithms=[timesformer_rd] \
                                       protocol.smqtk.config.test_ids=[<comma seperated test ids>]
   ```

3. With files on the machine using the following command
   ```
     HYDRA_FULL_ERROR=1 sail-on-client --config-dir configs/ \
                                       --config-name given_detection_local \
                                       test_root=<root directory with tests> \
                                       model_root=<root directory where models are stored> \
                                       protocol.smqtk.config.feature_dir=<root directory with features> \
                                       protocol.smqtk.config.dataset_root=<root directory with videos> \
                                       algorithms@protocol.smqtk.config.algorithms=[timesformer_rd]
                                       protocol.smqtk.config.test_ids=[<comma seperate test ids>]
   ```

### Given Detection With Detection Feedback

1. Download the models (*.pyth, *.bin and *.hdf5) available in [google drive](https://drive.google.com/drive/folders/1OG4gSL6luN0dWA_5L74vGAIrVi7HoGgP?usp=sharing).
2. With the evaluation server use the following command
   ```
     HYDRA_FULL_ERROR=1 sail-on-client --config-dir configs/ \
                                       --config-name given_detection_detection_feedback_par \
                                       server_url=<url for server> \
                                       model_root=<root directory where models are stored> \
                                       protocol.smqtk.config.feature_dir=<root directory where features are stored> \
                                       protocol.smqtk.config.dataset_root=<root directory of vidoes> \
                                       algorithms@protocol.smqtk.config.algorithms=[timesformer_detection_feedback] \
                                       protocol.smqtk.config.test_ids=[<comma seperated test ids>]
   ```
3. With files on the machine using the following command
   ```
     HYDRA_FULL_ERROR=1 sail-on-client --config-dir configs/ \
                                       --config-name given_detection_detection_feedback_local \
                                       test_root=<root directory with tests> \
                                       model_root=<root directory where models are stored> \
                                       protocol.smqtk.config.feature_dir=<root directory with features> \
                                       protocol.smqtk.config.dataset_root=<root directory with videos> \
                                       algorithms@protocol.smqtk.config.algorithms=[timesformer_detection_feedback]
                                       protocol.smqtk.config.test_ids=[<comma seperate test ids>]
   ```

### System Detection

1. Download the models (*.pyth and *.hdf5) available in [google drive](https://drive.google.com/drive/folders/1OG4gSL6luN0dWA_5L74vGAIrVi7HoGgP?usp=sharing).
2. With the evaluation server use the following command
   ```
     HYDRA_FULL_ERROR=1 sail-on-client --config-dir configs/ \
                                       --config-name system_detection_par \
                                       server_url=<url for server> \
                                       model_root=<root directory where models are stored> \
                                       protocol.smqtk.config.feature_dir=<root directory where features are stored> \
                                       protocol.smqtk.config.dataset_root=<root directory of vidoes> \
                                       algorithms@protocol.smqtk.config.algorithms=[timesformer_base] \
                                       protocol.smqtk.config.test_ids=[<comma seperated test ids>]
   ```
3. With files on the machine using the following command
   ```
     HYDRA_FULL_ERROR=1 sail-on-client --config-dir configs/ \
                                       --config-name system_detection_par \
                                       test_root=<root directory with tests> \
                                       model_root=<root directory where models are stored> \
                                       protocol.smqtk.config.feature_dir=<root directory with features> \
                                       protocol.smqtk.config.dataset_root=<root directory with videos> \
                                       algorithms@protocol.smqtk.config.algorithms=[timesformer_base]
                                       protocol.smqtk.config.test_ids=[<comma seperate test ids>]
   ```


### System Detection With Classification Feedback
1. Download the models (*.pyth, *.bin and *.hdf5) available in [google drive](https://drive.google.com/drive/folders/1OG4gSL6luN0dWA_5L74vGAIrVi7HoGgP?usp=sharing).
2. Download additional [data](https://drive.google.com/drive/folders/1YZ5fO14bnfJYVE6nXMaaJGpccWdx8N4u?usp=sharing)
   and [model](https://drive.google.com/drive/folders/1exAjXZ6uBrkBr904pjXYxU8BAYM7ONOF?usp=sharing) folder
   and place then in the model_root where the network and evm model is present. The `model_root` would have the
   following structure after this
   ```
      ├── checkpoint_epoch_00015.pyth
      ├── data
      │   ├── clip
      │   │   └── k700_templates.txt
      │   ├── kinetics
      │   │   └── kinetics600
      │   │       ├── clip_sorted_label_text_encs.pt
      │   │       └── unique_sorted_actions.csv
      │   └── par
      │       ├── clip_par_ontology_idx_sorted_label_text_encs.pt
      │       └── ordered_par_classes.txt
      ├── models
      │   └── clip
      │       └── clip_ViT-B_32.pt
      ├── timesformer_feats_evm.hdf5
      ├── timesformer_test_feats.bin
      └── timesformer_train_feats.bin
   ```
3. With the evaluation server use the following command
   ```
     HYDRA_FULL_ERROR=1 sail-on-client --config-dir configs/ \
                                       --config-name system_detection_classification_feedback_par \
                                       server_url=<url for server> \
                                       model_root=<root directory where models are stored> \
                                       protocol.smqtk.config.feature_dir=<root directory where features are stored> \
                                       protocol.smqtk.config.dataset_root=<root directory of vidoes> \
                                       algorithms@protocol.smqtk.config.algorithms=[timesformer_feedback] \
                                       protocol.smqtk.config.test_ids=[<comma seperated test ids>]
   ```
4. With files on the machine using the following command
   ```
     HYDRA_FULL_ERROR=1 sail-on-client --config-dir configs/ \
                                       --config-name system_detection_classification_feedback_local \
                                       test_root=<root directory with tests> \
                                       protocol.smqtk.config.feature_dir=<root directory with features> \
                                       protocol.smqtk.config.dataset_root=<root directory with videos> \
                                       algorithms@protocol.smqtk.config.algorithms=[timesformer_feedback]
                                       protocol.smqtk.config.test_ids=[<comma seperate test ids>]

### Given Detection With Characterization

1. Download the network model (pyth from) from [google drive](https://drive.google.com/file/d/16TCu-9rmifx4RnGIG-UoVGRJ76SGBUTV/view?usp=sharing)
2. Download all the evm models (HDF5 Files) from [google drive](https://drive.google.com/drive/folders/1WrAS-Ip-edvFruypVvF0uraaaeIV1FiV?usp=sharing)
3. With the evaluation server use the following command
   ```
     HYDRA_FULL_ERROR=1 sail-on-client --config-dir configs/ \
                                       --config-name given_detection_par \
                                       server_url=<url for server> \
                                       model_root=<root directory where models are stored> \
                                       protocol.smqtk.config.feature_dir=<root directory where features are stored> \
                                       protocol.smqtk.config.dataset_root=<root directory of vidoes> \
                                       algorithms@protocol.smqtk.config.algorithms=[timesformer_characterization] \
                                       protocol.smqtk.config.test_ids=[<comma seperated test ids>]
   ```

4. With files on the machine using the following command
   ```
     HYDRA_FULL_ERROR=1 sail-on-client --config-dir configs/ \
                                       --config-name given_detection_local \
                                       test_root=<root directory with tests> \
                                       model_root=<root directory where models are stored> \
                                       protocol.smqtk.config.feature_dir=<root directory with features> \
                                       protocol.smqtk.config.dataset_root=<root directory with videos> \
                                       algorithms@protocol.smqtk.config.algorithms=[timesformer_characterization]
                                       protocol.smqtk.config.test_ids=[<comma seperate test ids>]
   ```

Note: Instructions for running M24 Agent is available in [M24.md](M24.md).

# Training Network

## Dataset Preparation

Please use the dataset preparation instructions provided in [DATASET.md](timesformer/datasets/DATASET.md).

## Training the Default TimeSformer

Training the default TimeSformer that uses divided space-time attention, and operates on 8-frame clips cropped at 224x224 spatial resolution, can be done using the following command:

```
python tools/run_net.py \
  --cfg configs/Kinetics/TimeSformer_divST_8x32_224.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  NUM_GPUS 8 \
  TRAIN.BATCH_SIZE 8 \
```
You may need to pass location of your dataset in the command line by adding `DATA.PATH_TO_DATA_DIR path_to_your_dataset`, or you can simply add

```
DATA:
  PATH_TO_DATA_DIR: path_to_your_dataset
```

To the yaml configs file, then you do not need to pass it to the command line every time.

## Using a Different Number of GPUs

If you want to use a smaller number of GPUs, you need to modify .yaml configuration files in [`configs/`](configs/). Specifically, you need to modify the NUM_GPUS, TRAIN.BATCH_SIZE, TEST.BATCH_SIZE, DATA_LOADER.NUM_WORKERS entries in each configuration file. The BATCH_SIZE entry should be the same or higher as the NUM_GPUS entry. In [`configs/Kinetics/TimeSformer_divST_8x32_224_4gpus.yaml`](configs/Kinetics/TimeSformer_divST_8x32_224_4gpus.yaml), we provide a sample configuration file for a 4 GPU setup.


## Using Different Self-Attention Schemes

If you want to experiment with different space-time self-attention schemes, e.g., space-only or joint space-time attention, use the following commands:


```
python tools/run_net.py \
  --cfg configs/Kinetics/TimeSformer_spaceOnly_8x32_224.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  NUM_GPUS 8 \
  TRAIN.BATCH_SIZE 8 \
```

and

```
python tools/run_net.py \
  --cfg configs/Kinetics/TimeSformer_jointST_8x32_224.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  NUM_GPUS 8 \
  TRAIN.BATCH_SIZE 8 \
```

## Training Different TimeSformer Variants

If you want to train more powerful TimeSformer variants, e.g., TimeSformer-HR (operating on 16-frame clips sampled at 448x448 spatial resolution), and TimeSformer-L (operating on 96-frame clips sampled at 224x224 spatial resolution), use the following commands:

```
python tools/run_net.py \
  --cfg configs/Kinetics/TimeSformer_divST_16x16_448.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  NUM_GPUS 8 \
  TRAIN.BATCH_SIZE 8 \
```

and

```
python tools/run_net.py \
  --cfg configs/Kinetics/TimeSformer_divST_96x4_224.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  NUM_GPUS 8 \
  TRAIN.BATCH_SIZE 8 \
```

Note that for these models you will need a set of GPUs with ~32GB of memory.

## Inference

Use `TRAIN.ENABLE` and `TEST.ENABLE` to control whether training or testing is required for a given run. When testing, you also have to provide the path to the checkpoint model via TEST.CHECKPOINT_FILE_PATH.
```
python tools/run_net.py \
  --cfg configs/Kinetics/TimeSformer_divST_8x32_224_TEST.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  TEST.CHECKPOINT_FILE_PATH path_to_your_checkpoint \
  TRAIN.ENABLE False \
```

## Single-Node Training via Slurm

To train TimeSformer via Slurm, please check out our single node Slurm training script [`slurm_scripts/run_single_node_job.sh`](slurm_scripts/run_single_node_job.sh).


## Multi-Node Training via Submitit

Distributed training is available via Slurm and submitit

```
pip install submitit
```

To train TimeSformer model on Kinetics using 4 nodes with 8 gpus each use the following command:
```
python tools/submit.py --cfg configs/Kinetics/TimeSformer_divST_8x32_224.yaml --job_dir  /your/job/dir/${JOB_NAME}/ --num_shards 4 --name ${JOB_NAME} --use_volta32
```

We provide a script for launching slurm jobs in [`slurm_scripts/run_multi_node_job.sh`](slurm_scripts/run_multi_node_job.sh).

## Finetuning

To finetune from an existing PyTorch checkpoint add the following line in the command line, or you can also add it in the YAML config:

```
TRAIN.CHECKPOINT_FILE_PATH path_to_your_PyTorch_checkpoint
TRAIN.FINETUNE True
```

## HowTo100M Dataset Split

If you want to experiment with the long-term video modeling task on HowTo100M, please download the train/test split files from [here](https://www.dropbox.com/sh/ttvsxwqypijjuda/AACmJx1CnddW6cVBoc21eSuva?dl=0).


# Environment

The code was developed using python 3.7 on Ubuntu 20.04. For training, we used four GPU compute nodes each node containing 8 Tesla V100 GPUs (32 GPUs in total). Other platforms or GPU cards have not been fully tested.

# License

The majority of this work is licensed under [CC-NC 4.0 International license](LICENSE). However portions of the project are available under separate license terms: [SlowFast](https://github.com/facebookresearch/SlowFast) and [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) are licensed under the Apache 2.0 license.

# Contributing

We actively welcome your pull requests. Please see [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for more info.

# Acknowledgements

The fork uses the original [TimeSformer](https://github.com/facebookresearch/TimeSformer) ([Creative Commons Attribution-NonCommercial 4.0 International Public
License](https://github.com/facebookresearch/TimeSformer/blob/main/LICENSE)),
[PySlowFast](https://github.com/facebookresearch/SlowFast) ([Apache License 2.0](https://github.com/facebookresearch/SlowFast/blob/main/LICENSE)), and
[pytorch-image-models](https://github.com/rwightman/pytorch-image-models) ([Apache License 2.0](https://github.com/rwightman/pytorch-image-models/blob/main/LICENSE)) by [Ross Wightman](https://github.com/rwightman).
We thank the authors for releasing their code. If you use our model, please consider citing these works as well:

```BibTeX
@inproceedings{gberta_2021_ICML,
    author  = {Gedas Bertasius and Heng Wang and Lorenzo Torresani},
    title = {Is Space-Time Attention All You Need for Video Understanding?},
    booktitle   = {Proceedings of the International Conference on Machine Learning (ICML)}, 
    month = {July},
    year = {2021}
}
```

```BibTeX
@misc{fan2020pyslowfast,
  author =       {Haoqi Fan and Yanghao Li and Bo Xiong and Wan-Yen Lo and
                  Christoph Feichtenhofer},
  title =        {PySlowFast},
  howpublished = {\url{https://github.com/facebookresearch/slowfast}},
  year =         {2020}
}
```

```BibTeX
@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/rwightman/pytorch-image-models}}
}
```

# Acknowledgement of Support and Disclaimer

This material is based upon work supported by the Defense Advanced Research Projects Agency (DARPA) under Contract No. HR001120C0055. Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the DARPA.
