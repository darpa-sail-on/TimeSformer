# Note that test_root is not just an output director to be generated, it
# expects a relative directory structure

HYDRA_FULL_ERROR=1 sail-on-client \
    --config-dir "$(pwd)/TimeSformer/configs/" \
    --config-name system_detection_classification_feedback_local \
    test_root="$(pwd)" \
    model_root="$(pwd)/data/model_root/" \
    protocol.smqtk.config.feature_dir="$(pwd)/data/" \
    protocol.smqtk.config.dataset_root="$(pwd)/data/OND/activity_recognition/" \
    protocol.smqtk.config.test_ids=[OND.0.10001.6438158] \
    algorithms@protocol.smqtk.config.algorithms=[finetune_windowed_mean_kldiv]
