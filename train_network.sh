#!/usr/bin/env bash

DATASET_DIR="/tmp/tfrecords"
TRAIN_DIR="./logs/"
CHECKPOINT_PATH="./checkpoints/ssd_300_vgg.ckpt"


python train_network.py     \
    --dataset_name=pascalvoc_2007    \
    --dataset_dir=${DATASET_DIR}    \
    --dataset_split_name=train  \
    --batch_size=1

