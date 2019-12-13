#!/usr/bin/env bash

#VOC2007_TRAIN="../../DATA/VOC2007/train/"
#VOC2007_TEST="../../DATA/VOC2007/test/"

VOC2007_TRAIN="../../DATA/DEMO/train/"
VOC2007_TEST="../../DATA/DEMO/test/"

OUTPUT_DIR="/tmp/tfrecords"

if [[ ! -d ${OUTPUT_DIR} ]]; then
    mkdir ${OUTPUT_DIR}
fi

# Convert Pascal VOC train to tfrecord files.
python tf_convert_data.py   \
    --dataset_name=pascalvoc    \
    --dataset_dir=${VOC2007_TRAIN}  \
    --output_name=voc_2007_train    \
    --output_dir=${OUTPUT_DIR}
