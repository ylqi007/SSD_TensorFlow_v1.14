# Copyright 2015 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the Pascal VOC Dataset (images + annotations).
"""
import os
import glob

import tensorflow as tf
from datasets import dataset_utils


VOC_LABELS = {
    'none': (0, 'Background'),
    'aeroplane': (1, 'Vehicle'),
    'bicycle': (2, 'Vehicle'),
    'bird': (3, 'Animal'),
    'boat': (4, 'Vehicle'),
    'bottle': (5, 'Indoor'),
    'bus': (6, 'Vehicle'),
    'car': (7, 'Vehicle'),
    'cat': (8, 'Animal'),
    'chair': (9, 'Indoor'),
    'cow': (10, 'Animal'),
    'diningtable': (11, 'Indoor'),
    'dog': (12, 'Animal'),
    'horse': (13, 'Animal'),
    'motorbike': (14, 'Vehicle'),
    'person': (15, 'Person'),
    'pottedplant': (16, 'Indoor'),
    'sheep': (17, 'Animal'),
    'sofa': (18, 'Indoor'),
    'train': (19, 'Vehicle'),
    'tvmonitor': (20, 'Indoor'),
}

# Create a dictionary for parsing features from tf.Example.
image_features_description = {
    'image/raw_data': tf.io.FixedLenFeature((), tf.string, default_value=''),
    'image/format': tf.io.FixedLenFeature((), tf.string, default_value='jpeg'),
    'image/height': tf.io.FixedLenFeature([1], tf.int64),
    'image/width': tf.io.FixedLenFeature([1], tf.int64),
    'image/channels': tf.io.FixedLenFeature([1], tf.int64),
    'image/shape': tf.io.FixedLenFeature([3], tf.int64),
    'image/object/bbox/xmin': tf.io.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/label': tf.io.VarLenFeature(dtype=tf.int64),
    'image/object/bbox/difficult': tf.io.VarLenFeature(dtype=tf.int64),
    'image/object/bbox/truncated': tf.io.VarLenFeature(dtype=tf.int64),
}


def get_dataset(split_name, dataset_dir, file_pattern):
    """
    Gets a dataset tuple with instructions for reading Pascal VOC dataset.
    Args:
        split_name: A train/test split name.
        dataset_dir: The base directory of the dataset sources.
        file_pattern: The file pattern to use when matching the dataset sources.
            It is assumed that the pattern contains a '%s' string so that the split
            name can be inserted.
    Returns:
        A `Dataset` namedtuple.
    Raises:
        ValueError: if `split_name` is not a valid train/test split.
    """
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)
    files = glob.glob(file_pattern)
    raw_image_dataset = tf.data.TFRecordDataset(files)  # <TFRecordDatasetV1 shapes: (), types: tf.string>
    print('raw_image_dataset: ', raw_image_dataset)

    parsed_image_dataset = raw_image_dataset.map(_parse_example_function)
    #
    # parsed_feature_dataset = parsed_image_dataset.map(_parse_feature_function)

    # return parsed_feature_dataset
    return parsed_image_dataset


# def _parse_example_function(example_proto):
#     image_features = tf.io.parse_single_example(example_proto, image_features_description)
#     image_features['image/raw_data'] = tf.io.decode_jpeg(image_features['image/raw_data'], channels=3)
#     image_features['image/object/bbox/difficult'] = tf.sparse.to_dense(image_features['image/object/bbox/difficult'])
#     image_features['image/object/bbox/label'] = tf.sparse.to_dense(image_features['image/object/bbox/label'])
#     image_features['image/object/bbox/truncated'] = tf.sparse.to_dense(image_features['image/object/bbox/truncated'])
#     image_features['image/object/bbox/xmin'] = tf.sparse.to_dense(image_features['image/object/bbox/xmin'])
#     image_features['image/object/bbox/ymin'] = tf.sparse.to_dense(image_features['image/object/bbox/ymin'])
#     image_features['image/object/bbox/xmax'] = tf.sparse.to_dense(image_features['image/object/bbox/xmax'])
#     image_features['image/object/bbox/ymax'] = tf.sparse.to_dense(image_features['image/object/bbox/ymax'])
#
#     return image_features


def _parse_example_function(example_proto):
    image_features = tf.io.parse_single_example(example_proto, image_features_description)
    image = tf.io.decode_jpeg(image_features['image/raw_data'], channels=3)

    difficult = tf.sparse.to_dense(image_features['image/object/bbox/difficult'])
    truncated = tf.sparse.to_dense(image_features['image/object/bbox/truncated'])
    label = tf.sparse.to_dense(image_features['image/object/bbox/label'])
    xmin = tf.sparse.to_dense(image_features['image/object/bbox/xmin'])
    ymin = tf.sparse.to_dense(image_features['image/object/bbox/ymin'])
    xmax = tf.sparse.to_dense(image_features['image/object/bbox/xmax'])
    ymax = tf.sparse.to_dense(image_features['image/object/bbox/ymax'])
    channels = image_features['image/channels']
    image_format = image_features['image/format']
    height = image_features['image/height']
    width = image_features['image/width']
    shape = image_features['image/shape']

    return difficult, truncated, label,\
           xmin, ymin, xmax, ymax,\
           channels, image_format, height, width, image, shape


def _parse_feature_function(_image_features):
    image_features = {
        'image': _image_features['image/raw_data'],
        'shape': _image_features['image/shape'],
        'object/bbox': tf.transpose([tf.sparse_to_dense(_image_features['image/object/bbox/ymin']),
                                     tf.sparse_to_dense(_image_features['image/object/bbox/xmin']),
                                     tf.sparse_to_dense(_image_features['image/object/bbox/ymax']),
                                     tf.sparse_to_dense(_image_features['image/object/bbox/xmax'])]),
        'object/label': tf.sparse_to_dense(_image_features['image/object/bbox/label'])
    }

    return image_features
