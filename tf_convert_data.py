# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""
Convert a dataset to TFRecords format, which can be easily integrated into
a TensorFlow pipeline.

Usage:
    ```shell
    python tf_convert_data.py \
        --dataset_name=pascalvoc \
        --dataset_dir=/tmp/pascalvoc \
        --output_name=pascalvoc \
        --output_dir=/tmp/
    ```
"""
import argparse

from datasets import pascalvoc_to_tfrecords


parser = argparse.ArgumentParser()

parser.add_argument('--dataset_name',
                    default='pascalvoc',
                    help='The name of the dataset to load.')
parser.add_argument('--dataset_dir',
                    default=None,
                    help='Directory where the original dataset is stored.')
parser.add_argument('--output_name',
                    default='pascalvoc',
                    help='Basename used for TFRecord output files.')
parser.add_argument('--output_dir',
                    default='./',
                    help='Output directory where to store TFRecord files.')

args = parser.parse_args()


def main():
    if not args.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
    print('Dataset directory:', args.dataset_dir)
    print('Output directory:', args.output_dir)

    if args.dataset_name == 'pascalvoc':
        pascalvoc_to_tfrecords.run(args.dataset_dir, args.output_dir, args.output_name)
    else:
        raise ValueError('Dataset [%s] was not recognized.' % args.dataset_name)


if __name__ == '__main__':
    main()
