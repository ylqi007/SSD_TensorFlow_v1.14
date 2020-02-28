# Copyright 2016 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Generic training script that trains a SSD model using a given dataset.
"""
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

import tf_utils

from nets import nets_factory
from datasets import dataset_factory
from preprocessing import preprocessing_factory

import tf_extended as tfe

tf.config.optimizer.set_jit(True)

DATA_FORMAT = 'NHWC'
# DATA_FORMAT = 'NCHW'

parser = argparse.ArgumentParser()

# =========================================================================== #
# SSD Network flags.
# =========================================================================== #
parser.add_argument('--loss_alpha', default=1,
                    help='Alpha parameter in the loss function.')
parser.add_argument('--negative_ratio', default=3.,
                    help='Negative ratio in the loss function.')
parser.add_argument('--match_threshold', default=0.5,
                    help='Matching threshold in the loss function.')


# =========================================================================== #
# General Flags.
# =========================================================================== #


# =========================================================================== #
# Optimization Flags.
# =========================================================================== #
parser.add_argument('--weight_decay', default=0.00004,
                    help='The weight decay on the model weight.')
parser.add_argument('--optimizer', default='rmsprop',
                    help='The name of optimizer, one of "adadelta", "adagrad", "adam"'
                         '"ftrl", "momentum", "sgd", or "rmsprop"')
parser.add_argument('--opt_epsilon', default=1.0,
                    help='Epsilon term fro the optimizer.')
parser.add_argument('--rmsprop_decay', default=0.9,
                    help='Decay term for RMSProp.')
parser.add_argument('--rmsprop_momentum', default=0.9,
                    help='Momentum.')


# =========================================================================== #
# Learning Rate Flags.
# =========================================================================== #
parser.add_argument('--learning_rate_decay_type', default='exponential',
                    help='Specifics how the learning rate is decayed.'
                         'One of "fixed", "exponential", or "polynomina".')
parser.add_argument('--learning_rate', default=0.001,
                    help='initial learning rate.')
parser.add_argument('--learning_rate_decay_factor', default=0.94,
                    help='Learning rate decay factor.')
parser.add_argument('--moving_average_decay', default=None,
                    help='The decay rate to use for the moving average.'
                         'If left None, then moving averages are not used.')
parser.add_argument('--label_smoothing', default=0.0,
                    help='The amount of label smoothing.')
parser.add_argument('--num_epochs_per_decay', default=2.0,
                    help='Number of epochs after which ')

# =========================================================================== #
# Dataset Flags.
# =========================================================================== #
parser.add_argument('--dataset_name', default='pascalvoc',
                    help='The name of dataset to load.')
parser.add_argument('--num_classes', default=21,
                    help='Number of classes to use in the dataset.')
parser.add_argument('--dataset_split_name', default='train',
                    help='The name of train/test split.')
parser.add_argument('--dataset_dir', default=None,
                    help='The directory where the tfrecord files are stored.')
parser.add_argument('--model_name', default='ssd_300_vgg',
                    help='The name of the architecture to train.')
parser.add_argument('--preprocessing_name', default=None,
                    help='The name of the preprocessing function to use.'
                         'If left as None, then the model_name is used.')
parser.add_argument('--batch_size', default=None, type=int,
                    help='The number of samples in each batch.')

# =========================================================================== #
# Fine-Tuning Flags.
# =========================================================================== #


args = parser.parse_args()


# =========================================================================== #
# Main training routine.
# =========================================================================== #
def main():
    if not args.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    # tf.logging.set_verbosity(tf.logging.DEBUG)
    with tf.Graph().as_default():
        # Create global_step.
        with tf.device('/cpu:0'):   # <tf.Variable 'global_step:0' shape=() dtype=int64_ref>
            global_step = tf.train.get_or_create_global_step()

        # Get the SSD network and its anchors.
        ssd_class = nets_factory.get_network(args.model_name)   # ssd_class:  <class 'nets.ssd_vgg_300.SSDNet'>
        ssd_params = ssd_class.default_params._replace(num_classes=args.num_classes)
        ssd_net = ssd_class(ssd_params)
        ssd_shape = ssd_net.params.img_shape
        ssd_anchors = ssd_net.anchors(ssd_shape)    # ssd_anchors is a list with len equals 6.

        # Select the preprocessing function.
        preprocessing_name = args.preprocessing_name or args.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(preprocessing_name, is_training=True)
        # Select the dataset.
        dataset = dataset_factory.get_dataset(args.dataset_name,
                                              args.dataset_split_name,
                                              args.dataset_dir)     # image, shape, label, bboxes
        dataset = dataset.map(lambda image, shape, label, bboxes:
                              image_preprocessing_fn(image, shape, label, bboxes,
                                                     out_shape=ssd_shape,
                                                     data_format=DATA_FORMAT))  # image, labels, bboxes
        dataset = dataset.map(lambda image, labels, bboxes:
                              ssd_net.bboxes_encode(image, labels, bboxes,
                                                    anchors=ssd_anchors))
        dataset = dataset.repeat()
        dataset = dataset.batch(4)
        iterator = tf.data.make_one_shot_iterator(dataset)
        r = iterator.get_next()
        batch_shape = [1] + [len(ssd_anchors)] * 3
        b_image, b_gclasses, b_glocalisations, b_gscores = tf_utils.reshape_list(r, batch_shape)    # line 251
        # test_after_reshape_list(b_image, b_gclasses, b_glocalisations, b_gscores)

        # Construct network and add losses.
        predictions, localisations, logits, end_points = ssd_net.net(b_image, is_training=True)
        n_positive = ssd_net.losses(logits, localisations,
                                    b_gclasses, b_glocalisations, b_gscores,
                                    match_threshold=args.match_threshold,
                                    negative_ratio=args.negative_ratio,
                                    alpha=args.loss_alpha,
                                    label_smoothing=args.label_smoothing)

        # =================================================================== #
        # Configure the moving averages.
        # =================================================================== #
        # if args.moving_average_decay:
        #     # moving_average_variables = slim.get_model_variables()
        #     # variable_averages = tf.train.ExponentialMovingAverage(args.moving_average_decay, global_step)
        #     moving_average_variables, variable_averages = None, None
        # else:
        #     moving_average_variables, variable_averages = None, None
        #
        # =================================================================== #
        # Configure the optimization procedure.
        # =================================================================== #
        with tf.device('/cpu:0'):
            learning_rate = tf_utils.configure_learning_rate(args,
                                                             10,  # dataset.num_samples,
                                                             global_step)
            optimizer = tf_utils.configure_optimizer(args, learning_rate)
            # summaries.add(tf.summary.scalar('learning_rate', learning_rate))

        # # Variables to train.
        # # variables_to_train = tf_utils.get_variables_to_train(args)
        # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        losses = tf.get_collection(tf.GraphKeys.LOSSES)
        print('losses: ', losses)
        print('===================================')
        losses1 = tf.math.add_n(losses)
        print(losses)
        print(losses1)
        # train_step = optimizer.minimize(losses1, global_step=global_step)
        train_step = optimizer.minimize(losses1)
        # =================================================================== #
        # Gather summaries.
        # =================================================================== #
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        print('summaries: ', summaries, tf.GraphKeys.SUMMARIES, tf.get_collection(tf.GraphKeys.SUMMARIES))
        print('losses: ', tf.GraphKeys.LOSSES, tf.get_collection(tf.GraphKeys.LOSSES))
        # Add summaries for losses (and extra losses).
        for loss in tf.get_collection(tf.GraphKeys.LOSSES):
            summaries.add(tf.summary.scalar(loss.op.name, loss))
        print('summaries: ', summaries, tf.GraphKeys.SUMMARIES, tf.get_collection(tf.GraphKeys.SUMMARIES))
        # Merge all summaries together.
        summary_op = tf.summary.merge(list(summaries), name='summary_op')
        print(summary_op)
        train_writer = tf.summary.FileWriter('./logs/train', tf.get_default_graph())
        test_writer = tf.summary.FileWriter('./logs/test/')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(100):
                _losses1, _train_step, _global_step, _summary = sess.run([losses1, train_step, global_step, summary_op])
                print('Iteration %3d Losses: %f' % (i, _losses1))
                # print(_train_step)
                if i % 10 == 9:
                    train_writer.add_summary(_summary, i)
        train_writer.close()
                # _n_positive = sess.run(n_positive)
                # print('### _n_positive: ', _n_positive)
        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     try:
        #         while True:
        #             print('\n=================== In Session =============================\n')
        #             _net = sess.run(b_image)
        #             tmp = (_net[0]).round().astype(np.uint8)
        #             print(_net[-1][-1])
        #             print(type(tmp), tmp.shape, tmp.min(), tmp.max())
        #             img = Image.fromarray(tmp)
        #             img.show()
        #             # _logits = sess.run(fnmask)
        #             # count = sum(i==1. for i in _logits)
        #             # print('#$#$ count: ', count)
        #             # print(_logits.shape)
        #             # print('glocalisations: ', _logits)
        #             # print('min & max: ', np.min(_logits), np.max(_logits))
        #             # print('@@@@@@@@@@@@@@@@@@')
        #             # for i in _logits:
        #             #     if i > 0.25:
        #             #         print('$$$ i: ', i)
        #     except tf.errors.OutOfRangeError:
        #         pass


def test_after_reshape_list(b_image, b_gclasses, b_glocalisations, b_gscores):
    print('===================================================')
    print('===================== After reshaping ======================')
    print('b_image:\n', b_image)
    print('b_gclasses:\n', b_gclasses)
    print('b_glocalisations:\n', b_glocalisations)
    print('b_gscores:\n', b_gscores)
    print('===================================================')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            while True:
                print('\n=================== In Session =============================\n')
                ### b_image information
                # _b_image = sess.run(b_image)
                # tmp = (_b_image[0]*255).round().astype(np.uint8)
                # print(tmp[-1][-1])
                # print(type(tmp), tmp.shape, tmp.min(), tmp.max())
                # img = Image.fromarray(tmp)
                # img.show()
                ## b_gclasses information
                # _b_gclasses = sess.run(b_gclasses)  # <class 'tuple'>
                # tmp = _b_gclasses[4]
                # print(type(tmp), tmp.shape, tmp.min(), tmp.max())
                # for cls in np.nditer(tmp):
                #     if cls != 0:
                #         print(cls)
                ## b_glocalisations
                # _b_glocalisations = sess.run(b_glocalisations)
                # tmp = _b_glocalisations[-1]
                # print(tmp[-1])
                # print(type(tmp), tmp.shape, tmp.min(), tmp.max())
                ## b_gscores
                # _b_gscores = sess.run(b_gscores)
                # tmp = _b_gscores[-1]
                # print(tmp[-1])
                # print(type(tmp), tmp.shape, tmp.min(), tmp.max())
        except tf.errors.OutOfRangeError:
            pass


if __name__ == '__main__':
    main()
