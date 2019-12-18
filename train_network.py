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

# DATA_FORMAT = 'NHWC'    # 'NCHW'
DATA_FORMAT = 'NCHW'

parser = argparse.ArgumentParser()


# =========================================================================== #
# SSD Network flags.
# =========================================================================== #


# =========================================================================== #
# General Flags.
# =========================================================================== #


# =========================================================================== #
# Optimization Flags.
# =========================================================================== #


# =========================================================================== #
# Learning Rate Flags.
# =========================================================================== #


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

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)
    with tf.Graph().as_default():
        # Create global_step.
        with tf.device('/cpu:0'):
            global_step = tf.compat.v1.train.create_global_step()
        print(global_step)

        # Get the SSD network and its anchors.
        ssd_class = nets_factory.get_network(args.model_name)   # ssd_class:  <class 'nets.ssd_vgg_300.SSDNet'>
        ssd_params = ssd_class.default_params._replace(num_classes=args.num_classes)
        ssd_net = ssd_class(ssd_params)
        ssd_shape = ssd_net.params.img_shape
        ssd_anchors = ssd_net.anchors(ssd_shape)    # ssd_anchors is a list with len equals 6.
        print('=============================================')
        print('#### ssd_anchors: ', len(ssd_anchors), type(ssd_anchors))
        # print(ssd_anchors)

        # Select the preprocessing function.
        preprocessing_name = args.preprocessing_name or args.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name, is_training=True)
        # image_preprocessing_fn = image_preprocessing_fn(out_shape=ssd_shape)
        # Select the dataset.
        # resize_image = resize_image_func(output_size=ssd_shape)
        dataset = dataset_factory.get_dataset(args.dataset_name,
                                              args.dataset_split_name,
                                              args.dataset_dir)     # image, shape, label, bboxes
        dataset = dataset.map(lambda image, shape, label, bboxes:
                              image_preprocessing_fn(image, shape, label, bboxes,
                                                     out_shape=ssd_shape,
                                                     data_format=DATA_FORMAT))
        # dataset = dataset.map(resize_image)

        # bboxes_encode = ssd_net.bboxes_encode(anchors=ssd_anchors, scope=None)
        # dataset = dataset.map(lambda image, shape, label, bboxes:
        #                       ssd_net.bboxes_encode(label, bboxes, ssd_anchors))
        # dataset = dataset.batch(2)
        print('################################################')
        print('Info of dataset: ', dataset)
        # print('\nBefore batching: ', dataset)
        # dataset = dataset.batch(2)
        # batched_dataset = dataset.batch(3, drop_remainder=True)
        # print('\nAfter batching : ', batched_dataset)
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        print('\niterator: ', iterator)

        # image, shape, label, bboxes = iterator.get_next()
        image, labels, bboxes = iterator.get_next()
        print('\n## image: ', image)
        # print('## shape: ', shape)
        print('## label: ', labels)
        print('## bboxes: ', bboxes)
        print()

        image_with_box = draw_bounding_boxes(image, bboxes)
        print('@@ image: ', image)
        print('@@ bboxes: ', bboxes)
        print('@@ image_with_box: ', image_with_box)

        # Encode groundtruth labels and bboxes.
        # gclasses, glocalisations, gscores = ssd_net.bboxes_encode(label, bboxes, ssd_anchors)
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        # print('Info gclasses: ', len(gclasses))
        # print('\nInfo glocalisations: ', len(glocalisations))
        # print('\nInfo gscores: ', len(gscores))
        # reshaped_info = tf_utils.reshape_list([image, gclasses, glocalisations, gscores])
        print('====================================================')
        with tf.compat.v1.Session() as sess:
            try:
                while True:
                    print('\n=================== In Session =============================\n')
                    # _image_with_box = sess.run(image_with_box)
                    # # print(_image_with_box[0])
                    # print(_image_with_box.shape, _image_with_box.shape, _image_with_box.min(), _image_with_box.max())
                    # tmp = (_image_with_box[0] * 255).round().astype(np.uint8)
                    _image = sess.run(image)
                    print(type(_image), _image.shape, _image.min(), _image.max())
                    tmp = (_image * 255).astype(np.uint8)
                    print(type(tmp), tmp.shape, tmp.min(), tmp.max())
                    img = Image.fromarray(tmp)
                    img.show()
            except tf.errors.OutOfRangeError:
                pass

            # while(True):
            #     print('\n==================\n')
            #     print('#### iterator: ', iterator)
            #     print('#### sample: ', sample)
            #     print('\n')
            #     _features = [difficult, truncated, label,
            #                  xmin, ymin, xmax, ymax,
            #                  channels, _format, height, width, image, shape]
            #     print('_features: ', _features)
            #     features = sess.run(_features)
            #     print('\nfeatures: ', features)
            #     img = Image.fromarray(features[-2])
            #     img.show()
            #     print(image)

            # print('\ndifficult, truncated, label: ')
            # _difficult, _truncated, _label = sess.run([difficult, truncated, label])
            # print(_difficult, _truncated, _label)

            # print('new_info1: ', new_info1, sess.run(new_info1))
            # print('new_info1: ', new_info1, sess.run(new_info1))
            # print('new_info1: ', new_info1, sess.run(new_info1))

        # for raw_record in dataset.take(3):
        #     print(repr(raw_record))
        # # Get the SSD network and its anchors.
        # ssd_class = nets_factory.get_network(FLAGS.model_name)
        # ssd_params = ssd_class.default_params._replace(num_classes=FLAGS.num_classes)
        # ssd_net = ssd_class(ssd_params)
        # ssd_shape = ssd_net.params.img_shape
        # ssd_anchors = ssd_net.anchors(ssd_shape)
        #
        # # Select the preprocessing function.
        # preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        # image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        #     preprocessing_name, is_training=True)
        #
        # tf_utils.print_configuration(FLAGS.__flags, ssd_params,
        #                              dataset.data_sources, FLAGS.train_dir)
        # # =================================================================== #
        # # Create a dataset provider and batches.
        # # =================================================================== #
        # with tf.device(deploy_config.inputs_device()):
        #     with tf.name_scope(FLAGS.dataset_name + '_data_provider'):
        #         provider = slim.dataset_data_provider.DatasetDataProvider(
        #             dataset,
        #             num_readers=FLAGS.num_readers,
        #             common_queue_capacity=20 * FLAGS.batch_size,
        #             common_queue_min=10 * FLAGS.batch_size,
        #             shuffle=True)
        #     # Get for SSD network: image, labels, bboxes.
        #     [image, shape, glabels, gbboxes] = provider.get(['image', 'shape',
        #                                                      'object/label',
        #                                                      'object/bbox'])
        #     # Pre-processing image, labels and bboxes.
        #     image, glabels, gbboxes = \
        #         image_preprocessing_fn(image, glabels, gbboxes,
        #                                out_shape=ssd_shape,
        #                                data_format=DATA_FORMAT)
        #     # Encode groundtruth labels and bboxes.
        #     gclasses, glocalisations, gscores = \
        #         ssd_net.bboxes_encode(glabels, gbboxes, ssd_anchors)
        #     batch_shape = [1] + [len(ssd_anchors)] * 3
        #
        #     # Training batches and queue.
        #     r = tf.train.batch(
        #         tf_utils.reshape_list([image, gclasses, glocalisations, gscores]),
        #         batch_size=FLAGS.batch_size,
        #         num_threads=FLAGS.num_preprocessing_threads,
        #         capacity=5 * FLAGS.batch_size)
        #     b_image, b_gclasses, b_glocalisations, b_gscores = \
        #         tf_utils.reshape_list(r, batch_shape)
        #
        #     # Intermediate queueing: unique batch computation pipeline for all
        #     # GPUs running the training.
        #     batch_queue = slim.prefetch_queue.prefetch_queue(
        #         tf_utils.reshape_list([b_image, b_gclasses, b_glocalisations, b_gscores]),
        #         capacity=2 * deploy_config.num_clones)
        #
        # # =================================================================== #
        # # Define the model running on every GPU.
        # # =================================================================== #
        # def clone_fn(batch_queue):
        #     """Allows data parallelism by creating multiple
        #     clones of network_fn."""
        #     # Dequeue batch.
        #     b_image, b_gclasses, b_glocalisations, b_gscores = \
        #         tf_utils.reshape_list(batch_queue.dequeue(), batch_shape)
        #
        #     # Construct SSD network.
        #     arg_scope = ssd_net.arg_scope(weight_decay=FLAGS.weight_decay,
        #                                   data_format=DATA_FORMAT)
        #     with slim.arg_scope(arg_scope):
        #         predictions, localisations, logits, end_points = \
        #             ssd_net.net(b_image, is_training=True)
        #     # Add loss function.
        #     ssd_net.losses(logits, localisations,
        #                    b_gclasses, b_glocalisations, b_gscores,
        #                    match_threshold=FLAGS.match_threshold,
        #                    negative_ratio=FLAGS.negative_ratio,
        #                    alpha=FLAGS.loss_alpha,
        #                    label_smoothing=FLAGS.label_smoothing)
        #     return end_points
        #
        # # Gather initial summaries.
        # summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        #
        # # =================================================================== #
        # # Add summaries from first clone.
        # # =================================================================== #
        # clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
        # first_clone_scope = deploy_config.clone_scope(0)
        # # Gather update_ops from the first clone. These contain, for example,
        # # the updates for the batch_norm variables created by network_fn.
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)
        #
        # # Add summaries for end_points.
        # end_points = clones[0].outputs
        # for end_point in end_points:
        #     x = end_points[end_point]
        #     summaries.add(tf.summary.histogram('activations/' + end_point, x))
        #     summaries.add(tf.summary.scalar('sparsity/' + end_point,
        #                                     tf.nn.zero_fraction(x)))
        # # Add summaries for losses and extra losses.
        # for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
        #     summaries.add(tf.summary.scalar(loss.op.name, loss))
        # for loss in tf.get_collection('EXTRA_LOSSES', first_clone_scope):
        #     summaries.add(tf.summary.scalar(loss.op.name, loss))
        #
        # # Add summaries for variables.
        # for variable in slim.get_model_variables():
        #     summaries.add(tf.summary.histogram(variable.op.name, variable))
        #
        # # =================================================================== #
        # # Configure the moving averages.
        # # =================================================================== #
        # if FLAGS.moving_average_decay:
        #     moving_average_variables = slim.get_model_variables()
        #     variable_averages = tf.train.ExponentialMovingAverage(
        #         FLAGS.moving_average_decay, global_step)
        # else:
        #     moving_average_variables, variable_averages = None, None
        #
        # # =================================================================== #
        # # Configure the optimization procedure.
        # # =================================================================== #
        # with tf.device(deploy_config.optimizer_device()):
        #     learning_rate = tf_utils.configure_learning_rate(FLAGS,
        #                                                      dataset.num_samples,
        #                                                      global_step)
        #     optimizer = tf_utils.configure_optimizer(FLAGS, learning_rate)
        #     summaries.add(tf.summary.scalar('learning_rate', learning_rate))
        #
        # if FLAGS.moving_average_decay:
        #     # Update ops executed locally by trainer.
        #     update_ops.append(variable_averages.apply(moving_average_variables))
        #
        # # Variables to train.
        # variables_to_train = tf_utils.get_variables_to_train(FLAGS)
        #
        # # and returns a train_tensor and summary_op
        # total_loss, clones_gradients = model_deploy.optimize_clones(
        #     clones,
        #     optimizer,
        #     var_list=variables_to_train)
        # # Add total_loss to summary.
        # summaries.add(tf.summary.scalar('total_loss', total_loss))
        #
        # # Create gradient updates.
        # grad_updates = optimizer.apply_gradients(clones_gradients,
        #                                          global_step=global_step)
        # update_ops.append(grad_updates)
        # update_op = tf.group(*update_ops)
        # train_tensor = control_flow_ops.with_dependencies([update_op], total_loss,
        #                                                   name='train_op')
        #
        # # Add the summaries from the first clone. These contain the summaries
        # summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
        #                                    first_clone_scope))
        # # Merge all summaries together.
        # summary_op = tf.summary.merge(list(summaries), name='summary_op')
        #
        # # =================================================================== #
        # # Kicks off the training.
        # # =================================================================== #
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
        # config = tf.ConfigProto(log_device_placement=False,
        #                         gpu_options=gpu_options)
        # saver = tf.train.Saver(max_to_keep=5,
        #                        keep_checkpoint_every_n_hours=1.0,
        #                        write_version=2,
        #                        pad_step_number=False)
        # slim.learning.train(
        #     train_tensor,
        #     logdir=FLAGS.train_dir,
        #     master='',
        #     is_chief=True,
        #     init_fn=tf_utils.get_init_fn(FLAGS),
        #     summary_op=summary_op,
        #     number_of_steps=FLAGS.max_number_of_steps,
        #     log_every_n_steps=FLAGS.log_every_n_steps,
        #     save_summaries_secs=FLAGS.save_summaries_secs,
        #     saver=saver,
        #     save_interval_secs=FLAGS.save_interval_secs,
        #     session_config=config,
        #     sync_optimizer=None)


def draw_bounding_boxes(image, bboxes):
    # Convert tf.uint8 to tf.float32.
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # print('#####################################')
    # print('$$$$$ draw_bounding_boxes $$$$$')
    # print('before expanding dims')
    # print('#### image ', image.get_shape(), image.get_shape().ndims)
    # print('#### bboxes: ', bboxes, bboxes.get_shape(), bboxes.get_shape().ndims)
    if image.get_shape().ndims == 3:
        image = tf.expand_dims(image, axis=0)
    if bboxes.get_shape().ndims == 2:
        bboxes = tf.expand_dims(bboxes, axis=0)
    # print('After expanding dims')
    # print('#### image: ', image, image.get_shape(), image.get_shape().ndims)
    # print('#### bboxes: ', bboxes, bboxes.get_shape(), bboxes.get_shape().ndims)
    print('#########################################################')
    print('image: ', image)
    print('bboxes: ', bboxes)
    image_with_box = tf.image.draw_bounding_boxes(image, bboxes)
    return image_with_box


# Resize without expanding dims
def resize_image_func(output_size):
    def _resize_image(image, shape, label, bboxes, size=output_size):
        """
        image: image can be a single image or a batch of images.
        """
        with tf.name_scope('resize_image'):
            image = tf.image.resize(image, size=size,
                                    method=tf.image.ResizeMethod.BILINEAR,
                                    align_corners=False)
            shape = tf.stack([size[0], size[1], 3])
            # image = tf.reshape(image, tf.stack([size[0], size[1], 3]))
            image = tf.reshape(image, shape)
            return image, shape, label, bboxes
    return _resize_image


# Encode
def _bboxes_encode(func=None, anchors=None, scope=None):
    if func is None:
        raise ValueError('You must provide ssd_encode_func')
    if anchors is None:
        raise ValueError('You must prove anchors.')
    return func(anchors=anchors, scope=scope)


if __name__ == '__main__':
    main()
