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
"""Diverse TensorFlow utils, for training, evaluation and so on!
"""
import os
from pprint import pprint

import tensorflow as tf
from tensorflow.contrib.slim.python.slim.data import parallel_reader

slim = tf.contrib.slim


# =========================================================================== #
# General tools.
# =========================================================================== #
def reshape_list(l, shape=None):
    """Reshape list of (list): 1D to 2D or the other way around.
    Args:
      l: List or List of list.
      shape: 1D or 2D shape.
    Return
      Reshaped list.
    """
    r = []
    if shape is None:
        # Flatten everything.
        for a in l:
            if isinstance(a, (list, tuple)):
                r = r + list(a)
            else:
                r.append(a)
    else:
        # Reshape to list of list.
        i = 0
        for s in shape:
            if s == 1:
                r.append(l[i])
            else:
                r.append(l[i:i+s])
            i += s
    return r

#
# # =========================================================================== #
# # Training utils.
# # =========================================================================== #
# def print_configuration(flags, ssd_params, data_sources, save_dir=None):
#     """Print the training configuration.
#     """
#     def print_config(stream=None):
#         print('\n# =========================================================================== #', file=stream)
#         print('# Training | Evaluation flags:', file=stream)
#         print('# =========================================================================== #', file=stream)
#         pprint(flags, stream=stream)
#
#         print('\n# =========================================================================== #', file=stream)
#         print('# SSD net parameters:', file=stream)
#         print('# =========================================================================== #', file=stream)
#         pprint(dict(ssd_params._asdict()), stream=stream)
#
#         print('\n# =========================================================================== #', file=stream)
#         print('# Training | Evaluation dataset files:', file=stream)
#         print('# =========================================================================== #', file=stream)
#         data_files = parallel_reader.get_data_files(data_sources)
#         pprint(sorted(data_files), stream=stream)
#         print('', file=stream)
#
#     print_config(None)
#     # Save to a text file as well.
#     if save_dir is not None:
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#         path = os.path.join(save_dir, 'training_config.txt')
#         with open(path, "w") as out:
#             print_config(out)


def configure_learning_rate(args, num_samples_per_epoch, global_step):
    """Configures the learning rate.
    Args:
      num_samples_per_epoch: The number of samples in each epoch of training.
      global_step: The global_step tensor.
    Returns:
      A `Tensor` representing the learning rate.
    """
    # print('*********************************************************************')
    # print('num_samples_per_epoch: ', num_samples_per_epoch)
    # print('global_step: ', global_step)
    # print('args.batch_size: ', args.batch_size, type(args.batch_size))
    # print('args.num_epochs_per_decay: ', args.num_epochs_per_decay)
    decay_steps = int(num_samples_per_epoch / args.batch_size *
                      args.num_epochs_per_decay)

    if args.learning_rate_decay_type == 'exponential':
        print('===== config learning rate =====')
        print('args.learning_rate: ', args.learning_rate)
        print('global_step: ', global_step)
        print('decay steps: ', decay_steps)
        print('learning_rate_decay_factor: ', args.learning_rate_decay_type)
        return tf.train.exponential_decay(args.learning_rate,
                                          global_step,
                                          decay_steps,
                                          args.learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    # elif args.learning_rate_decay_type == 'fixed':
    #     return tf.constant(args.learning_rate, name='fixed_learning_rate')
    # elif args.learning_rate_decay_type == 'polynomial':
    #     return tf.train.polynomial_decay(args.learning_rate,
    #                                      global_step,
    #                                      decay_steps,
    #                                      args.end_learning_rate,
    #                                      power=1.0,
    #                                      cycle=False,
    #                                      name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized',
                         args.learning_rate_decay_type)


def configure_optimizer(args, learning_rate):
    """Configures the optimizer used for training.
    Args:
      learning_rate: A scalar or `Tensor` learning rate.
    Returns:
      An instance of an optimizer.
    """
    if args.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=args.adadelta_rho,
            epsilon=args.opt_epsilon)
    elif args.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=args.adagrad_initial_accumulator_value)
    elif args.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=args.adam_beta1,
            beta2=args.adam_beta2,
            epsilon=args.opt_epsilon)
    elif args.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=args.ftrl_learning_rate_power,
            initial_accumulator_value=args.ftrl_initial_accumulator_value,
            l1_regularization_strength=args.ftrl_l1,
            l2_regularization_strength=args.ftrl_l2)
    elif args.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=args.momentum,
            name='Momentum')
    elif args.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=args.rmsprop_decay,
            momentum=args.rmsprop_momentum,
            epsilon=args.opt_epsilon)
    elif args.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', args.optimizer)
    return optimizer


# def add_variables_summaries(learning_rate):
#     summaries = []
#     for variable in slim.get_model_variables():
#         summaries.append(tf.summary.histogram(variable.op.name, variable))
#     summaries.append(tf.summary.scalar('training/Learning Rate', learning_rate))
#     return summaries
#
#
# def update_model_scope(var, ckpt_scope, new_scope):
#     return var.op.name.replace(new_scope,'vgg_16')
#
#
# def get_init_fn(flags):
#     """Returns a function run by the chief worker to warm-start the training.
#     Note that the init_fn is only run when initializing the model during the very
#     first global step.
#     Returns:
#       An init function run by the supervisor.
#     """
#     if flags.checkpoint_path is None:
#         return None
#     # Warn the user if a checkpoint exists in the train_dir. Then ignore.
#     if tf.train.latest_checkpoint(flags.train_dir):
#         tf.logging.info(
#             'Ignoring --checkpoint_path because a checkpoint already exists in %s'
#             % flags.train_dir)
#         return None
#
#     exclusions = []
#     if flags.checkpoint_exclude_scopes:
#         exclusions = [scope.strip()
#                       for scope in flags.checkpoint_exclude_scopes.split(',')]
#
#     # TODO(sguada) variables.filter_variables()
#     variables_to_restore = []
#     for var in slim.get_model_variables():
#         excluded = False
#         for exclusion in exclusions:
#             if var.op.name.startswith(exclusion):
#                 excluded = True
#                 break
#         if not excluded:
#             variables_to_restore.append(var)
#     # Change model scope if necessary.
#     if flags.checkpoint_model_scope is not None:
#         variables_to_restore = \
#             {var.op.name.replace(flags.model_name,
#                                  flags.checkpoint_model_scope): var
#              for var in variables_to_restore}
#
#
#     if tf.gfile.IsDirectory(flags.checkpoint_path):
#         checkpoint_path = tf.train.latest_checkpoint(flags.checkpoint_path)
#     else:
#         checkpoint_path = flags.checkpoint_path
#     tf.logging.info('Fine-tuning from %s. Ignoring missing vars: %s' % (checkpoint_path, flags.ignore_missing_vars))
#
#     return slim.assign_from_checkpoint_fn(
#         checkpoint_path,
#         variables_to_restore,
#         ignore_missing_vars=flags.ignore_missing_vars)
#
#
# def get_variables_to_train(flags):
#     """Returns a list of variables to train.
#     Returns:
#       A list of variables to train by the optimizer.
#     """
#     if flags.trainable_scopes is None:
#         return tf.trainable_variables()
#     else:
#         scopes = [scope.strip() for scope in flags.trainable_scopes.split(',')]
#
#     variables_to_train = []
#     for scope in scopes:
#         variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
#         variables_to_train.extend(variables)
#     return variables_to_train


def draw_bounding_boxes(image, bboxes):
    # Convert tf.uint8 to tf.float32.
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    if image.get_shape().ndims == 3:
        image = tf.expand_dims(image, axis=0)
    if bboxes.get_shape().ndims == 2:
        bboxes = tf.expand_dims(bboxes, axis=0)
    # print('After expanding dims')
    # print('#### image: ', image, image.get_shape(), image.get_shape().ndims)
    # print('#### bboxes: ', bboxes, bboxes.get_shape(), bboxes.get_shape().ndims)
    # print('#########################################################')
    # print('image: ', image)
    # print('bboxes: ', bboxes)
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
