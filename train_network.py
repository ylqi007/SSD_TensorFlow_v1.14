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
import tensorflow as tf
from PIL import Image

from datasets import dataset_factory


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
parser.add_argument('--dataset_name',
                    default='pascalvoc',
                    help='The name of dataset to load.')
parser.add_argument('--num_classes',
                    default=21,
                    help='Number of classes to use in the dataset.')
parser.add_argument('--dataset_split_name',
                    default='train',
                    help='The name of train/test split.')
parser.add_argument('--dataset_dir',
                    default=None,
                    help='The directory where the tfrecord files are stored.')

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
        # Config model_deploy. Keep TF Slim Models structure.
        # Useful if want to need multiple GPUs and/or servers in the future.
        # deploy_config = model_deploy.DeploymentConfig(
        #     num_clones=FLAGS.num_clones,
        #     clone_on_cpu=FLAGS.clone_on_cpu,
        #     replica_id=0,
        #     num_replicas=1,
        #     num_ps_tasks=0)

        # Create global_step.
        with tf.device('/cpu:0'):
            global_step = tf.compat.v1.train.create_global_step()
        print(global_step)

        # Select the dataset.
        dataset = dataset_factory.get_dataset(args.dataset_name,
                                              args.dataset_split_name,
                                              args.dataset_dir)

        print('dataset: ', dataset)
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        # print('iterator: ', iterator)
        # sample = iterator.get_next()
        # print('sample: ', sample)
        # print('samples: ', *sample)

        sample = iterator.get_next()
        difficult = sample['image/object/bbox/difficult']
        label = sample['image/object/bbox/label']
        truncated = sample['image/object/bbox/truncated']
        xmin = sample['image/object/bbox/xmin']
        ymin = sample['image/object/bbox/ymin']
        xmax = sample['image/object/bbox/xmax']
        ymax = sample['image/object/bbox/ymax']
        channels = sample['image/channels']
        _format = sample['image/format']
        height = sample['image/height']
        width = sample['image/width']
        image = sample['image/raw_data']
        shape = sample['image/shape']

        new_info1 = tf.concat([difficult, truncated, label], axis=0)
        print('iterator: ', iterator)
        with tf.Session() as sess:
            try:
                while True:
                    print('\n======================================================\n')
                    _features = [difficult, truncated, label,
                                 xmin, ymin, xmax, ymax,
                                 channels, _format, height, width, image, shape]
                    print('_features: ', _features)
                    features = sess.run(_features)
                    img = Image.fromarray(features[-2])
                    img.show()
                    print(image)
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


if __name__ == '__main__':
    main()