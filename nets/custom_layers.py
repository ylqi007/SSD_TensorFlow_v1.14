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
"""Implement some custom layers, not provided by TensorFlow.
Trying to follow as much as possible the style/standards used in
tf.contrib.layers
"""
import tensorflow as tf


def conv2d(inputs, filters, kernel_size):
    # with tf.variable_scope(name_scope):
    net = tf.layers.conv2d(inputs=inputs,
                           filters=filters,
                           kernel_size=kernel_size,
                           strides=(1, 1),
                           padding='SAME',
                           data_format='channels_last',
                           dilation_rate=(1, 1),
                           activation=tf.nn.leaky_relu,
                           use_bias=True,
                           kernel_initializer=tf.glorot_uniform_initializer(),
                           bias_initializer=tf.zeros_initializer(),
                           kernel_regularizer=None,
                           bias_regularizer=None,
                           activity_regularizer=None,
                           kernel_constraint=None,
                           bias_constraint=None,
                           trainable=True,
                           name=None,
                           reuse=None)
    return net


def max_pool2d(inputs, pool_size=[2, 2], strides=[2, 2], padding='same', scope=None, data_format='channels_last'):
    with tf.name_scope(scope):
        outputs = tf.layers.max_pooling2d(inputs,
                                          pool_size,
                                          strides,
                                          padding=padding,
                                          data_format='channels_last',
                                          name=None)
    return outputs


def pad2d(inputs,
          pad=(0, 0),
          mode='CONSTANT',
          data_format='NHWC',
          trainable=True,
          scope=None):
    """2D Padding layer, adding a symmetric padding to H and W dimensions.
    Aims to mimic padding in Caffe and MXNet, helping the port of models to
    TensorFlow. Tries to follow the naming convention of `tf.contrib.layers`.
    Args:
      inputs: 4D input Tensor;
      pad: 2-Tuple with padding values for H and W dimensions;
      mode: Padding mode. C.f. `tf.pad`
      data_format:  NHWC or NCHW data format.
    """
    with tf.name_scope(scope, 'pad2d', [inputs]):
        # Padding shape.
        if data_format == 'NHWC':
            paddings = [[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]]
        elif data_format == 'NCHW':
            paddings = [[0, 0], [0, 0], [pad[0], pad[0]], [pad[1], pad[1]]]
        net = tf.pad(inputs, paddings, mode=mode)
        return net



