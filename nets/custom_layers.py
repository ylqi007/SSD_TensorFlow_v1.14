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

# from tensorflow.contrib.framework.python.ops import add_arg_scope
# from tensorflow.contrib.layers.python.layers import initializers
# from tensorflow.contrib.framework.python.ops import variables
# from tensorflow.contrib.layers.python.layers import utils
# from tensorflow.python.ops import nn
from tensorflow.python.ops import init_ops
# from tensorflow.python.ops import variable_scope


def abs_smooth(x):
    """Smoothed absolute function. Useful to compute an L1 smooth error.
    Define as:
        x^2/2           if abs(x) < 1,
        abs(x) - 0.5,   if abs(x) < 1
    We use here a differentiable definition using min(x) and abs(x).
    Clearly not optimal, but good enough for our purpose.
    """
    absx = tf.abs(x)
    minx = tf.minimum(absx, 1)
    r = 0.5 * ((absx - 1) * minx + absx)
    return r


def conv2d(inputs, filters, kernel_size, padding='SAME', trainable=True):
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
                           # kernel_initializer=tf.compat.v1.initializers.he_normal(),
                           bias_initializer=tf.zeros_initializer(),
                           kernel_regularizer=None,
                           bias_regularizer=None,
                           activity_regularizer=None,
                           kernel_constraint=None,
                           bias_constraint=None,
                           trainable=trainable,
                           name=None,
                           reuse=None)
    return net

#
# def conv2d(input, filters_shape, rate=1, padding='SAME', trainable=True, activate=True):
#     if padding == 'VALIE':
#         # pad_h, pad_w = (filter_shape[0] - 2) // 2 + 1, (filter_shape[1] - 2) // 2 + 1
#         # paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
#         # input_data = tf.pad(input, paddings, 'CONSTANT')
#         # strides = (1, 2, 2, 1)
#         # padding = 'VALID'
#         raise ValueError('No done yet.')
#     else:
#         strides = (1, 1, 1, 1)
#         padding = "SAME"
#
#     weight = tf.get_variable(name='weight',
#                              shape=filters_shape,
#                              dtype=tf.float32,
#                              trainable=True,
#                              initializer=tf.random_normal_initializer(stddev=0.01))
#     bias = tf.get_variable(name='bias',
#                            shape=filters_shape[-1],
#                            dtype=tf.float32,
#                            trainable=True,
#                            initializer=tf.constant_initializer(0.0))
#     conv = tf.nn.conv2d(input=input,
#                         filter=weight,
#                         strides=strides,
#                         padding=padding)    # data_format='NHWC'
#     conv = tf.nn.bias_add(conv, bias)
#
#     if activate:
#         conv = tf.nn.leaky_relu(conv, alpha=0.1)
#     return conv
#

def fully_connected():
    pass


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


def l2_normalization(inputs,
                     scaling=False,
                     scale_initializer=tf.initializers.ones(),
                     reuse=None,
                     variables_collections=None,
                     outputs_collections=None,
                     data_format='NHWC',
                     trainable=True,
                     scope=None):
    """
    Implement L2 normalization on every feature (i.e. spatial normalization).
    Should be extended in some near future to other dimensions, providing a more
    flexible normalization framework.
    Args:
        inputs: a 4-D tensor with dimensions [batch_size, height, width, channels].
        scaling: whether or not to add a post scaling operation along the dimensions
            which have been normalized.
        scale_initializer: An initializer for the weights.
        reuse: whether or not the layer and its variables should be reused. To be
            able to reuse the layer scope must be given.
        variables_collections: optional list of collections for all the variables or
            a dictionary containing a different list of collection per variable.
        outputs_collections: collection to add the outputs.
        data_format:  NHWC or NCHW data format.
        trainable: If `True` also add variables to the graph collection
            `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
        scope: Optional scope for `variable_scope`.
    Returns:
        A `Tensor` representing the output of the operation.
    """

    with tf.variable_scope(scope, 'L2Normalization', [inputs], reuse=reuse) as sc:
        inputs_shape = inputs.get_shape()
        inputs_rank = inputs_shape.ndims
        dtype = inputs.dtype.base_dtype
        if data_format == 'NHWC':
            # norm_dim = tf.range(1, inputs_rank-1)
            norm_dim = tf.range(inputs_rank-1, inputs_rank)     # channel
            params_shape = inputs_shape[-1:]
        elif data_format == 'NCHW':
            # norm_dim = tf.range(2, inputs_rank)
            norm_dim = tf.range(1, 2)       # 1,
            params_shape = (inputs_shape[1])

        # Normalize along spatial dimensions.
        outputs = tf.math.l2_normalize(inputs, norm_dim, epsilon=1e-12)
        # Additional scaling.
        if scaling:
            scale_collections = tf.compat.v1.get_collection(variables_collections, 'scale')
            scale = tf.compat.v1.get_variable('gamma',
                                              shape=params_shape,
                                              dtype=dtype,
                                              initializer=scale_initializer,
                                              collections=scale_collections,
                                              trainable=trainable)
            if data_format == 'NHWC':
                outputs = tf.multiply(outputs, scale)
            elif data_format == 'NCHW':
                scale = tf.expand_dims(scale, axis=-1)
                scale = tf.expand_dims(scale, axis=-1)
                outputs = tf.multiply(outputs, scale)
                # outputs = tf.transpose(outputs, perm=(0, 2, 3, 1))
        return outputs
        # return utils.collect_named_outputs(outputs_collections,
        #                                    sc.original_name_scope, outputs)


def channel_to_last(inputs,
                    data_format='NHWC',
                    scope=None):
    """Move the channel axis to the last dimension. Allows to
    provide a single output format whatever the input data format.
    Args:
      inputs: Input Tensor;
      data_format: NHWC or NCHW.
    Return:
      Input in NHWC format.
    """
    with tf.name_scope(scope, 'channel_to_last', [inputs]):
        if data_format == 'NHWC':
            net = inputs
        elif data_format == 'NCHW':
            net = tf.transpose(inputs, perm=(0, 2, 3, 1))
        return net
