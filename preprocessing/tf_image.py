# Copyright 2015 The TensorFlow Authors and Paul Balanca. All Rights Reserved.
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
"""Custom image operations.
Most of the following methods extend TensorFlow image library, and part of
the code is shameless copy-paste of the former!
"""
import tensorflow as tf

#
# def _ImageDimensions(image):
#     """Returns the dimensions of an image tensor.
#     Args:
#       image: A 3-D Tensor of shape `[height, width, channels]`.
#     Returns:
#       A list of `[height, width, channels]` corresponding to the dimensions of the
#         input image.  Dimensions that are statically known are python integers,
#         otherwise they are integer scalar tensors.
#     """
#     if image.get_shape().is_fully_defined():
#         return image.get_shape().as_list()
#     else:
#         static_shape = image.get_shape().with_rank(3).as_list()
#         dynamic_shape = array_ops.unstack(array_ops.shape(image), 3)
#         return [s if s is not None else d
#                 for s, d in zip(static_shape, dynamic_shape)]
#
#
# def resize_image(image, size,
#                  method=tf.image.ResizeMethod.BILINEAR,
#                  align_corners=False):
#     """Resize an image and bounding boxes.
#     """
#     # Resize image.
#     with tf.name_scope('resize_image'):
#         height, width, channels = _ImageDimensions(image)
#         image = tf.expand_dims(image, 0)
#         image = tf.image.resize_images(image, size,
#                                        method, align_corners)
#         image = tf.reshape(image, tf.stack([size[0], size[1], channels]))
#         return image
