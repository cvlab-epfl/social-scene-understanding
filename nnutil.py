import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base

_RGB_MEANS = [123.68, 116.779, 103.939]

def multiscale_features(graph, names, dims, size, scope='features'):
  """
  extract features from multiple endpoints, do dimensionality
  reduction and resize to the given size
  """
  with tf.variable_scope(scope):
    endpoints = []
    for i, name in enumerate(names):
      endpoint = graph.get_tensor_by_name(name)
      if not dims is None:
        endpoint = slim.conv2d(endpoint, dims[i], 1,
                               activation_fn=None,
                               normalizer_fn=None)
      endpoint = tf.image.resize_images(endpoint, size[0], size[1])
      endpoints.append(endpoint)
  return tf.concat(3, endpoints)

def pairwise_concat(inputs, num_pairs):
  def concat_single(x):
    return tf.concat(1, [tf.tile(tf.expand_dims(x, 0), [num_pairs,1]), inputs])
  return tf.map_fn(concat_single, inputs)

def resize_concat(values, height, width,
                  resize_method=tf.image.ResizeMethod.BILINEAR):
  """Resizes multiple 4D tensors and stack them together
  Args:
    values - a sequence of 4D tensors
    height - new height
    width - new width
  """
  resized = [tf.image.resize_images(value, height, width, resize_method)
             for value in values]
  return tf.concat(3, resized)

def pairwise_distance(X):
  """
  computes pairwise distance between each point
  Args:
    X - [N,D] matrix representing N D-dimensional vectors
  Returns:
    [N,N] matrix of euclidean distances
  """
  r = tf.reduce_sum(X * X, 1)
  r = tf.reshape(r, [-1, 1])
  dist = r - 2 * tf.matmul(X, tf.transpose(X)) + tf.transpose(r)
  return dist

def cdist(X, Y):
  """
  computes pairwise distance between each point
  Args:
    X - [N,D] matrix
    Y - [M,D] matrix
  Returns:
    [N,M] matrix of euclidean distances
  """
  # NOTE: overflow possible
  rx = tf.reduce_sum(tf.square(X), 1)
  ry = tf.reduce_sum(tf.square(Y), 1)
  rx = tf.reshape(rx, [-1, 1])
  ry = tf.reshape(ry, [-1, 1])
  dist = rx - 2.0 * tf.matmul(X, tf.transpose(Y)) + tf.transpose(ry)
  return dist


def resnet_features(images, is_training=False):
  with tf.variable_scope('features') as scope:
    num_blocks = [3, 4, 6, 3]
    resnet_logits = resnet.inference(resnet._imagenet_preprocess(images),
                                     is_training,
                                     num_classes=None,
                                     num_blocks=num_blocks)
    names = ['%s/scale%d/block%d/Relu:0' % (scope.name, scale, num_blocks[scale-2])
             for scale in range(2, 6)]
    tensors = map(tf.get_default_graph().get_tensor_by_name, names)
  return list(tensors)


def inception_v3_features(images,
                          trainable=True,
                          is_training=True,
                          weight_decay=0.00004,
                          stddev=0.1,
                          dropout_keep_prob=0.8,
                          use_batch_norm=True,
                          scope='InceptionV3'):
  """Builds an Inception V3 subgraph for image embeddings.

  Args:
    images: A float32 Tensor of shape [batch, height, width, channels].
    trainable: Whether the inception submodel should be trainable or not.
    is_training: Boolean indicating training mode or not.
    weight_decay: Coefficient for weight regularization.
    stddev: The standard deviation of the trunctated normal weight initializer.
    dropout_keep_prob: Dropout keep probability.
    use_batch_norm: Whether to use batch normalization.
    batch_norm_params: Parameters for batch normalization. See
      tf.contrib.layers.batch_norm for details.
    scope: Optional Variable scope.

  Returns:
    end_points: A dictionary of activations from inception_v3 layers.
  """
  # Only consider the inception model to be in training mode if it's trainable.
  is_inception_model_training = trainable and is_training

  if use_batch_norm:
    batch_norm_params = {
        "is_training": is_inception_model_training,
        "trainable": trainable,
        # Decay for the moving averages.
        "decay": 0.9997,
        # Epsilon to prevent 0s in variance.
        "epsilon": 0.001,
        # Collection containing the moving mean and moving variance.
        "variables_collections": {
            "beta": None,
            "gamma": None,
            "moving_mean": ["moving_vars"],
            "moving_variance": ["moving_vars"],
        }
    }
  else:
    batch_norm_params = None

  if trainable:
    weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  else:
    weights_regularizer = None

  with tf.variable_scope(scope, 'InceptionV3', [images]) as scope:
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_regularizer=weights_regularizer,
        trainable=trainable):
      with slim.arg_scope(
          [slim.conv2d],
          weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
          activation_fn=tf.nn.relu,
          normalizer_fn=slim.batch_norm,
          normalizer_params=batch_norm_params):
        net, end_points = inception_v3_base(images, scope=scope)

  return end_points

def process_images_inception(images, is_training):
  images = tf.to_float(images) / 255.0
  images = tf.sub(images, 0.5)
  images = tf.mul(images, 2.0)
  return images

def normalization(inputs, epsilon=1e-3, has_shift=True, has_scale=True,
                  activation_fn=None, scope='normalization'):
  with tf.variable_scope(scope):
    inputs_shape = inputs.get_shape()
    inputs_rank = inputs_shape.ndims
    axis = list(range(inputs_rank - 1))
    mean, variance = tf.nn.moments(inputs, axis)

    shift, scale = None, None
    if has_shift:
      shift = tf.get_variable('shift',
                              shape=inputs_shape[-1:],
                              dtype=inputs.dtype,
                              initializer=tf.zeros_initializer)
    if has_scale:
      scale = tf.get_variable('scale',
                              shape=inputs_shape[-1:],
                              dtype=inputs.dtype,
                              initializer=tf.ones_initializer)
      x = tf.nn.batch_normalization(inputs, mean, variance, shift, scale, epsilon)
    return x if activation_fn is None else activation_fn(x)

def block(x, num_outputs,
          is_training=True,
          block_stride=1,
          activation_fn=tf.nn.relu):
  """Builds a single residual block"""
  shortcut = x

  with tf.variable_scope('a'):
    x = slim.conv2d(x, num_outputs, [3, 3], block_stride,
                    activation_fn=activation_fn)
    x = normalization(x)

  with tf.variable_scope('b'):
    x = slim.conv2d(x, num_outputs, [3, 3], 1,
                    activation_fn=activation_fn)
    x = normalization(x)

  with tf.variable_scope('shortcut'):
    if shortcut.get_shape()[-1] != num_outputs or block_stride != 1:
      shortcut = slim.conv2d(shortcut, num_outputs, [1, 1], block_stride)
      shortcut = normalization(shortcut)

  return activation_fn(x + shortcut)

def stack(x, num_blocks, num_outputs, downsample=True):
  """Stacks multiuple residual blocks together"""
  for i in range(num_blocks):
    with tf.variable_scope('block%d' % i):
      stride = 2 if i == 0 and downsample else 1
      x = block(x, num_outputs=num_outputs, block_stride=stride)
  return x


def l1_robust_loss(predictions, targets, name=None):
  with tf.name_scope(name, 'HuberLoss', [predictions, targets]):
    delta = predictions - targets
    return tf.select(tf.abs(delta) < 1,
                     0.5 * tf.square(delta),
                     tf.abs(delta) - 0.5)
