import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base

def process_images(images):
  images = tf.to_float(images) / 255.0
  images = tf.sub(images, 0.5)
  images = tf.mul(images, 2.0)
  return images


def inception_v3(images,
                 trainable=True,
                 is_training=True,
                 create_logits=True,
                 weight_decay=0.00004,
                 stddev=0.1,
                 dropout_keep_prob=0.8,
                 use_batch_norm=True,
                 batch_norm_params=None,
                 scope="InceptionV3"):
  # Only consider the inception model to be in training mode if it's trainable.
  is_inception_model_training = trainable and is_training

  if use_batch_norm:
    # Default parameters for batch normalization.
    if not batch_norm_params:
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

  with tf.variable_scope(scope, "InceptionV3", [images]) as scope:
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
        if create_logits:
          with tf.variable_scope("logits"):
            shape = net.get_shape()
            net = slim.avg_pool2d(net, shape[1:3], padding="VALID", scope="pool")
            net = slim.dropout(net,
                               keep_prob=dropout_keep_prob,
                               #is_training=is_inception_model_training,
                               scope="dropout")
            net = slim.flatten(net, scope="flatten")
            end_points["logits"] = net
  return net, end_points
