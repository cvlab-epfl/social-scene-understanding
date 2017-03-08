# coding: utf-8
import matplotlib.pyplot as plt

import datetime
import numpy as np

import skimage.io
import skimage.transform

import tensorflow as tf
import tensorflow.contrib.slim as slim

import inception
import pickle

from detnet import det_net, det_net_loss
import nnutil
import volleyball
from volleyball import *

import os

## config
class Config(object):

  def __init__(self):

    # shared
    self.image_size = 720, 1280
    self.out_size = 87, 157
    self.batch_size = 8
    self.num_boxes = 12
    self.epsilon = 1e-5
    self.features_multiscale_names = ['Mixed_5d', 'Mixed_6e']
    self.train_inception = True

    # DetNet
    self.build_detnet = True
    self.num_resnet_blocks = 1
    self.num_resnet_features = 512
    self.reg_loss_weight = 10.0
    self.nms_kind = 'greedy'

    # ActNet
    self.crop_size = 5, 5
    self.num_features_boxes = 2048
    self.num_actions = 9
    self.num_activities = 8
    self.actions_loss_weight = 0.5
    self.actions_weights = [[ 1., 1., 2., 3., 1., 2., 2., 0.2, 1.]]

    self.attention_kind = 'maxpool'
    self.attention_tau = 10.0
    self.attention_num_hidden = 512

    # sequence
    self.num_features_hidden = 1024
    self.num_frames = 10
    self.num_before = 5
    self.num_after = 4

    # training parameters
    self.train_num_steps = 5000
    self.train_random_seed = 0
    self.train_learning_rate = 1e-5
    self.train_dropout_prob = 0.8
    self.train_save_every_steps = 500
    self.train_max_to_keep = 8

c = Config()
# NOTE: you have to fill this
c.models_path = '<path-to-models>'
c.data_path = '<path-to-data>'
# reading images of a certain resolution
c.images_path = '/%dp' % c.image_size[0]
# you can download pre-trained models at
# http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
c.inception_model_path = c.models_path + '/imagenet/inception_v3.ckpt'

## reading the dataset
train = volley_read_dataset(c.data_path, TRAIN_SEQS + VAL_SEQS)
train_frames = volley_all_frames(train)
test = volley_read_dataset(c.data_path, TEST_SEQS)
test_frames = volley_all_frames(test)
all_anns = {**train, **test}
all_frames = train_frames + test_frames

# these you can get in the repository
src_image_size = pickle.load(open(c.data_path + '/src_image_size.pkl', 'rb'))
all_tracks = pickle.load(open(c.data_path + '/tracks_normalized.pkl', 'rb'))


def load_samples_full(anns, tracks, images_path, frames, num_boxes=12):
  # NOTE: this assumes we got the same # of boxes for this batch
  images, reg_masks, seg_masks = [], [], []
  boxes, boxes_idx, activities, actions = [], [], [], []

  for i, (sid, src_fid) in enumerate(frames):
    # TODO: change me for sequence
    fid = src_fid
    images.append(skimage.io.imread(images_path + '/%d/%d/%d.jpg' %
                                    (sid, src_fid, fid)))
    image_size = images[-1].shape[:2]

    yx = np.mgrid[0:image_size[0],0:image_size[1]]

    bbs = np.copy(tracks[(sid, src_fid)][fid])
    bbs[:,[0,2]] = bbs[:,[0,2]].astype(np.float32) * image_size[0]
    bbs[:,[1,3]] = bbs[:,[1,3]].astype(np.float32) * image_size[1]
    bbs = bbs.astype(np.int32)

    reg_mask = np.zeros(image_size + (4,), dtype=np.float32)
    for y0,x0,y1,x1 in bbs:
      reg_mask[y0:y1,x0:x1,0] = (yx[0][y0:y1,x0:x1] - y0) / image_size[0]
      reg_mask[y0:y1,x0:x1,1] = (yx[1][y0:y1,x0:x1] - x0) / image_size[1]
      reg_mask[y0:y1,x0:x1,2] = (y1 - yx[0][y0:y1,x0:x1]) / image_size[0]
      reg_mask[y0:y1,x0:x1,3] = (x1 - yx[1][y0:y1,x0:x1]) / image_size[1]
    seg_mask = np.any(reg_mask, axis=2).astype(np.uint8)
    reg_masks.append(reg_mask)
    seg_masks.append(seg_mask)

    boxes.append(tracks[(sid, src_fid)][fid])
    actions.append(anns[sid][src_fid]['actions'])
    if len(boxes[-1]) != num_boxes:
      boxes[-1] = np.vstack([boxes[-1], boxes[-1][:num_boxes-len(boxes[-1])]])
      actions[-1] = actions[-1] + actions[-1][:num_boxes-len(actions[-1])]
    boxes_idx.append(i * np.ones(num_boxes, dtype=np.int32))
    activities.append(anns[sid][src_fid]['group_activity'])

  images = np.stack(images)
  activities = np.array(activities, dtype=np.int32)
  boxes = np.vstack(boxes).reshape([-1, num_boxes, 4])
  boxes_idx = np.hstack(boxes_idx).reshape([-1, num_boxes])
  actions = np.hstack(actions).reshape([-1, num_boxes])
  reg_masks = np.stack(reg_masks)
  seg_masks = np.stack(seg_masks)

  return images, activities, boxes, boxes_idx, actions, reg_masks, seg_masks


# ### joint model graph definition

# In[20]:

def attention_weights(features, tau=10.0, num_hidden=512):
  """computing attention weights
  Args:
    features: [B,N,F]
  Returns:
    [B,N] tensor with soft attention weights for each sample
  """
  B, N, F = features.get_shape().as_list()

  with tf.variable_scope('attention'):
    x = tf.reshape(features, [-1, F])
    x = slim.fully_connected(x, num_hidden, scope='fc0')
    x = slim.fully_connected(x, 1, activation_fn=None, scope='fc1')
    x = tf.reshape(x, features.get_shape()[:2])
    alpha = tf.reshape(slim.softmax(x / tau), [B,N,])
  return alpha

tf.reset_default_graph()

with tf.device('/gpu:0'):
  H, W = c.image_size
  OH, OW = c.out_size
  B, T, N = c.batch_size, c.num_frames, c.num_boxes
  NFB, NFH = c.num_features_boxes, c.num_features_hidden
  EPS = c.epsilon

  images_in = tf.placeholder(tf.uint8, [B,H,W,3], 'images_in')
  seg_masks_in = tf.placeholder(tf.uint8, [B,H,W], 'seg_masks_in')
  reg_masks_in = tf.placeholder(tf.float32, [B,H,W,4], 'reg_masks_in')
  boxes_gt_in = tf.placeholder(tf.float32, [B,N,4], 'boxes_gt_in')
  boxes_gt_idx_in = tf.placeholder(tf.int32, [B,N,], 'boxes_gt_idx_in')
  activities_in = tf.placeholder(tf.int32, [B,], 'activities_in')
  actions_in = tf.placeholder(tf.int32, [B,N,], 'actions_in')
  dropout_keep_prob_in = tf.placeholder(tf.float32, [], 'dropout_keep_prob_in')

  # extracting multiscale features
  _, inception_endpoints = inception.inception_v3(inception.process_images(images_in),
                                                  trainable=c.train_inception,
                                                  is_training=False,
                                                  create_logits=False,
                                                  scope='InceptionV3')
  inception_vars = tf.get_collection(tf.GraphKeys.VARIABLES, 'InceptionV3')

  # extracting multiscale features
  features_multiscale = []
  for name in c.features_multiscale_names:
    features = inception_endpoints[name]
    if features.get_shape()[1:3] != tf.TensorShape([OH, OW]):
      features = tf.image.resize_images(features, OH, OW)
    features_multiscale.append(features)
  features_multiscale = tf.concat(3, features_multiscale)

  if c.build_detnet:
    seg_preds, reg_preds, boxes_proposals, detections = det_net(features_multiscale,
                                                                c.num_resnet_blocks,
                                                                c.num_resnet_features,
                                                                N,
                                                                [H, W],
                                                                c.nms_kind)
    boxes_preds, boxes_confidence = detections
    det_net_vars = tf.get_collection(tf.GraphKeys.VARIABLES, 'DetNet')

  with tf.variable_scope('ActNet'):
    boxes_flat = tf.reshape(boxes_gt_in, [B*N,4])
    # TODO: construct index on-the-fly
    boxes_idx = tf.tile(tf.range(0, B)[:,tf.newaxis], [1, N])
    boxes_idx_flat = tf.reshape(boxes_idx, [B*N,])
    actions_in_flat = tf.reshape(actions_in, [B*N,])
    boxes_features_multiscale = tf.image.crop_and_resize(features_multiscale,
                                                         boxes_flat,
                                                         boxes_idx_flat,
                                                         c.crop_size)
    boxes_features_multiscale_flat = slim.flatten(boxes_features_multiscale)

    with tf.variable_scope('shared'):
      boxes_features_flat = slim.fully_connected(boxes_features_multiscale_flat, NFB)
      boxes_features_flat_dropout = slim.dropout(boxes_features_flat, dropout_keep_prob_in)
      boxes_features = tf.reshape(boxes_features_flat, [B,N,-1])

    with tf.variable_scope('actions'):
      actions_logits = slim.fully_connected(boxes_features_flat,
                                            c.num_actions,
                                            activation_fn=None)
      actions_preds = slim.softmax(actions_logits, 'preds')
      actions_in_one_hot = slim.one_hot_encoding(actions_in_flat, c.num_actions)
      actions_loss = - tf.reduce_mean(tf.constant(c.actions_weights) *
                                      actions_in_one_hot * tf.log(actions_preds + EPS))
      actions_labels = tf.argmax(actions_logits, 1)
      actions_accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.to_int32(actions_labels),
                                                             actions_in_flat)))

    with tf.variable_scope('activities'):
      if c.attention_kind == 'attention':
        weights = attention_weights(boxes_features, c.attention_tau, c.attention_num_hidden)[:,:,tf.newaxis]
        boxes_features_pooled = tf.reduce_sum(weights * boxes_features, [1])
      elif c.attention_kind == 'maxpool':
        boxes_features_pooled = tf.reduce_max(boxes_features, [1])
      else:
        raise RuntimeError('Unknown attention kind: %s' % c.attention_kind)
      # TODO: we should be able to
      activities_logits = slim.fully_connected(boxes_features_pooled,
                                               c.num_activities,
                                               activation_fn=None)
      activities_labels = tf.argmax(activities_logits, 1)
      activities_accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.to_int32(activities_labels),
                                                                activities_in)))
      activities_preds = slim.softmax(activities_logits, 'preds')
      activities_in_one_hot = slim.one_hot_encoding(activities_in, c.num_activities)
      activities_loss = - tf.reduce_mean(activities_in_one_hot * tf.log(activities_preds + EPS))

  act_net_vars = tf.get_collection(tf.GraphKeys.VARIABLES, 'ActNet')

  with tf.variable_scope('train'):
    global_step = slim.create_global_step()

    detection_loss = det_net_loss(seg_masks_in, reg_masks_in,
                                  seg_preds, reg_preds,
                                  c.reg_loss_weight,
                                  c.epsilon)
    total_loss = (detection_loss +
                  activities_loss +
                  c.actions_loss_weight * actions_loss)
    train_op = slim.optimize_loss(total_loss,
                                  global_step,
                                  c.train_learning_rate,
                                  tf.train.AdamOptimizer)
  train_vars = tf.get_collection(tf.GraphKeys.VARIABLES, 'train')

# TODO: generate the tag
c.tag = ('single-actions-%d-%dx%d-weight-%.1f' %
         (c.num_features_boxes,
          c.crop_size[0], c.crop_size[1],
          c.actions_loss_weight))
c.out_model_path = c.models_path + '/activity/volleyball/stage-a-%s.ckpt' % c.tag
c.out_config_path = c.models_path + '/activity/volleyball/config-%s.pkl' % c.tag

print('finished building the model %s' % c.out_model_path)
pickle.dump(c, open(c.out_config_path, 'wb'))


tf_config = tf.ConfigProto()
tf_config.log_device_placement = True
tf_config.gpu_options.allow_growth = True
tf_config.allow_soft_placement = True

np.random.seed(c.train_random_seed)
tf.set_random_seed(c.train_random_seed)

with tf.Session(config=tf_config) as sess:

  print('loading pre-trained weights...')
  restorer = tf.train.Saver(inception_vars)
  restorer.restore(sess, c.inception_model_path)
  print('done!')

  print('initializing the variables...')
  saver = tf.train.Saver(max_to_keep=c.train_max_to_keep)
  sess.run(tf.initialize_variables(det_net_vars +
                                   act_net_vars +
                                   train_vars))
  print('done!')

  fetches = [
    train_op,
    actions_loss,
    actions_accuracy,
    actions_preds,
    activities_loss,
    activities_accuracy,
    activities_preds,
    total_loss,
    global_step,
  ]

  for step in range(1, c.train_num_steps):
    frames = volley_random_frames(train, c.batch_size)
    batch = load_samples_full(all_anns, all_tracks, c.images_path, frames, c.num_boxes)

    feed_dict = {
      images_in : batch[0],
      activities_in : batch[1],
      boxes_gt_in : batch[2],
      boxes_gt_idx_in : batch[3],
      actions_in : batch[4],
      reg_masks_in : batch[5],
      seg_masks_in : batch[6],
      dropout_keep_prob_in : c.train_dropout_prob,
    }

    outputs = sess.run(fetches, feed_dict)

    if step % 5 == 0:
      ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
      print('%s step:%d ' % (ts, outputs[-1]) +
            'total.L: %.4f ' % (outputs[-2]) +
            'actv.L: %.4f, actv.A: %.4f ' % (outputs[4], outputs[5]) +
            'actn.L: %.4f, actn.A: %.4f ' % (outputs[1], outputs[2]))

    if step % c.train_save_every_steps == 0:
      print('saving the model at %d steps' % step)
      saver.save(sess, c.out_model_path, global_step)
      print('done!')

  print('saving the final model...')
  saver.save(sess, c.out_model_path)
  print('done!')
