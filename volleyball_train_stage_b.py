# coding: utf-8

# In[4]:

import matplotlib.pyplot as plt
import datetime
import os

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


# In[5]:

## config
class Config(object):

  def __init__(self):

    # shared
    self.image_size = 720, 1280
    self.out_size = 87, 157
    self.batch_size = 4
    self.num_boxes = 12
    self.epsilon = 1e-5
    self.features_multiscale_names = ['Mixed_5d', 'Mixed_6e']
    self.train_inception = False

    # DetNet
    self.build_detnet = False
    self.num_resnet_blocks = 1
    self.num_resnet_features = 512
    self.reg_loss_weight = 10.0
    self.nms_kind = 'greedy'

    # ActNet
    self.use_attention = False
    self.crop_size = 5, 5
    self.num_features_boxes = 4096
    self.num_actions = 9
    self.num_activities = 8
    self.actions_loss_weight = 4.0
    self.actions_weights = [[ 1., 1., 2., 3., 1., 2., 2., 0.2, 1.]]

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

src_model_config = '<path-to-your-stage-a-config-pickle>'

# we load config from the first stage
c = pickle.load(open(src_model_config, 'rb'))

c.tag = c.tag.replace('single', 'temporal')
# smaller batch size for the temporal
c.batch_size = 4
# not finetuning inception at training
c.train_inception = False
# don't need detection net during training
c.build_detnet = False
# 'hidden', 'hidden-soft', 'boxes-soft'
c.match_kind = 'hidden-soft'
c.tag += '-match-' + c.match_kind

c.src_model_path = c.out_model_path
c.out_model_path = c.models_path + '/activity/volleyball/stage-b-%s.ckpt' % c.tag
c.out_config_path = c.models_path + '/activity/volleyball/config-%s.pkl' % c.tag


# In[6]:

## reading the dataset
train = volley_read_dataset(c.data_path, TRAIN_SEQS + VAL_SEQS)
train_frames = volley_all_frames(train)
test = volley_read_dataset(c.data_path, TEST_SEQS)
test_frames = volley_all_frames(test)
src_image_size = pickle.load(open(c.data_path + '/src_image_size.pkl', 'rb'))

all_anns = {**train, **test}
all_frames = train_frames + test_frames
all_tracks = pickle.load(open(c.data_path + '/tracks_normalized.pkl', 'rb'))


# In[7]:

# data loading utils
def _frames_around(frame, num_before=5, num_after=4):
  sid, src_fid = frame
  return [(sid, src_fid, fid)
          for fid in range(src_fid-num_before, src_fid+num_after+1)]

def load_samples_sequence(anns, tracks, images_path, frames, num_boxes=12):
  # NOTE: this assumes we got the same # of boxes for this batch
  images, boxes, boxes_idx = [], [], []
  activities, actions = [], []
  for i, (sid, src_fid, fid) in enumerate(frames):
    images.append(skimage.io.imread(images_path + '/%d/%d/%d.jpg' %
                                    (sid, src_fid, fid)))

    boxes.append(tracks[(sid, src_fid)][fid])
    actions.append(anns[sid][src_fid]['actions'])
    if len(boxes[-1]) != num_boxes:
      boxes[-1] = np.vstack([boxes[-1], boxes[-1][:num_boxes-len(boxes[-1])]])
      actions[-1] = actions[-1] + actions[-1][:num_boxes-len(actions[-1])]
    boxes_idx.append(i * np.ones(num_boxes, dtype=np.int32))
    activities.append(anns[sid][src_fid]['group_activity'])


  images = np.stack(images)
  activities = np.array(activities, dtype=np.int32)
  bboxes = np.vstack(boxes).reshape([-1, num_boxes, 4])
  bboxes_idx = np.hstack(boxes_idx).reshape([-1, num_boxes])
  actions = np.hstack(actions).reshape([-1, num_boxes])

  return images, activities, bboxes, bboxes_idx, actions


# In[10]:

# In[ ]:

tf.reset_default_graph()

with tf.device('/gpu:0'):
  H, W = c.image_size
  OH, OW = c.out_size
  B, T, N = c.batch_size, c.num_frames, c.num_boxes
  NFB, NFH = c.num_features_boxes, c.num_features_hidden
  EPS = c.epsilon

  # each batch is a single (small) sequence
  images_in = tf.placeholder(tf.uint8, [B,T,H,W,3], 'images_in')
  boxes_in = tf.placeholder(tf.float32, [B,T,N,4], 'boxes_in')
  boxes_idx_in = tf.placeholder(tf.int32, [B,T,N,], 'boxes_idx_in')
  actions_in = tf.placeholder(tf.int32, [B,T,N,], 'actions_in')
  activities_in = tf.placeholder(tf.int32, [B,T,], 'activities_in')
  dropout_keep_prob_in = tf.placeholder(tf.float32, [], 'dropout_keep_prob_in')

  images_in_flat = tf.reshape(images_in, [B*T,H,W,3])
  boxes_in_flat = tf.reshape(boxes_in, [B*T*N,4])
  boxes_idx_in_flat = tf.reshape(boxes_idx_in, [B*T*N,])
  actions_in_flat = tf.reshape(actions_in, [B*T*N,])
  activities_in_flat = tf.reshape(activities_in, [B*T,])

  # TODO: only construct inception until a certain level
  _, inception_endpoints = inception.inception_v3(inception.process_images(images_in_flat),
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
    # TODO: instead of boxes_in
    seg_preds, reg_preds, boxes_proposals, detections = det_net(features_multiscale,
                                                                c.num_resnet_blocks,
                                                                c.num_resnet_features,
                                                                N,
                                                                [H, W],
                                                                c.nms_kind)
    boxes_preds, boxes_confidence = detections
    det_net_vars = tf.get_collection(tf.GraphKeys.VARIABLES, 'DetNet')


  with tf.variable_scope('ActNet'):
    boxes_flat = boxes_in_flat
    boxes_idx_flat = boxes_idx_in_flat
#     boxes_flat = tf.reshape(boxes_preds, [B*N,4])
#     # TODO: double-check
#     boxes_idx_flat = tf.tile(tf.range(0, B)[:,tf.newaxis], [1, N*T])
    boxes_features_multiscale = tf.image.crop_and_resize(features_multiscale,
                                                         boxes_flat,
                                                         boxes_idx_flat,
                                                         c.crop_size)
    boxes_features_multiscale_flat = slim.flatten(boxes_features_multiscale)

    with tf.variable_scope('shared'):
      boxes_features_flat = slim.fully_connected(boxes_features_multiscale_flat, NFB)
      boxes_features_flat_dropout = slim.dropout(boxes_features_flat, dropout_keep_prob_in)
    shared_vars = tf.get_collection(tf.GraphKeys.VARIABLES, 'ActNet/shared')

    with tf.variable_scope('sequence'):
      # embedding to "hidden space"
      boxes_hidden_flat = slim.fully_connected(boxes_features_flat_dropout, NFH, tf.nn.tanh)
      boxes_hidden = tf.reshape(boxes_hidden_flat, [B,T,N,NFH])

      def _construct_sequence(batch):
        hidden, boxes = batch
        # initializing the state with features
        states = [hidden[0]]
        # TODO: make this dependent on the data
        # TODO: make it with scan ?
        for t in range(1, T):
          # find the matching boxes. TODO: try with the soft matching function
          if c.match_kind == 'boxes':
            dists = nnutil.cdist(boxes[t-1], boxes[t])
            idxs = tf.argmin(dists, 1, 'idxs')
            state_prev = tf.gather(states[t-1], idxs)
          elif c.match_kind == 'hidden':
            # TODO: actually it makes more sense to compare on states
            dists = nnutil.cdist(hidden[t-1], hidden[t])
            idxs = tf.argmin(dists, 1, 'idxs')
            state_prev = tf.gather(states[t-1], idxs)
          elif c.match_kind == 'hidden-soft':
            dists = nnutil.cdist(hidden[t-1], hidden[t])
            weights = slim.softmax(dists)
            state_prev = tf.matmul(weights, states[t-1])
          else:
            raise RuntimeError('Unknown match_kind: %s' % c.match_kind)

          def _construct_update(reuse):
            state = tf.concat(1, [state_prev, hidden[t]])
            # TODO: initialize jointly
            reset = slim.fully_connected(state, NFH, tf.nn.sigmoid,
                                         reuse=reuse,
                                         scope='reset')
            step = slim.fully_connected(state, NFH, tf.nn.sigmoid,
                                        reuse=reuse,
                                        scope='step')
            state_r = tf.concat(1, [reset * state_prev, hidden[t]])
            state_up = slim.fully_connected(state_r, NFH, tf.nn.tanh,
                                            reuse=reuse,
                                            scope='state_up')
            return state_up, step
          try:
            state_up, step = _construct_update(reuse=True)
          except ValueError:
            state_up, step = _construct_update(reuse=False)

          state = step * state_up + (1.0 - step) * state_prev
          states.append(state)
        return tf.pack(states)

      boxes_states = tf.map_fn(_construct_sequence,
                               [boxes_hidden, boxes_in],
                               dtype=np.float32)

      # prediction!
      # for each of the states, we reuse the same weights
      with tf.variable_scope('actions_hidden'):
        boxes_states_flat = tf.reshape(boxes_states, [-1, NFH])
        actions_logits = slim.fully_connected(boxes_states_flat,
                                              c.num_actions,
                                              None)
        actions_preds = slim.softmax(actions_logits, 'preds')
        actions_in_one_hot = slim.one_hot_encoding(actions_in_flat, c.num_actions)
        actions_loss = - tf.reduce_mean(tf.constant(c.actions_weights) *
                                        actions_in_one_hot * tf.log(actions_preds + EPS))
        actions_labels = tf.argmax(actions_logits, 1)
        actions_accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.to_int32(actions_labels),
                                                               actions_in_flat)))

      with tf.variable_scope('activities_hidden'):
        boxes_states_pooled = tf.reduce_max(boxes_states, [2])
        boxes_states_pooled_flat = tf.reshape(boxes_states_pooled, [-1, NFH])

        # TODO: we should be able to
        activities_logits = slim.fully_connected(boxes_states_pooled_flat,
                                                 c.num_activities,
                                                 None)
        activities_labels = tf.argmax(activities_logits, 1)
        activities_accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.to_int32(activities_labels),
                                                                  activities_in_flat)))
        activities_preds = slim.softmax(activities_logits, 'preds')
        activities_in_one_hot = slim.one_hot_encoding(activities_in_flat, c.num_activities)
        activities_loss = - tf.reduce_mean(activities_in_one_hot * tf.log(activities_preds + EPS))

        activities_avg_preds = tf.reduce_mean(tf.reshape(activities_preds, [B,T,c.num_activities]), [1])
        activities_avg_labels = tf.to_int32(tf.argmax(activities_avg_preds, 1))

        activities_avg_accuracy = tf.reduce_mean(tf.to_float(tf.equal(activities_avg_labels,
                                                                      activities_in[:,5])))

    sequence_vars = tf.get_collection(tf.GraphKeys.VARIABLES, 'ActNet/sequence')

  with tf.variable_scope('train'):
    global_step = slim.create_global_step()
    learning_rate = c.train_learning_rate
    total_loss = activities_loss + c.actions_loss_weight * actions_loss
    train_op = slim.optimize_loss(total_loss,
                                  global_step,
                                  learning_rate,
                                  tf.train.AdamOptimizer)
  train_vars = tf.get_collection(tf.GraphKeys.VARIABLES, 'train')


# TODO: generate the tag
print('finished building the model %s' % c.out_model_path)
pickle.dump(c, open(c.out_config_path, 'wb'))


# In[ ]:

# In[ ]:

## training

tf_config = tf.ConfigProto()
tf_config.log_device_placement = True
tf_config.gpu_options.allow_growth = True
tf_config.allow_soft_placement = True

np.random.seed(c.train_random_seed)
tf.set_random_seed(c.train_random_seed)

with tf.Session(config=tf_config) as sess:

  print('loading pre-trained model...')
  restorer = tf.train.Saver(inception_vars + shared_vars)
  restorer.restore(sess, c.src_model_path)
  print('done!')

  print('initializing the variables...')
  sess.run(tf.initialize_variables(sequence_vars +
                                   train_vars))
  print('done!')

  saver = tf.train.Saver()

  fetches = [
    train_op,
    actions_loss,
    actions_accuracy,
    actions_preds,
    activities_loss,
    activities_accuracy,
    activities_preds,
    global_step,
  ]

  for step in range(1, c.train_num_steps):
    p = np.random.choice(len(train_frames), size=c.batch_size)
    batch_frames = sum([_frames_around(train_frames[i],
                                       c.num_before,
                                       c.num_after)
                        for i in p], [])
    batch = load_samples_sequence(all_anns, all_tracks, c.images_path, batch_frames)
    batch = [b.reshape((c.batch_size,c.num_frames) + b.shape[1:]) for b in batch]

    feed_dict = {
      images_in : batch[0],
      activities_in : batch[1],
      boxes_in : batch[2],
      boxes_idx_in : batch[3],
      actions_in : batch[4],
      dropout_keep_prob_in : c.train_dropout_prob,
    }

    outputs = sess.run(fetches, feed_dict)

    if step % 5 == 0:
      ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
      print('%s step:%d ' % (ts, outputs[-1]) +
            'actv.L: %.4f, actv.A: %.4f ' % (outputs[4], outputs[5]) +
            'actn.L: %.4f, actn.A: %.4f ' % (outputs[1], outputs[2]))

    if step % c.train_save_every_steps == 0:
      print('saving the model at %d steps' % step)
      saver.save(sess, c.out_model_path, global_step)
      print('done!')

  print('saving the final model...')
  saver.save(sess, c.out_model_path)
  print('done!')
