import tensorflow as tf
import tensorflow.contrib.slim as slim
import nnutil
import numpy as np


def grid_2d(in_size, out_size=None):
  grid_ys, grid_xs = tf.meshgrid(tf.range(0, in_size[0]),
                                 tf.range(0, in_size[1]),
                                 indexing='ij')

  if not out_size is None:
    grid_yxs = tf.image.resize_images(tf.pack([grid_ys, grid_xs], axis=2),
                                      out_size[0], out_size[1])
    grid_ys, grid_xs = grid_yxs[:,:,0], grid_yxs[:,:,1]

  grid_ys = grid_ys / tf.to_float(in_size[0])
  grid_xs = grid_xs / tf.to_float(in_size[1])
  return grid_ys, grid_xs

def reg_to_boxes(reg_preds, in_size, out_size):
  '''converting relative predictions to global bbox coordinates'''
  grid_ys, grid_xs = grid_2d(in_size, out_size)
  return tf.pack([grid_ys - reg_preds[:,:,:,0],
                  grid_xs - reg_preds[:,:,:,1],
                  grid_ys + reg_preds[:,:,:,2],
                  grid_xs + reg_preds[:,:,:,3]],
                 axis=3)

def create_reg_mask(boxes, size):
  yx = np.mgrid[0:size[0],0:size[1]]  
  reg_mask = np.zeros(size + (4,), dtype=np.float32)
  for y0,x0,y1,x1 in boxes:
    reg_mask[y0:y1,x0:x1,0] = (yx[0][y0:y1,x0:x1] - y0) / size[0]
    reg_mask[y0:y1,x0:x1,1] = (yx[1][y0:y1,x0:x1] - x0) / size[1] 
    reg_mask[y0:y1,x0:x1,2] = (y1 - yx[0][y0:y1,x0:x1]) / size[0]
    reg_mask[y0:y1,x0:x1,3] = (x1 - yx[1][y0:y1,x0:x1]) / size[1]
  return reg_mask

def refine_boxes(boxes, num_iters, step, sigma):
  assert num_iters > 1

  def iteration(prev, i):
    state_prev, _ = prev
    features = state_prev / sigma
    dists = tf.nn.relu(nnutil.pairwise_distance(features))
    weights = tf.exp(-dists)
    confidence = tf.reduce_sum(weights, [1], True)
    weights = weights / confidence
    state_up = tf.matmul(weights, state_prev)
    return (1.0 - step) * state_prev + step * state_up, confidence

  states = tf.scan(iteration,
                   tf.range(0, num_iters),
                   initializer=(boxes, boxes[:,0:1]))
  return states[0][-1], states[1][-1]


def compute_detections_refine_nms(seg_preds, boxes_preds, num_outputs,
                                  seg_threshold=0.2,
                                  sigma=5e-3, step=0.2, num_iters=50,
                                  iou_threshold=0.6):

  mask_flat = tf.reshape(seg_preds[:,:,1], [-1])
  boxes_flat = tf.reshape(boxes_preds, [-1, 4])

  # TODO: also collect (y,x) coordinates
  idxs = tf.where(mask_flat > seg_threshold)[:,0]
  boxes = tf.gather(boxes_flat, idxs)
  boxes, confidence = refine_boxes(boxes, num_iters, step, sigma)
  # TODO: here we want to maybe run barinova-style nms
  idxs = tf.image.non_max_suppression(boxes,
                                      confidence[:,0],
                                      max_output_size=num_outputs,
                                      iou_threshold=iou_threshold)
  # TODO: we should
  # simply repeating missing detections.
  # TODO: test this, or output zeros and mask
  num_dets = tf.shape(idxs)[0]
  # TODO: this doesn't work with high seg threshold
  idxs = tf.concat(0, [idxs, idxs[0:num_outputs-num_dets]])
  return tf.gather(boxes, idxs), tf.gather(confidence, idxs)

def compute_detections_greedy(seg_preds, boxes_preds, num_outputs,
                              seg_threshold=0.2,
                              sigma=5e-3, step=0.2, num_iters=20,
                              dist_threshold=20.0):

  mask_flat = tf.reshape(seg_preds[:,:,1], [-1])
  boxes_flat = tf.reshape(boxes_preds, [-1, 4])

  # TODO: also collect (y,x) coordinates
  idxs = tf.where(mask_flat > seg_threshold)[:,0]
  boxes = tf.gather(boxes_flat, idxs)
  boxes, confidence = refine_boxes(boxes, num_iters, step, sigma)

  num_boxes = tf.shape(boxes)[0]

  dists = tf.nn.relu(nnutil.pairwise_distance(boxes / sigma))
  weights = tf.exp(-dists)

  def _next_detection(prev, i):
    _, _, presence = prev
    confidence_curr = tf.reduce_sum(weights * presence, [1], True)
    idx = tf.to_int32(tf.argmax(confidence_curr, 0)[0])
    mask = tf.to_float(tf.gather(dists, idx) > dist_threshold)[:,tf.newaxis]
    presence = presence * mask
    confidence = tf.gather(confidence_curr, idx)[0]
    return idx, confidence, presence

  idxs, confidence, presences = tf.scan(_next_detection,
                                         tf.range(0, num_outputs),
                                         initializer=(0,
                                                      0.0,
                                                      tf.ones([num_boxes,1])))
  return tf.gather(boxes, idxs), confidence


def compute_detections_nms(seg_preds, boxes_preds, num_keep,
                           seg_threshold=0.2,
                           iou_threshold=0.6):

  mask_flat = tf.reshape(seg_preds[:,:,1], [-1])
  boxes_flat = tf.reshape(boxes_preds, [-1, 4])

  # TODO: also collect (y,x) coordinates
  idxs = tf.where(mask_flat > seg_threshold)[:,0]
  boxes = tf.gather(boxes_flat, idxs)
  confidence = tf.gather(mask_flat, idxs)
  # TODO: here we want to maybe run barinova-style nms
  idxs = tf.image.non_max_suppression(boxes,
                                      confidence,
                                      max_output_size=num_keep,
                                      iou_threshold=iou_threshold)
  num_dets = tf.shape(idxs)[0]
  # TODO: this doesn't work with high seg threshold
  idxs = tf.concat(0, [idxs, idxs[0:num_keep-num_dets]])
  return tf.gather(boxes, idxs), tf.gather(confidence, idxs)

def compute_detections_batch(segs, boxes, num_keep,
                             seg_threshold=0.2,
                             sigma=5e-3, step=0.2, num_iters=20,
                             dist_threshold=20.0,
                             iou_threshold=0.5,
                             nms_kind='greedy'):

  if nms_kind == 'greedy':
    # TODO: rename it to CRF?
    _compute_frame = (lambda x: compute_detections_greedy(x[0], x[1], num_keep,
                                                          seg_threshold,
                                                          sigma, step, num_iters,
                                                          dist_threshold))
  elif nms_kind == 'nms':
    _compute_frame = (lambda x: compute_detections_nms(x[0], x[1], num_keep,
                                                       seg_threshold,
                                                       iou_threshold))
  boxes, confidence = tf.map_fn(_compute_frame, (segs, boxes))
  return boxes, confidence


def det_net(features, num_resnet_blocks, num_resnet_features,
            num_keep, in_size,
            nms_kind='greedy',
            scope=None):

  with tf.variable_scope(scope, 'DetNet'):
    out_size = features.get_shape()[1:3]

    x = nnutil.stack(features,
                     num_resnet_blocks,
                     num_resnet_features,
                     downsample=False)

    with tf.variable_scope('seg'):
      seg_logits = slim.conv2d(x, 2, [1, 1],
                               activation_fn=None,
                               weights_initializer=tf.random_normal_initializer(stddev=1e-1),
                               scope='logits')
      seg_preds = slim.softmax(seg_logits)

    with tf.variable_scope('reg'):
      # TODO: use reg masks instead
      reg_preds = slim.conv2d(x, 4, [1, 1],
                              weights_initializer=tf.random_normal_initializer(stddev=1e-3),
                              activation_fn=tf.nn.relu,
                              scope='reg_preds')

    with tf.variable_scope('boxes'):
      boxes_proposals = reg_to_boxes(reg_preds, in_size, out_size)
      boxes_preds = compute_detections_batch(seg_preds, boxes_proposals,
                                             num_keep, nms_kind=nms_kind)

  return seg_preds, reg_preds, boxes_proposals, boxes_preds


def det_net_loss(seg_masks_in, reg_masks_in,
                 seg_preds, reg_preds,
                 reg_loss_weight=10.0,
                 epsilon=1e-5):

  with tf.variable_scope('loss'):
    out_size = seg_preds.get_shape()[1:3]
    seg_masks_in_ds = tf.image.resize_images(seg_masks_in[:,:,:,tf.newaxis],
                                             out_size[0], out_size[1],
                                             tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    reg_masks_in_ds = tf.image.resize_images(reg_masks_in,
                                             out_size[0], out_size[1],
                                             tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # segmentation loss
    seg_masks_onehot = slim.one_hot_encoding(seg_masks_in_ds[:,:,:,0], 2)
    seg_loss = - tf.reduce_mean(seg_masks_onehot * tf.log(seg_preds + epsilon))

    # regression loss
    mask = tf.to_float(seg_masks_in_ds)
    reg_loss = tf.reduce_sum(mask * (reg_preds - reg_masks_in_ds)**2)
    reg_loss = reg_loss / (tf.reduce_sum(mask) + 1.0)

  return seg_loss + reg_loss_weight * reg_loss
