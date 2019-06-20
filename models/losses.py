#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import sys
sys.path.append('..')
from config import cfg

PRINT_LAYER_LOG = cfg.PRINT_LAYER_LOG

def loss(preds, labels):
    labels = tf.cast(labels, tf.int64)
    if PRINT_LAYER_LOG:
        print('pre labels', labels.get_shape())
    labels = tf.reshape(labels, (cfg.batch_size, -1))
    if PRINT_LAYER_LOG:
        print('labels', labels.get_shape())
    labels = tf.one_hot(labels, cfg.classes)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=preds, labels=labels)
    pred_loss = tf.reduce_mean(cross_entropy)
    return pred_loss

def loss_ohem(preds, labels):
    labels = tf.cast(labels, tf.int64)
    labels = tf.reshape(labels, (cfg.batch_size,))
    print('pre labels', labels.get_shape())
    labels = tf.one_hot(labels, cfg.classes)
    print('labels', labels.get_shape())
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=preds, labels=labels)
    print('cross_entropy', cross_entropy.get_shape())
    keep_num = tf.cast(cfg.batch_size * cfg.train.ohem_ratio, tf.int32)
    cross_entropy = tf.reshape(cross_entropy, (cfg.batch_size,))
    print('cross_entropy', cross_entropy.get_shape())
    _, k_index = tf.nn.top_k(cross_entropy, keep_num)
    loss = tf.gather(cross_entropy, k_index)
    print('ohem loss', loss.get_shape())

    return tf.reduce_mean(loss)
