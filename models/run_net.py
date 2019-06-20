#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import sys
sys.path.append('..')
from models.network import Network
from config import cfg
from models.losses import loss, loss_ohem

class SenseClsNet:
    def __init__(self, img, truth, is_training, batcn_norm_decay=0.997):
        self.img = img
        self.truth = truth
        self.is_training = is_training
        self.batch_norm_decay = batcn_norm_decay
        self.img_shape = tf.shape(self.img)
        backbone = Network()
        if is_training:
            self.head, self.l2_loss = backbone.inference(self.is_training, self.img)
        else:
            self.head = backbone.inference(self.is_training, self.img)

    def compute_loss(self):
        with tf.name_scope('loss_0'):
            cls_loss = loss(self.head, self.truth)
            self.all_loss = cls_loss + self.l2_loss
        return self.all_loss

    def predict(self):
        '''
        only support single image prediction
        '''
        pred_score = tf.reshape(self.head, (-1, cfg.classes))
        score = tf.nn.softmax(tf.reshape(self.head, (-1, cfg.classes)))
        class_index = tf.argmax(pred_score, 1)
        return class_index, score
