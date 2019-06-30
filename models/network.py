#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops
import sys
sys.path.append('..')
import numpy as np
from config import cfg
from models import resnet_utils
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope

PRINT_LAYER_LOG = cfg.PRINT_LAYER_LOG

def network_arg_scope(is_training=True,
                      weight_decay=cfg.train.weight_decay,
                      batch_norm_decay=0.997,
                      batch_norm_epsilon=1e-5,
                      batch_norm_scale=True):
    batch_norm_params = {
        'is_training': is_training, 'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon, 'scale': batch_norm_scale,
        'updates_collections': ops.GraphKeys.UPDATE_OPS,
        #'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
        'trainable': cfg.train.bn_training,
    }

    with slim.arg_scope(
            [slim.conv2d, slim.separable_convolution2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            trainable=is_training,
            activation_fn=tf.nn.relu6,
            #activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params,
            padding='SAME'):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc

class Network(object):
    def __init__(self):
        pass

    def inference(self, mode, inputs, scope='SenseCls'):
        is_training = mode
        with slim.arg_scope(network_arg_scope(is_training=is_training)):
            with tf.variable_scope(scope, reuse=False):
                conv0 = slim.conv2d(inputs,
                                    num_outputs=64,
                                    kernel_size=[7,7],
                                    stride=2,
                                    scope='conv0')
                if PRINT_LAYER_LOG:
                    print(conv0.name, conv0.get_shape())
                pool0 = slim.max_pool2d(conv0, kernel_size=[3, 3], stride=2, scope='pool0')
                if PRINT_LAYER_LOG:
                    print('pool0', pool0.get_shape())

                block0_0 = block(pool0, 64, 1, 'block0_0')
                block0_1 = block(block0_0, 64, 1, 'block0_1')
                block0_2 = block(block0_1, 64, 1, 'block0_2')

                block1_0 = block(block0_2, 128, 2, 'block1_0')
                block1_1 = block(block1_0, 128, 1, 'block1_1')
                block1_2 = block(block1_1, 128, 1, 'block1_2')
                block1_3 = block(block1_2, 128, 1, 'block1_3')

                block2_0 = block(block1_3, 256, 2, 'block2_0')
                block2_1 = block(block2_0, 256, 1, 'block2_1')
                block2_2 = block(block2_1, 256, 1, 'block2_2')
                block2_3 = block(block2_2, 256, 1, 'block2_3')
                block2_4 = block(block2_3, 256, 1, 'block2_4')
                block2_5 = block(block2_4, 256, 1, 'block2_5')

                block3_0 = block(block2_5, 512, 2, 'block3_0')
                block3_1 = block(block3_0, 512, 1, 'block3_1')
                block3_2 = block(block3_1, 512, 1, 'block3_2')

                net = tf.reduce_mean(block3_2, [1, 2], keepdims=True, name='global_pool_v4')
                if PRINT_LAYER_LOG:
                    print('avg_pool', net.get_shape())
                net = slim.flatten(net, scope='PreLogitsFlatten')
                net = slim.dropout(net, 0.8, is_training=is_training, scope='dropout')
                logits = fully_connected(net, cfg.classes, name='fc')
                if PRINT_LAYER_LOG:
                    print('logits', logits.get_shape())
                if is_training:
                    l2_loss = tf.add_n(tf.losses.get_regularization_losses())
                    return logits, l2_loss
                else:
                    return logits

def block(inputs, c_outputs, s, name):
    se_module = True
    out1 = slim.conv2d(inputs,
                       num_outputs=c_outputs,
                       kernel_size=[3,3],
                       stride=s,
                       scope=name+'_0')
    if PRINT_LAYER_LOG:
        print(name+'_0', out1.get_shape())
    output = slim.conv2d(out1,
                       num_outputs=c_outputs,
                       kernel_size=[3,3],
                       stride=1,
                       activation_fn=None,
                       scope=name+'_1')
    if PRINT_LAYER_LOG:
        print(name+'_1', output.get_shape())
    if s == 2:
        return nn_ops.relu(output)
    else:
        if se_module:
            squeeze = tf.reduce_mean(output, [1, 2], keepdims=True, name='global_pool_v4')
            if PRINT_LAYER_LOG:
                print('squeeze', squeeze.get_shape())
            fc1 = slim.conv2d(squeeze,
                            num_outputs=squeeze.get_shape()[-1] // 16,
                            normalizer_fn=None,
                            normalizer_params=None,
                            weights_regularizer=None,
                            kernel_size=[1,1],
                            stride=1,
                            activation_fn=tf.nn.relu,
                            scope=name+'_fc1')
            if PRINT_LAYER_LOG:
                print('fc1', fc1.get_shape())
            fc2 = slim.conv2d(fc1,
                            num_outputs=squeeze.get_shape()[-1],
                            normalizer_fn=None,
                            normalizer_params=None,
                            weights_regularizer=None,
                            kernel_size=[1,1],
                            stride=1,
                            activation_fn=tf.sigmoid,
                            scope=name+'_fc2')
            if PRINT_LAYER_LOG:
                print('fc2', fc2.get_shape())
            output = output * fc2
        output = nn_ops.relu(inputs + output)
        if PRINT_LAYER_LOG:
            print(name, output.get_shape())
        return output

def dense_block(inputs, depth, depth_bottleneck, stride, name, rate=1):
    depth_in = inputs.get_shape()[3]
    if depth == depth_in:
        if stride == 1:
            shortcut = inputs
        else:
            shortcut = layers.max_pool2d(inputs, [1, 1], stride=factor, scope=name+'_shortcut')
    else:
        shortcut = layers.conv2d(
            inputs,
            depth, [1, 1],
            stride=stride,
            activation_fn=None,
            scope=name+'_shortcut')
    if PRINT_LAYER_LOG:
        print(name+'_shortcut', shortcut.get_shape())

    residual = layers.conv2d(
        inputs, depth_bottleneck, [1, 1], stride=1, scope=name+'_conv1')
    if PRINT_LAYER_LOG:
        print(name+'_conv1', residual.get_shape())
    residual = resnet_utils.conv2d_same(
        residual, depth_bottleneck, 3, stride, rate=rate, scope=name+'_conv2')
    if PRINT_LAYER_LOG:
        print(name+'_conv2', residual.get_shape())
    residual = layers.conv2d(
        residual, depth, [1, 1], stride=1, activation_fn=None, scope=name+'_conv3')
    if PRINT_LAYER_LOG:
        print(name+'_conv3', residual.get_shape())
    output = nn_ops.relu(shortcut + residual)
    return output

def conv2d(inputs, c_outputs, s, name):
    output = slim.conv2d(inputs, num_outputs=c_outputs, kernel_size=[3,3], stride=s, scope=name)
    if PRINT_LAYER_LOG:
        print(name, output.get_shape())
    return output

def maxpool2x2(input, name):
    output = slim.max_pool2d(input, kernel_size=[2, 2], stride=2, scope=name)
    if PRINT_LAYER_LOG:
        print(name, output.get_shape())
    return output

def fully_connected(input, c_outputs, name):
    output = slim.fully_connected(input, c_outputs, activation_fn=None, scope=name)
    if PRINT_LAYER_LOG:
        print(name, output.get_shape())
    return output

def d_p_conv(inputs, c_outputs, s, name):
    output = slim.separable_convolution2d(inputs,
                                          num_outputs=None,
                                          stride=s,
                                          depth_multiplier=1,
                                          kernel_size=[3, 3],
                                          normalizer_fn=slim.batch_norm,
                                          scope=name+'_d_conv')
    if PRINT_LAYER_LOG:
        print(name, output.get_shape())

    output = slim.conv2d(output,
                         num_outputs=c_outputs,
                         kernel_size=[1,1],
                         stride=1,
                         scope=name+'_p_conv')
    if PRINT_LAYER_LOG:
        print(name, output.get_shape())
    return output

def route(input_list, name):
    with tf.name_scope(name):
        output = tf.concat(input_list, 3, name='concat')
    if PRINT_LAYER_LOG:
        print(name, output.get_shape())
    return output
