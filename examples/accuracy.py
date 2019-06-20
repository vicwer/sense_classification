#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import tensorflow as tf
import sys
sys.path.append('..')
from models.run_net import SenseClsNet
from config import cfg
import cv2
import os
from tqdm import tqdm
import zipfile
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def accuracy(img_path, label_file, epoch):
    forders = os.listdir(img_path)
    labels_f = open(label_file, 'r')
    labels = labels_f.readlines()
    label_dict = dict()
    for l in labels:
        key_value = l.strip().split(':')[0::2]
        label_dict.update({key_value[0] : key_value[1]})

    is_training = False
    cfg.batch_size = 1
    ckpt_dir = cfg.ckpt_path

    correct = 0
    wrong = 0
    all_image = 0

    configer = tf.ConfigProto()
    configer.gpu_options.per_process_gpu_memory_fraction = 0.1
    with tf.Session(config=configer) as sess:
        imgs_holder = tf.placeholder(tf.float32, shape=[1, 224, 224, 3])
        model = SenseClsNet(imgs_holder, None, is_training)
        classes, scores = model.predict()

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        saver.restore(sess, ckpt_dir + 'senceCls-' + str(epoch))
        sess.run(tf.local_variables_initializer())

        for f in tqdm(forders):
            label = float(label_dict[f])
            imgs = os.listdir(os.path.join(img_path, f))
            for img in imgs:
                path = os.path.join(os.path.join(img_path, f), img)
                img = cv2.imread(path)

                image = cv2.resize(img, (224, 224))
                img_data = image.astype(np.float32) / 255.0 * 2.0

                all_image += 1
                classes_index, scores_0 = sess.run([classes, scores], feed_dict={imgs_holder: np.reshape(img_data, [1, 224, 224, 3])})
                print(str(classes_index[0]), label)
                if classes_index[0] + 1 == label:
                    correct += 1
                else:
                    wrong += 1

        accuracy = float(correct) / float(correct + wrong)
        print('global_step: ', g_step)
        print("All images:\n {}".format(int(correct + wrong)))
        print("Accuracy: {:.4f}".format(accuracy))

    tf.reset_default_graph()

if __name__ == '__main__':
    img_path = '../data/rssrai_sense_cls/train'
    label_file = '../data/rssrai_sense_cls/ClsName2id.txt'

    epoch = np.arange(12, 1, -1)
    print(epoch)
    for i in epoch:
        accuracy(img_path, label_file, i)
