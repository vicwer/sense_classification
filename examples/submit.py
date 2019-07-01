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

def submit(img_path, submit_file, epoch):
    submit_f = open(submit_file ,'w')
    is_training = False
    cfg.batch_size = 1
    ckpt_dir = cfg.ckpt_path

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

        for idx in tqdm(range(1, 89234)):
            img = cv2.imread(os.path.join(img_path, str(idx).zfill(5) + '.jpg'))
            image = cv2.resize(img, (224, 224))
            img_data = image.astype(np.float32) / 255.0 * 2.0

            classes_index, scores_0 = sess.run([classes, scores], feed_dict={imgs_holder: np.reshape(img_data, [1, 224, 224, 3])})
            submit_f.write(str(idx).zfill(5) + '.jpg' + ' ' + str(classes_index[0] + 1) + '\n')

    submit_f.close()
    zf = zipfile.ZipFile('./submit/classification.zip', 'w', zipfile.zlib.DEFLATED)
    zf.write(submit_file)
    zf.close()

    tf.reset_default_graph()

if __name__ == '__main__':
    img_path = '../data/rssrai_sense_cls/test'
    if not os.path.exists('./submit'):
        os.makedirs('./submit')
    submit_file = './submit/classification.txt'

    epoch = [80]
    print(epoch)
    for i in epoch:
        submit(img_path, submit_file, i)
