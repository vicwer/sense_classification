#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import numpy as np
import sys
sys.path.append('..')
from config import cfg
import os
import re
import cv2

def distorted_bounding_box_crop(image,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.9, 1.0),
                                max_attempts=100,
                                scope=None):

    with tf.name_scope(scope, 'distort_image', [image, image.shape[0], image.shape[1], bbox]):
        if bbox is None:
            bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                                dtype=tf.float32,
                                shape=[1, 1, 4])
    if image.dtype != tf.float32:
        img = tf.image.convert_image_dtype(image, dtype=tf.float32)

    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=bbox,
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

        # Crop the image to the specified bounding box.
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        return cropped_image, distort_bbox

def parser(example):
    feats = tf.parse_single_example(example, features={'label' : tf.FixedLenFeature([1], tf.float32),
                                                       'feature': tf.FixedLenFeature([], tf.string)})
    coord = feats['label']

    img = tf.decode_raw(feats['feature'], tf.uint8)
    img = tf.reshape(img, [256, 256, 3])

    rot_img = tf.image.rot90(img)
    rot_seed = tf.random_uniform([], maxval=1.0)
    img = tf.cond(rot_seed > 0.5, lambda: img, lambda: rot_img)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)

    crop_img = tf.random_crop(img, [224, 224, 3])
    img = tf.image.resize_images(img, [224, 224])
    img = tf.cast(img, tf.uint8)
    # img, _ = distorted_bounding_box_crop(img, bbox=None)
    # crop_img = tf.image.resize_images(crop_img, [224, 224])

    rand_seed = tf.random_uniform([], maxval=1.0)
    img = tf.cond(rand_seed > 0.5, lambda: img, lambda: crop_img)

    img = tf.cast(img, tf.float32) / 255.0
    img = tf.image.random_hue(img, max_delta=0.1)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    img = tf.image.random_brightness(img, max_delta=0.05)
    img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
    img = tf.clip_by_value(img, 0.0, 1.0)
    img = tf.multiply(img, 2.0)
    return img, coord

def gen_data_batch(tf_records_filename, batch_size):
    dt = tf.data.TFRecordDataset(tf_records_filename)
    dt = dt.map(parser, num_parallel_calls=4)
    dt = dt.prefetch(batch_size)
    dt = dt.shuffle(buffer_size=8*batch_size)
    dt = dt.repeat()
    dt = dt.batch(batch_size)
    iterator = dt.make_one_shot_iterator()
    imgs, true_boxes = iterator.get_next()

    return imgs, true_boxes

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    tf_records_filename = cfg.data_path

    imgs, true_boxes = gen_data_batch(tf_records_filename, cfg.batch_size)
    imgs_split = tf.split(imgs, cfg.train.num_gpus)
    true_boxes_split = tf.split(true_boxes, cfg.train.num_gpus)
    configer = tf.ConfigProto()
    configer.gpu_options.per_process_gpu_memory_fraction = 0.3
    sess=tf.Session(config=configer)
    for i in range(2):
        for j in range(cfg.train.num_gpus):
            imgs_, true_boxes_ = sess.run([imgs_split[j], true_boxes_split[j]])
            print(true_boxes_.shape)

            for k in range(imgs_.shape[0]):
                cv2.imshow('img', imgs_[k].astype(np.uint8))
                cv2.waitKey(0)

