#-*- coding:utf-8 -*-

import sys
import os
import numpy as np
import numpy.random as npr
sys.path.append('..')
from prepare_data.gen_tf_records import run_encode
import cv2
from tqdm import tqdm

def gen_img_list(img_path, label_file, img_list_path):
    forders = os.listdir(img_path)
    print(forders)
    labels_f = open(label_file, 'r')
    labels = labels_f.readlines()
    img_f = open(os.path.join(img_list_path, 'train.txt'), 'w')

    label_dict = dict()
    for l in labels:
        key_value = l.strip().split(':')[0::2]
        label_dict.update({key_value[0] : key_value[1]})
    print(label_dict)

    print('remove dead img')
    cnt_path = os.getcwd()
    for f in tqdm(forders):
        label = label_dict[f]
        imgs = os.listdir(os.path.join(img_path, f))
        for img in imgs:
            path = os.path.join(cnt_path, os.path.join(os.path.join(img_path, f), img))
            img = cv2.imread(path)
            try:
                h, w, _ = img.shape
                img_f.write(path + ' ' + label + '\n')
            except:
                print('dead img: {}'.format(path))
    img_f.close()

def shuffle_list(img_list_path):
    with open(os.path.join(img_list_path, 'train.txt'), 'r') as f:
        lines = f.readlines()

    with open(os.path.join(img_list_path, 'train.txt'), "w") as f:
        num = len(lines)
        lines_keep = npr.choice(len(lines), size=int(num),replace=False)

        for i in lines_keep:
            f.write(lines[i])

def encode(img_path, label_file, img_list_path, records_path):
    for dir_path in [img_list_path, records_path]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    gen_img_list(img_path, label_file, img_list_path)
    for i in range(10):
        shuffle_list(img_list_path)

    records_f = os.path.join(records_path, 'train.records')
    run_encode(os.path.join(img_list_path, 'train.txt'), records_f)    

if __name__ == '__main__':
    img_path = '../data/rssrai_sense_cls/train'
    label_file = '../data/rssrai_sense_cls/ClsName2id.txt'
    img_list_path = '../data/train_list/'
    records_path = '../data/tf_records'
    encode(img_path, label_file, img_list_path, records_path)
