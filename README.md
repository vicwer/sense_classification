# sense_classification

[中文版本](README_CN.md)

## Introduction
[RSCUP: 遥感图像场景分类](http://rscup.bjxintong.com.cn/#/theme/1)



This repo is organized as follows:

```
sense_classification/
    |->examples
    |->models
    |->prepare_data
    |->data
    |   |->rssrai_sense_cls
    |   |   |->train
    |   |   |->val
    |   |   |->test
    |   |->tf_records
    |   |->train_list
    |->ckpt
    |->tools
```

## Requirements
1. tensorflow-gpu==1.12.0 (I only test on tensorflow 1.12.0)
2. python==3.4.3
3. numpy
4. easydict
5. opencv==3.4.1
6. Python packages might missing. pls fix it according to the error message.

## Installation, Prepare data, Training, Val, Generate submit
### Installation
1. Clone the sense_classification repository, and we'll call the directory that you cloned sense_classification as `${sense_classification_ROOT}`.

```
git clone https://github.com/vicwer/sense_classification.git
```

### Prepare data
data should be organized as follows:

```
data/
    |->rssrai_sense_cls
    |   |->train
    |   |->val
    |   |->test
    |   |->ClsName2id.txt
    |->train_list/train.txt
    |->tf_records
```
1. Download dataset and unzip: train.zip, val.zip, test.zip, ClsName2id.txt

2. Generate tf_records:

```
cd tools
python3 img_encode.py
```

### Training

I provide common used config.py in ${sense_classification_ROOT}, which can set hyperparameters.

e.g.
```
cd ${sense_classification_ROOT}
vim config.py
cfg.train.num_gpus = {your gpu nums}
etc.

cd ${sense_classification_ROOT}/examples/
python3 multi_gpus_train.py
```

### Val

```
cd ${sense_classification_ROOT}/examples/
python3 accuracy.py
```

### Generate submit

```
cd ${sense_classification_ROOT}/examples/
python3 submit.py
```

## Result:

Val: 0.908+
Test: 0.90509
