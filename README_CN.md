# 遥感图像场景分类

[English Version](README.md)

## 简介
[RSCUP: 遥感图像场景分类](http://rscup.bjxintong.com.cn/#/theme/1)



仓库路径应该组织成如下结构:

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

## 环境依赖
1. tensorflow-gpu==1.2.0 (I only test on tensorflow 1.12.0)
2. python==3.4.3
3. numpy
4. easydict
5. opencv==3.4.1
6. 有些包可能没列出来,根据错误提示安装

## 安装, 准备数据, 训练, 验证, 生成提交文件
### 安装
1. 下载代码

```
git clone https://github.com/vicwer/sense_classification.git
```

### 准备数据
data目录结构:

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
1. 下载数据集并解压: train.zip, val.zip, test.zip, ClsName2id.txt

2. 生成 tf_records:

```
cd tools
python3 img_encode.py
```

### 训练

${sense_classification_ROOT}目录提供了config.py, 可设置超参数

例如
```
cd ${sense_classification_ROOT}
vim config.py
cfg.train.num_gpus = {your gpu nums}
etc.

cd ${sense_classification_ROOT}/examples/
python3 multi_gpus_train.py
```

### 验证

```
cd ${sense_classification_ROOT}/examples/
python3 accuracy.py
```

### 生成提交文件

```
cd ${sense_classification_ROOT}/examples/
python3 submit.py
```

## 结果:

验证集: 0.908+
