# transfer_learning

利用pytorch迁移学习实现图像分类，其中包含的densenet，resnext，mobilenet，efficientnet, resnet等图像分类网络，可以根据需要再行利用torchvision扩展其他的分类算法

## 实现功能
* 基础功能利用pytorch迁移学习实现图像分类
* 包含带有warmup的cosine学习率调整
* warmup的step学习率优调整
* 多模型融合预测，加权与投票融合
* 利用flask实现模型云端api部署
* 使用tta测试时增强进行预测
* 添加label smooth的pytorch实现（标签平滑）
* 更新添加了模型蒸馏的的训练方法
* 添加中间层可视化

  

## 运行环境
* python3.7
* pytorch 1.1
* torchvision 0.3.0

## 代码仓库的使用

### 数据集形式
原始数据集存储形式为，同个类别的图像存储在同一个文件夹下，所有类别的图像存储在一个主文件夹data下。

```
|-- data
    |-- train
        |--label1
            |--*.jpg
        |--label2
            |--*.jpg
        |--label    
            |--*.jpg
        ...

    |-- val
        |--*.jpg
```

利用preprocess.py将数据集格式进行转换（个人习惯这种数据集的方式）

```
python ./data/preprocess.py
```

转换后的数据集为，将训练集的路径与类别存储在train.txt文件中，测试机存储在val.txt中.
其中txt文件中的内容为

```
# train.txt

/home/xxx/data/train/label1/*.jpg   label

# val.txt

/home/xxx/data/train/label1/*.jpg
```

```
|-- data
    |-- train
        |--label1
            |--*.jpg
        |--label2
            |--*.jpg
        |--label    
            |--*.jpg
        ...

    |-- val
        |--*.jpg
    |--train.txt
    |--val.txt
```

### 处理数据集


```shell
python dataset.py
```

### 训练


```shell
python train.py
```

### 预测
在cfg.py中`TRAINED_MODEL`参数修改为指定的权重文件存储位置,在predict文件中可以选定是否使用tta

```shell
python predict.py
```


# 
