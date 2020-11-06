
### Introduction
BERT fine-tune 多标签分类任务

###  requirements
python3
pytorch
tqdm
sklearn
numpy

###  Train & Test

* 预处理数据 - 在utils/config/preprocess_data_bert.config中设置好文件路径等
```
python3 preprocess_data.py
```
* 训练 - 在utils/config/train_bert.config中设置好文件路径等
```
python3 train_bert.py --mode train
```
```
nohup python3 -u train_bert.py  > myout.file 2>&1 &
tail -f myout.file
cat myout.file | grep 'loss='
```
* 测试
```
python3 train_bert.py --mode test
```

###  Use in your dataset

因为数据目前无法公开，因此需要你重写： preprocess_data.py



