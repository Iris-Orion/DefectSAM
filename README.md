# DefectSAM

[English](#english) | [中文](#中文)

---

## English

### Project structure

`baseline` folder: contains baseline model implementations.

```
|-- baseline
|   |-- magnetic_tile_baseline.py
|   |-- modelLib.py
|   |-- neu_seg_baseline.py
|   |-- retina_baseline.py
|   |-- sd_900_baseline.py
|   `-- severstal_baseline.py

|-- data
|   |-- archieve_data_utils.py
|   |-- data_utils_baseline.py
|   |-- data_utils_ft.py
|   |-- severstal.py

|-- train
|   |-- bsds500_ft.py
|   |-- floodSeg_finetune.py
|   |-- magnetic_tile_finetune.py
|   |-- neu_finetune.py
|   |-- retina_finetune.py
|   |-- sd_900_finetune.py
|   `-- severstal_finetune.py

|-- utils
|   |-- baseline_engine.py
|   |-- config.py
|   |-- engine.py
|   |-- finetune_engine.py
|   |-- helper_function.py
|   |-- loratask.py
|   |-- sam_arch.py
|   `-- utils.py
```

### How to train

For the Flood dataset, use:

```
python -m train.floodSeg_finetune --auto_seg
```

### How to use pretrained weights

For the NEU-SEG dataset, use:

```
python -m train.neu_finetune --infer_mode
```

```
python -m baseline.neu_seg_baseline --infer_mode --batch_size 32
```

### Other useful scripts for understanding how the dataset was sliced

```
python -m data.data_utils_baseline
```

Sample output (kept as originally printed, in Chinese):

```
正在为分层抽样生成策略键...
策略键生成完毕。
原始训练集中的类别组合分布:
Counter({'1': 992, '3': 972, '2': 971, '1_2': 238, '2_3': 227, '1_3': 218, '1_2_3': 12})

划分后训练集的分布:
Counter({'1': 744, '3': 729, '2': 728, '1_2': 178, '2_3': 170, '1_3': 164, '1_2_3': 9})

划分后验证集的分布:
Counter({'1': 248, '2': 243, '3': 243, '1_2': 60, '2_3': 57, '1_3': 54, '1_2_3': 3})
baseline 测试
训练集大小: 2722, 验证集大小: 908, 测试集大小: 840
```

### Dataset splits

**Retina**: train 60 / val 20 / test 20

---

## 中文

### 项目结构

baseline 文件夹：存放各类 baseline 模型实现。

```
|-- baseline
|   |-- magnetic_tile_baseline.py
|   |-- modelLib.py
|   |-- neu_seg_baseline.py
|   |-- retina_baseline.py
|   |-- sd_900_baseline.py
|   `-- severstal_baseline.py

|-- data
|   |-- archieve_data_utils.py
|   |-- data_utils_baseline.py
|   |-- data_utils_ft.py
|   |-- severstal.py

|-- train
|   |-- bsds500_ft.py
|   |-- floodSeg_finetune.py
|   |-- magnetic_tile_finetune.py
|   |-- neu_finetune.py
|   |-- retina_finetune.py
|   |-- sd_900_finetune.py
|   `-- severstal_finetune.py

|-- utils
|   |-- baseline_engine.py
|   |-- config.py
|   |-- engine.py
|   |-- finetune_engine.py
|   |-- helper_function.py
|   |-- loratask.py
|   |-- sam_arch.py
|   `-- utils.py
```

### 如何训练

对于 Flood 数据集，使用：

```
python -m train.floodSeg_finetune --auto_seg
```

### 如何使用预训练权重

对于 NEU-SEG 数据集，使用：

```
python -m train.neu_finetune --infer_mode
```

```
python -m baseline.neu_seg_baseline --infer_mode --batch_size 32
```

### 一些有助于理解数据集切分方式的脚本

```
python -m data.data_utils_baseline
```

输出示例：

```
正在为分层抽样生成策略键...
策略键生成完毕。
原始训练集中的类别组合分布:
Counter({'1': 992, '3': 972, '2': 971, '1_2': 238, '2_3': 227, '1_3': 218, '1_2_3': 12})

划分后训练集的分布:
Counter({'1': 744, '3': 729, '2': 728, '1_2': 178, '2_3': 170, '1_3': 164, '1_2_3': 9})

划分后验证集的分布:
Counter({'1': 248, '2': 243, '3': 243, '1_2': 60, '2_3': 57, '1_3': 54, '1_2_3': 3})
baseline 测试
训练集大小: 2722, 验证集大小: 908, 测试集大小: 840
```

### 数据集分布

Retina：训练集 60，验证集 20，测试集 20
