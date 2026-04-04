# Baseline 文件夹说明

本文件夹包含传统语义分割框架的训练与推理脚本，用于与 SAM 微调方法进行性能对比。

## 文件概览

### modelLib.py — 模型库

集中定义所有 baseline 分割模型的架构：

| 工厂函数 | 模型 | Backbone |
|---------|------|----------|
| `get_vanilla_unet()` | 原生 U-Net | 无预训练 |
| `get_smp_unet()` | U-Net (smp) | ResNet34, ImageNet 预训练 |
| `get_smp_deeplabv3plus()` | DeepLabV3+ (smp) | EfficientNet-B0/B3 |
| `build_segformer_model()` | SegFormer | MIT-B0 ~ B5 |

另外包含 `get_transforms()` 数据增强管线和 `get_model(model_name)` 统一工厂函数。

---

### 各数据集训练脚本

所有训练脚本共享相同的训练流程：

- **损失函数**: `monai.DiceCELoss(sigmoid=True, squared_pred=True)`
- **优化器**: AdamW (weight_decay=0.01)
- **调度器**: Cosine Annealing + 10% Linear Warmup
- **训练入口**: `baseline_experiment()` (来自 `utils/baseline_engine.py`)
- **推理入口**: `bsl_inference_engine()`

| 文件 | 数据集 | 输出目录 | 特殊说明 |
|------|--------|---------|---------|
| `neu_seg_baseline.py` | NEU-SEG 表面缺陷 | `new_weights/neu_seg_output/` | 逐 batch 更新 LR |
| `sd900_baseline.py` | SD900 表面缺陷 | `new_weights/baseline/sd900/` | 逐 batch 更新 LR；记录标签分布 |
| `severstal_baseline.py` | Severstal 钢材缺陷 | `new_weights/` | 逐 batch 更新 LR；支持 `--mini_dataset`；可选包含无缺陷样本 |
| `magnetic_tile_baseline.py` | 磁砖缺陷 | `new_weights/magnetic_tile_output/` | 逐 batch 更新 LR |
| `retina_baseline.py` | 视网膜血管 | `new_weights/baseline/retina/` | 逐 batch 更新 LR；固定推理权重路径 |

### 通用命令行参数

```
--batch_size       批大小 (默认 16)
--learning_rate    学习率 (默认 1e-4)
--num_epochs       训练轮数 (默认 100)
--weight_decay     权重衰减 (默认 0.01)
--patience         早停耐心值 (默认 10)
--device_id        GPU ID (默认 0)
--bse_model        模型类型: unet_res34 / smp_deeplabv3plus / segformer_b2
--infer_mode       推理模式 (加载权重评估)
--save_bse_model   保存最优模型
```

### run_baseline.sh 快速入口

项目根目录提供了统一的 baseline 启动脚本：

```bash
./run_baseline.sh <dataset> <model> [extra_args...]
```

当前支持的数据集缩写：

```bash
neu | sd900 | mag | sev | retina
```

常用模型：

```bash
unet_res34 | deeplabv3plus_effb3 | segformer_b2
```

脚本会自动补齐以下参数：

```bash
--batch_size
--learning_rate
--num_epochs
--device_id
--swanlab
--pj_name
```

对应入口模块映射：

| 缩写 | baseline 入口 |
|------|---------------|
| `neu` | `baseline.neu_seg_baseline` |
| `sd900` | `baseline.sd900_baseline` |
| `mag` | `baseline.magnetic_tile_baseline` |
| `sev` | `baseline.severstal_baseline` |
| `retina` | `baseline.retina_baseline` |

示例：

```bash
./run_baseline.sh neu unet_res34
./run_baseline.sh sd900 deeplabv3plus_effb3
./run_baseline.sh mag segformer_b2
./run_baseline.sh sev unet_res34 --include_no_defect
./run_baseline.sh neu unet_res34 --save
./run_baseline.sh sd900 deeplabv3plus_effb3 --save
./run_baseline.sh sev segformer_b2 --include_no_defect
./run_baseline.sh mag unet_res34 --device_id 0 --num_epochs 50
```

### 运行示例

```bash
# 直接调用单个训练入口
python -m baseline.neu_seg_baseline --bse_model unet_res34 --num_epochs 100 --device_id 0

# 推理
python -m baseline.neu_seg_baseline --bse_model unet_res34 --infer_mode
```
