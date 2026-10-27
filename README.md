# DefectSAM

[English](#english) | [中文](#中文)

---

## English

Code for **LoRA-DSC**, a parameter-efficient fine-tuning (PEFT) method that adapts the Segment Anything Model (SAM, ViT-B) to metal surface defect segmentation, together with a benchmark of full fine-tuning, decoder-only tuning, common PEFT methods and classic segmentation networks on four industrial datasets.

LoRA-DSC inserts a depthwise separable convolution into the low-rank branch of LoRA on the Q and V projections of the SAM image encoder, with a residual path and a zero-initialized gate. It trains 0.599 M parameters (0.64% of SAM ViT-B).

### Repository layout

```
baseline/      classic segmentation networks (U-Net, DeepLabV3+, SegFormer), one script per dataset
data/          dataset loading, letterbox preprocessing, stratified 6:2:2 split
train/         SAM fine-tuning entry points, one script per dataset
utils/         training engines, LoRA-DSC and other adapter definitions, CLI arguments
eval/          per-image Dice / IoU evaluation over all seeds
benchmark/     result CSVs, per-dataset reports and analysis / plotting / profiling scripts
tool/          figure rendering and helper scripts
zeroshot_sam2_sam3/  zero-shot evaluation of SAM 2 / SAM 3
```

Key files:

| File | Content |
| --- | --- |
| `utils/sam_arch.py` | LoRA-DSC and other custom LoRA variants injected into the SAM vision encoder |
| `utils/loratask.py` | `create_model_from_type`: builds the model for each `--ft_type` |
| `utils/finetune_engine.py` | SAM fine-tuning loop (bf16, `torch.compile` on the vision encoder, early stopping) |
| `utils/baseline_engine.py` | training loop for the classic networks |
| `utils/config.py` | all command-line arguments |
| `data/common_ops.py` | letterbox resize and stratified split |

### Environment

Tested with Python 3.11, PyTorch 2.7.0 (CUDA 12.8) and a single RTX 5090.

```bash
pip install -r requirements.txt
pip install swanlab   # optional, experiment logging
```

SAM ViT-B weights are read from `./HuggingfaceModel/sam_vit_base/model` and downloaded from `facebook/sam-vit-base` if missing. MedSAM (zero-shot comparison only) is read from `./HuggingfaceModel/wanglab/medsam-vit-base/model`.

### Datasets

| Dataset | Directory |
| --- | --- |
| Severstal | `data/severstal_steel_defect_detection/` |
| SD-900 | `data/sd900/` |
| Magnetic-Tile | `data/Magnetic-Tile/` |
| NEU-SEG | `data/NEU_Seg-main/` |

All datasets are split 6:2:2 (train / val / test) with stratification on the defect class combination, seed 42. See `data/README.md` for the expected directory layout.

Images are letterboxed to 1024 × 1024 (padding value 128) and divided by 255. Masks are binarized; a box prompt is derived from each ground-truth mask.

### Training

Run every command from the repository root.

SAM fine-tuning:

```bash
python -m train.sd900_finetune --ft_type loradsc_qv_residual_gated --batch_size 4 \
    --learning_rate 2e-4 --num_epochs 50 --disable_early_stop --lora_rank 16 --lora_alpha 16 \
    --use_loraplus_optim --lora_plus_lr_ratio 8 --save_custom_lora --seed 42 --device_id 0
```

Replace `sd900_finetune` with `severstal_finetune`, `magnetic_tile_finetune` or `neu_finetune` for the other datasets.

| Method | `--ft_type` | Extra flags |
| --- | --- | --- |
| Full fine-tuning | `sam_fully` | `--learning_rate 1e-5` |
| Decoder only | `sam_decoder` | |
| LoRA (Q, V) | `lora_attn_qv` | `--save_custom_lora` |
| LoHa (Q, V) | `loha_qv_encoder` | `--lora_rank 8 --lora_alpha 8 --save_hf_format` |
| LoKr (Q, V) | `lokr_qv_encoder` | `--save_hf_format` |
| AdaLoRA | `adalora_encoder` | `--save_hf_format` |
| LoRA-DSC (ours) | `loradsc_qv_residual_gated` | `--save_custom_lora --use_loraplus_optim` |

Batch sizes used in the paper: 4 for full fine-tuning and all LoRA-style methods, 8 for decoder only, 2 for AdaLoRA. Weights are saved under `./new_weights/`. LoHa, LoKr and AdaLoRA save in Hugging Face PEFT format and reject `--save_custom_lora`. The exact hyperparameters of every run (learning rate, LoRA+ ratio, seeds, epochs) are recorded in `benchmark/<dataset>/<dataset>_bench.csv`; they differ slightly across datasets.

Classic segmentation networks:

```bash
python -m baseline.sd900_baseline --bse_model unet_res34 --batch_size 16 --save
```

`--bse_model` is one of `unet_res34`, `deeplabv3plus_effb3`, `segformer_b2`.

Zero-shot SAM:

```bash
python -m train.sd900_finetune --zero_shot --zero_shot_model sam_base   # or medsam
```

Useful common flags: `--num_epochs`, `--learning_rate`, `--disable_early_stop`, `--no_compile`, `--use_swanlab`.

### Evaluation

Reported numbers are per-image Dice / IoU on the test set, averaged over images within each seed, then reported as mean ± std over 3 seeds. Inference runs in fp32 with threshold sigmoid > 0.5.

```bash
NO_ALBUMENTATIONS_UPDATE=1 python eval/per_image_metrics_all_seeds.py --dataset sd900 --gpus 0,1
```

`--dataset` is one of `sd900`, `neu`, `mag`, `sever`. Results go to `benchmark/per_image/<dataset>/`.

### Benchmark scripts

| Directory | Purpose |
| --- | --- |
| `benchmark/scripts/analyze/` | merge run CSVs and build the per-dataset `report.md` |
| `benchmark/scripts/fetch/` | export runs and curves from SwanLab |
| `benchmark/scripts/plot/` | loss / Dice curves and qualitative figures |
| `benchmark/scripts/profile/` | training memory and step-time measurement |
| `benchmark/scripts/eval/` | forgetting test on a natural-image dataset from the SAM paper |

### Citation

Citation information will be added after publication.

---

## 中文

本仓库是 **LoRA-DSC** 的代码。LoRA-DSC 是一种参数高效微调（PEFT）方法，用于把 Segment Anything Model（SAM，ViT-B）适配到金属表面缺陷分割。仓库同时包含在四个工业数据集上对全量微调、仅微调解码器、常见 PEFT 方法和通用分割网络的对比实验。

LoRA-DSC 在 SAM 图像编码器的 Q、V 投影上，向 LoRA 低秩分支中插入深度可分离卷积，并带有残差通路和零初始化门控。可训练参数为 0.599 M，占 SAM ViT-B 的 0.64%。

### 目录结构

```
baseline/      通用分割网络（U-Net、DeepLabV3+、SegFormer），每个数据集一个脚本
data/          数据集加载、letterbox 预处理、6:2:2 分层划分
train/         SAM 微调入口，每个数据集一个脚本
utils/         训练引擎、LoRA-DSC 及其他适配器定义、命令行参数
eval/          全部种子的逐图 Dice / IoU 评估
benchmark/     结果 CSV、各数据集报告，以及分析、绘图、性能测量脚本
tool/          出图与辅助脚本
zeroshot_sam2_sam3/  SAM 2 / SAM 3 零样本评估
```

关键文件：

| 文件 | 内容 |
| --- | --- |
| `utils/sam_arch.py` | LoRA-DSC 及其他自定义 LoRA 变体，注入 SAM 视觉编码器 |
| `utils/loratask.py` | `create_model_from_type`：按 `--ft_type` 构建模型 |
| `utils/finetune_engine.py` | SAM 微调循环（bf16，对视觉编码器使用 `torch.compile`，早停） |
| `utils/baseline_engine.py` | 通用分割网络的训练循环 |
| `utils/config.py` | 全部命令行参数 |
| `data/common_ops.py` | letterbox 缩放与分层划分 |

### 环境

测试环境为 Python 3.11、PyTorch 2.7.0（CUDA 12.8）、单张 RTX 5090。

```bash
pip install -r requirements.txt
pip install swanlab   # 可选，用于记录实验
```

SAM ViT-B 权重从 `./HuggingfaceModel/sam_vit_base/model` 读取，不存在时自动从 `facebook/sam-vit-base` 下载。MedSAM 仅用于零样本对比，从 `./HuggingfaceModel/wanglab/medsam-vit-base/model` 读取。

### 数据集

| 数据集 | 目录 |
| --- | --- |
| Severstal | `data/severstal_steel_defect_detection/` |
| SD-900 | `data/sd900/` |
| Magnetic-Tile | `data/Magnetic-Tile/` |
| NEU-SEG | `data/NEU_Seg-main/` |

所有数据集按缺陷类别组合分层，以 6:2:2 划分为训练、验证、测试集，随机种子 42。目录格式见 `data/README.md`。

图像 letterbox 到 1024 × 1024（填充值 128）后除以 255。掩码二值化，框提示由真值掩码生成。

### 训练

所有命令都在仓库根目录下运行。

SAM 微调：

```bash
python -m train.sd900_finetune --ft_type loradsc_qv_residual_gated --batch_size 4 \
    --learning_rate 2e-4 --num_epochs 50 --disable_early_stop --lora_rank 16 --lora_alpha 16 \
    --use_loraplus_optim --lora_plus_lr_ratio 8 --save_custom_lora --seed 42 --device_id 0
```

其他数据集把 `sd900_finetune` 换成 `severstal_finetune`、`magnetic_tile_finetune` 或 `neu_finetune`。

| 方法 | `--ft_type` | 附加参数 |
| --- | --- | --- |
| 全量微调 | `sam_fully` | `--learning_rate 1e-5` |
| 仅解码器 | `sam_decoder` | |
| LoRA (Q, V) | `lora_attn_qv` | `--save_custom_lora` |
| LoHa (Q, V) | `loha_qv_encoder` | `--lora_rank 8 --lora_alpha 8 --save_hf_format` |
| LoKr (Q, V) | `lokr_qv_encoder` | `--save_hf_format` |
| AdaLoRA | `adalora_encoder` | `--save_hf_format` |
| LoRA-DSC（本方法） | `loradsc_qv_residual_gated` | `--save_custom_lora --use_loraplus_optim` |

论文中的 batch size：全量微调和各类 LoRA 方法为 4，仅解码器为 8，AdaLoRA 为 2。权重保存在 `./new_weights/` 下。LoHa、LoKr 和 AdaLoRA 以 Hugging Face PEFT 格式保存，不能加 `--save_custom_lora`。每次运行的具体超参数（学习率、LoRA+ 倍率、种子、epoch 数）记录在 `benchmark/<dataset>/<dataset>_bench.csv` 中，不同数据集之间略有差别。

通用分割网络：

```bash
python -m baseline.sd900_baseline --bse_model unet_res34 --batch_size 16 --save
```

`--bse_model` 可选 `unet_res34`、`deeplabv3plus_effb3`、`segformer_b2`。

零样本 SAM：

```bash
python -m train.sd900_finetune --zero_shot --zero_shot_model sam_base   # 或 medsam
```

常用参数：`--num_epochs`、`--learning_rate`、`--disable_early_stop`、`--no_compile`、`--use_swanlab`。

### 评估

论文中的指标为测试集逐图 Dice / IoU：每个种子内先对所有图像取平均，再报告 3 个种子的 mean ± std。推理使用 fp32，阈值为 sigmoid > 0.5。

```bash
NO_ALBUMENTATIONS_UPDATE=1 python eval/per_image_metrics_all_seeds.py --dataset sd900 --gpus 0,1
```

`--dataset` 可选 `sd900`、`neu`、`mag`、`sever`。结果写入 `benchmark/per_image/<dataset>/`。

### Benchmark 脚本

| 目录 | 用途 |
| --- | --- |
| `benchmark/scripts/analyze/` | 合并运行结果 CSV，生成各数据集的 `report.md` |
| `benchmark/scripts/fetch/` | 从 SwanLab 导出运行记录和曲线 |
| `benchmark/scripts/plot/` | loss / Dice 曲线与定性对比图 |
| `benchmark/scripts/profile/` | 训练显存与单步耗时测量 |
| `benchmark/scripts/eval/` | 在 SAM 论文的自然图像数据集上做遗忘测试 |

### 引用

论文发表后补充引用信息。
