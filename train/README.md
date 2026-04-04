# Train 文件夹说明

本文件夹包含各数据集上 SAM 微调的训练入口脚本。所有脚本共享 `utils/finetune_engine.py` 中的训练引擎，通过 `--ft_type` 参数选择微调方法。

---

## 通用训练流程

1. 解析参数 → 加载数据集 → 创建 DataLoader
2. `create_model_from_type(args)` 根据 `ft_type` 构建模型
3. `run_finetune_engine()` 执行训练循环（早停 + SwanLab + 最优模型保存）
4. 加载最优权重，在测试集上评估 final_test_dice / final_test_iou / final_test_hd95（评估是否完全确定，需以各数据集的 prompt 扰动实现为准）

**通用损失函数**：`monai.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')`

### Finetune 过程中的 torch.compile 策略

当前 SAM 微调训练采用的 compile 策略是：

- 只对 `vision_encoder` 使用 `torch.compile`
- 不对 `prompt_encoder` 使用 `torch.compile`
- 不对 `mask_decoder` 使用 `torch.compile`
- 对应实现位于 `utils/finetune_engine.py`，当前配置为：`torch.compile(model.vision_encoder, mode='default')`

这样做的原因是：

- `vision_encoder` 是计算量主体，编译它最有机会换来真实训练提速
- `prompt_encoder` 参数量极小，compile 收益很有限
- `mask_decoder` 在 `notebooks/compile_prompt_mask_decoder_check.ipynb` 的验证里前向链路失败，不适合直接 compile

根据该 notebook 的结论：

- `prompt_encoder` 的 `compile_static`、`compile_dynamic`、`compile_reduce_overhead_dynamic` 都能运行，但与 eager 输出不完全一致
- `mask_decoder` 的 eager 和以上 compile 配置在该验证脚本里都失败，报 `TypeError`
- 因此当前仓库选择“只编译 `vision_encoder`”作为更稳妥的训练加速方案

---

## 各脚本说明

### neu_finetune.py — NEU-SEG 钢材表面缺陷

| 项目 | 说明 |
|------|------|
| 数据加载 | `create_neu_dataset_stratified()` |
| 批处理函数 | `_process_batch` |
| DDP 支持 | ✓（DistributedSampler + setup_ddp） |
| 权重保存 | `./new_weights/neu_seg_output/finetune/{ft_type}/` |
| 特殊功能 | 启动时打印数据集统计；zero-shot 模式使用 MedSam |

---

### sd900_finetune.py — SD900 显著性目标分割

| 项目 | 说明 |
|------|------|
| 数据加载 | `sd900_finetune_create_dataset()` + `sd900_finetune_create_dataloader()` |
| 批处理函数 | `_process_batch`（默认）或 `_process_batch_sam_style`（`--sam_style_train`） |
| DDP 支持 | ✗ |
| 权重保存 | `./new_weights/finetune/sd900_output/{ft_type}/` |
| 特殊功能 | 可选 SAM 风格多轮迭代提示训练；推理模式支持多 checkpoint 批量评估 |

---

### severstal_finetune.py — Severstal 钢材缺陷检测

| 项目 | 说明 |
|------|------|
| 数据加载 | `severstal.traindf_preprocess()` → `SteelDataset_WithBoxPrompt()` |
| 批处理函数 | `_process_batch_severstal`（处理 256×1600 原始分辨率） |
| DDP 支持 | ✓ |
| 权重保存 | `./new_weights/finetune/severstal_output/{ft_type}/` |
| 专用参数 | `--include_no_defect` 包含无缺陷样本；`--mini_dataset` 256 样本调试子集 |
| 特殊功能 | 预测掩码需从 letterbox 裁剪有效区域并上采样回 256×1600 |

---

### magnetic_tile_finetune.py — 磁砖表面缺陷

| 项目 | 说明 |
|------|------|
| 数据加载 | `create_magnetic_dataset()` |
| 批处理函数 | `_process_batch` |
| DDP 支持 | ✗ |
| 权重保存 | `./new_weights/finetune/mag_output/{ft_type}/` |
| 特殊功能 | 启动时检查样本 shape/dtype/range；推理模式多 checkpoint 批量评估 |

---

### retina_finetune.py — 视网膜血管分割

| 项目 | 说明 |
|------|------|
| 数据加载 | `create_retina_dataset_ft()` 或 `create_retina_dataset_ft_kfold()` |
| 批处理函数 | `_process_batch` |
| DDP 支持 | ✗ |
| 权重保存 | `./new_weights/finetune/retina/`（K-fold 时 `retina/fold_{i}/`） |
| 专用参数 | `--use_kfold`、`--num_folds`(5)、`--fold_index`(-1 跑全部) |
| 特殊功能 | **唯一支持 K-fold 交叉验证的脚本**；自动汇总各 fold 的 mean±std 到 JSON |

---

### floodSeg_finetune.py — FloodSeg 洪水分割

| 项目 | 说明 |
|------|------|
| 数据加载 | `floodseg_create_dataset()` |
| 批处理函数 | `_process_batch_with_point_grid`（16×16 密集点网格） |
| DDP 支持 | ✗ |
| 权重保存 | `./pretrained_weights/flood_seg_output/{ft_type}/` |
| 特殊功能 | **唯一使用点网格提示（非 bbox）的脚本**；推理模式尚未完善 |

---

## 对比总览

| 脚本 | 数据集 | 批处理函数 | DDP | K-Fold | 提示类型 |
|------|--------|-----------|-----|--------|---------|
| neu_finetune | NEU-SEG | `_process_batch` | ✓ | ✗ | Box |
| sd900_finetune | SD900 | `_process_batch` / `_sam_style` | ✗ | ✗ | Box / 多轮混合(正在测试多轮混合是否会提升性能) |
| severstal_finetune | Severstal | `_process_batch_severstal` | ✓ | ✗ | Box |
| magnetic_tile_finetune | Magnetic Tile | `_process_batch` | ✗ | ✗ | Box |
| retina_finetune | Retina | `_process_batch` | ✗ | ✓ | Box |
| floodSeg_finetune | FloodSeg | `_process_batch_with_point_grid` | ✗ | ✗ | 点网格 |

## 运行示例

```bash
# 单 GPU 训练
python -m train.neu_finetune \
    --batch_size 4 --learning_rate 2e-4 \
    --ft_type loradsc_qv --num_epochs 50 \
    --swanlab --pj_name neu_loradsc --device_id 0

# DDP 多 GPU 训练（NEU/Severstal）
torchrun --nproc_per_node=2 -m train.neu_finetune \
    --batch_size 2 --learning_rate 2e-4 \
    --ft_type loradsc_qv --num_epochs 50

# K-fold 交叉验证（Retina）
python -m train.retina_finetune \
    --ft_type loradsc_qv --use_kfold --num_folds 5 --fold_index -1

# Zero-shot 评估
python -m train.neu_finetune --zero_shot
```
