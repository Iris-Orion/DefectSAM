# Data 文件夹说明

本文件夹负责所有数据集的加载、预处理与划分，为 baseline 训练和 SAM 微调提供统一的 Dataset/DataLoader 接口。

如需复现实验，请下载数据集并解压到本目录下，目录结构见[底部](#数据集目录结构)。

---

## 公共模块

### common_ops.py — 共享工具函数

| 函数 | 功能 |
|------|------|
| `letterbox_image_np(image, size)` | 保持宽高比缩放图像，空白区域填充 128（灰色），使用 `cv2.INTER_CUBIC` |
| `letterbox_mask_np(mask, size)` | 保持宽高比缩放掩码，空白区域填充 0，使用 `cv2.INTER_NEAREST`（保持边缘锐利） |
| `build_binary_mask(mask, threshold=0)` | 二值化掩码：`(mask > threshold).astype(float32)` |
| `build_stratify_keys(mask_dir, mask_files)` | 根据掩码中的标签组合生成分层键（如 `"0_1_3"`），用于分层划分 |
| `split_stratified_6_2_2(labels, seed=42)` | 两步分层划分：60% 训练 / 20% 验证 / 20% 测试 |

### sam_dataset_base.py — SAM 微调数据集基类

`SegDatasetForFinetune(Dataset)` 提供 `letterbox_imagenp()` 和 `letterbox_mask_1ch()` 方法，供所有 SAM 微调数据集子类复用。

---

## Baseline vs SAM 微调：两套数据流

每个数据集文件都提供两种 Dataset 类，对应不同的训练框架：

| 特性 | Baseline 数据集 | SAM 微调数据集 |
|------|----------------|---------------|
| 图像尺寸 | Resize 到 **256×256** | Letterbox 到 **1024×1024** |
| 返回格式 | `(image, mask, label/id)` 元组 | `{"image", "mask", "bbox", ...}` 字典 |
| Bbox 提示 | 无 | 从增强后的掩码计算 `[x_min, y_min, x_max, y_max]` |
| 归一化 | `image / 255.0` | `image / 255.0` |
| 数据增强 | Resize + HFlip + VFlip + ToTensorV2 | HFlip + VFlip + ToTensorV2（无 Resize，letterbox 在 `__getitem__` 中完成） |

**Bbox 生成流程**（SAM 微调）：
1. 对图像和掩码施加空间增强（翻转等）
2. Letterbox 到 1024×1024
3. 从增强后的掩码计算 bounding box（确保 bbox 与增强后的掩码一致）
4. 训练时可对 bbox 施加 ±[0, 20) 像素的随机扰动
5. **验证/测试时关闭随机扰动（`perturb=False`），确保验证集指标完全可复现**
6. 空掩码返回 `[0, 0, 0, 0]`

---

## 各数据集文件

### neu_dataset.py — NEU-SEG 钢材表面缺陷

| 类 | 用途 |
|----|------|
| `NEU_SEG_Dataset_BSL` | Baseline（256×256，元组返回） |
| `NEU_SEG_Dataset` | SAM 微调（1024×1024，bbox 提示） |

**数据格式**：JPG 图像 + PNG 掩码（灰度值 1-3 → 二值化）

| 工厂函数 | 划分方式 | 返回 |
|---------|---------|------|
| `neu_bsl_create_dataset()` | 原始训练集 75:25 分层划分 + 独立测试集 | `(train, val, test)` Baseline |
| `create_neu_dataset_stratified()` | 同上 | `(train, val, test)` SAM FT |

---

### sd900_dataset.py — SD900 显著性目标分割

| 类 | 用途 |
|----|------|
| `SDsaliency900Dataset_BSL` | Baseline |
| `SDsaliency900Dataset_FT` | SAM 微调（训练时 bbox 带扰动） |

**数据格式**：BMP 图像 + PNG 掩码。按文件名前缀分 3 类（`In_` 室内 / `Pa_` 绘画 / `Sc_` 场景）。

| 工厂函数 | 划分方式 | 返回 |
|---------|---------|------|
| `sd900_bsl_create_dataset()` | 6:2:2 分层划分 | `(train, val, test, labels)` |
| `sd900_finetune_create_dataset()` | 6:2:2 分层划分 | `(train, val, test)` |
| `sd900_finetune_create_dataloader()` | 从已有 dataset 创建 DataLoader | `(train_loader, val_loader, test_loader)` |

辅助：`get_label_distribution(dataset)` 统计各类样本数。

---

### severstal.py — Severstal 钢材缺陷检测

| 类 | 用途 |
|----|------|
| `SteelDataset` | Baseline（4 通道掩码求和为 1 通道） |
| `SteelDataset_WithBoxPrompt` | SAM 微调（额外返回 `letterboxed_mask`） |

**数据格式**：JPG 图像 + CSV 中的 **RLE 编码掩码**（4 类缺陷，每类独立编码）。

**RLE 解码机制** (`rle2mask`):
- 优先从 `.npz` 压缩缓存读取，若未命中则走原始解码。
- 原始解码：读取 `EncodedPixels` 列，按列优先 reshape 为 `[256, 1600]`，多类独立解码合并。
- **缓存策略优化**：生成的掩码直接指定为极省空间的 `np.uint8`，并使用 `np.savez_compressed` 保存，体积相较于原始 `float32` npy 缩减 99% 以上。

**特有功能**：
- `traindf_preprocess()` — 从 CSV 构建 DataFrame，支持 `--include_no_defect`（包含无缺陷样本）和 `--mini_dataset`（调试子集）
- `reverse_letterbox_mask_1ch()` — 反向 letterbox，将预测掩码还原为原始分辨率

| 工厂函数 | 划分方式 | 返回 |
|---------|---------|------|
| `create_datasets_no_prompt()` | 接收已划分的 DataFrame | `(train, val, test)` |
| `create_dataset_with_prompt()` | 内部调用 `traindf_preprocess()` 划分 | `(train, val, test)` |

---

### magnetic_tile_dataset.py — 磁砖表面缺陷

| 类 | 用途 |
|----|------|
| `MagneticTileDataset_Baseline` | Baseline（letterbox 256×256，阈值 0.5） |
| `MagneticTileDatasetWithBoxPrompt` | SAM 微调（letterbox 1024×1024） |

**数据格式**：按类别子目录组织（`MT_Blowhole/Imgs/`、`MT_Crack/Imgs/` 等），JPG 图像 + PNG 掩码。

| 工厂函数 | 划分方式 | 返回 |
|---------|---------|------|
| `create_mag_dataset_baseline()` | 6:2:2 分层划分 | `(train, val, test)` |
| `create_magnetic_dataset()` | 6:2:2 分层划分 | `(train, val, test)` SAM FT |

辅助：`magtile_find_classes(dir)` 扫描类别子目录；`magtile_get_all_imgmsk_paths(root)` 收集所有图像/掩码路径。

---

### retina_dataset.py — 视网膜血管分割

| 类 | 用途 |
|----|------|
| `Retina_Dataset_Bsl` | Baseline |
| `Retina_Dataset_ft` | SAM 微调（继承 `SegDatasetForFinetune`） |

**数据格式**：PNG 图像 + PNG 掩码。包含独立的测试文件夹。

| 工厂函数 | 划分方式 | 返回 |
|---------|---------|------|
| `create_retina_dataset_baseline()` | 训练集随机分出 20 张作为验证集 + 独立测试集 | `(train, val, test)` |
| `create_retina_dataset_ft()` | 同上 | `(train, val, test)` |
| `create_retina_dataset_ft_kfold(num_folds, fold_index)` | K-Fold 交叉验证 | `(train, val, test)` |

---

### floodseg_ft.py — FloodSeg 洪水分割

仅提供一个 Dataset 类 `FloodSegDataset`（SAM 微调，Resize 到 1024×1024，无 letterbox），无 bbox 提示。

| 工厂函数 | 划分方式 | 返回 |
|---------|---------|------|
| `floodseg_create_dataset()` | 6:2:2 随机划分（`torch.random_split`） | `(train, val, test)` |

---

## 数据划分策略汇总

| 数据集 | 划分比例 | 方法 | 分层依据 |
|--------|---------|------|---------|
| NEU-SEG | 75:25 + 独立测试集 | `StratifiedShuffleSplit` | 掩码标签组合 |
| SD900 | 60:20:20 | `split_stratified_6_2_2` | 文件名前缀类别 |
| Severstal | 60:20:20 | `traindf_preprocess` | — |
| Magnetic Tile | 60:20:20 | `split_stratified_6_2_2` | 目录类别 |
| Retina | 随机 20 张验证 + 独立测试集 | `random_split` | — |
| FloodSeg | 60:20:20 | `torch.random_split` | — |

---

## 已弃用

- `bsd500_ft.py` — BSD500 数据集，不再使用。

---

## 数据集目录结构

如需复现实验，请将数据集解压到对应目录：

```
data/
├── NEU_Seg-main/
│   ├── annotations/
│   │   ├── test/
│   │   └── training/
│   └── images/
│       ├── test/
│       └── training/
├── sd900/
│   ├── Ground truth/
│   └── Source Images/
├── severstal_steel_defect_detection/
│   ├── train_images/
│   └── test_images/
├── Magnetic-Tile/
│   ├── MT_Blowhole/Imgs/
│   ├── MT_Break/Imgs/
│   ├── MT_Crack/Imgs/
│   ├── MT_Fray/Imgs/
│   ├── MT_Free/Imgs/
│   └── MT_Uneven/Imgs/
├── Retina_Blood_Vessel/
│   ├── train/
│   │   ├── image/
│   │   └── mask/
│   └── test/
│       ├── image/
│       └── mask/
├── FloodSeg/
│   ├── Image/
│   └── Mask/
├── Dataset_BUSI_with_GT/
│   ├── benign/
│   ├── malignant/
│   └── normal/
└── BSD500/          (已弃用)
