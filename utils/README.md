# Utils 文件夹说明

本文件夹包含 DefectSAM 项目的核心工具模块，涵盖模型架构定义、训练引擎、配置管理和辅助函数。

---

## config.py — 统一参数配置

集中管理所有实验的命令行参数。

| 公开接口 | 用途 |
|---------|------|
| `get_bse_args()` | Baseline 训练参数 |
| `get_common_ft_args()` | SAM 微调通用参数 |
| `get_severstal_ft_args()` | Severstal 微调参数 |
| `get_severstal_bsl_args()` | Severstal Baseline 参数 |

**基础参数** (`_add_base_args`)：`batch_size`、`learning_rate`、`num_epochs`、`weight_decay`、`patience`、`device_id`、`num_workers`、`no_compile`、`use_swanlab` 等。

**微调专用参数** (`_add_finetune_args`)：
- LoRA：`lora_rank`(16)、`lora_alpha`(16)、`lora_dropout`(0.0)
- LoRA+：`use_loraplus_optim`、`lora_plus_lr_ratio`(16.0)
- MoE：`moe_expert_type` ∈ {`conv`, `linear`, `lora_conv`}
- 微调策略：`ft_type`(默认 `loradsc_qv`)、`save_custom_lora`、`save_hf_format`
- 推理模式：`auto_seg`、`sam_style_train`、`zero_shot`
- 交叉验证：`use_kfold`、`num_folds`、`fold_index`

---

## sam_arch.py — LoRA 变体与模型工厂（核心）

定义所有自定义 LoRA 变体的注入结构，以及将其应用到 SAM 视觉编码器的工厂函数。

### LoRA 变体类继承关系

```
LoRACore (基类：冻结原始 qkv，定义 rank/alpha/scale)
├── LoRASam_Plus             — 标准 LoRA，支持 LoRA+ 差异化学习率
├── LoRASam_DepWiseConv      — LoRA + 深度可分离卷积 (DSC)
│   ├── LoRASam_DepWiseConv_Residual  — DSC 残差连接
│   └── LoRASam_DepWiseConv_Gated     — 可学习门控标量
└── LoRA_Moe_DepwiseConv_Samqv (独立)  — MoE 多专家路由
```

### 各变体详细说明

#### LoRASam_Plus — 标准 LoRA (支持 LoRA+)
- **结构**：`x → lora_a → lora_b → × scale`
- **特点**：通过 `get_loraplus_param_groups()` 为 A/B 矩阵设置不同学习率

#### LoRASam_DepWiseConv — LoRA-DSC
- **结构**：`x → lora_a → [DW Conv → PW Conv] → lora_b → × scale`
- **DSC 路径**：将低秩特征 reshape 为 2D（`[B, rank, H, W]`），经深度卷积 + 逐点卷积后还原
- **参数 `add_dsc_conv=False` 时退化为标准 LoRA**

#### LoRASam_DepWiseConv_Residual — 残差 LoRA-DSC
- **结构**：`lora_b(lora_a(x) + DSC(lora_a(x))) × scale`
- **优势**：DSC 无效时保留纯 LoRA 信号，降低优化难度

#### LoRASam_DepWiseConv_Gated — 门控 LoRA-DSC
- **附加参数**：每个 q/k/v 分支一个可学习门控标量
- **结构**：`gate[part] × LoRA-DSC_delta(x)`

#### LoRA_Moe_DepwiseConv_Samqv — MoE LoRA
- **参数**：`num_experts=3`、`kernel_sizes=[3,5,7]`、`expert_type` ∈ {`conv`, `linear`, `lora_conv`}
- **结构**：
  1. `x → lora_a` 投影到低秩空间
  2. 门控网络 `gate(delta)` → softmax 得到专家权重
  3. 各专家独立处理 delta，按权重加权求和
  4. `→ lora_b → × scale`
- **门控诊断**：forward 时缓存 `_last_gate_probs_q/v`，供 `collect_moe_gate_stats()` 采集 entropy/max_prob

### 模型工厂函数

| 函数 | 注入类 | 微调目标 |
|------|--------|---------|
| `get_loradsc_model()` | LoRASam_DepWiseConv | Q, V（可配置） |
| `get_loradsc_residual_model()` | LoRASam_DepWiseConv_Residual | Q, V |
| `get_loradsc_global_only_model()` | LoRASam_DepWiseConv | Q, V（仅全局注意力层注入 DSC） |
| `get_loradsc_gated_model()` | LoRASam_DepWiseConv_Gated | Q, V |
| `get_loraplus_model()` | LoRASam_Plus | Q, V |
| `get_moelora_model()` | LoRA_Moe_DepwiseConv_Samqv | Q, V |
| `get_loraga_model()` | SVD 初始化 LoRA | Q, V |
| `get_lorapro_model()` | LoRA + Sylvester 梯度修正 | Q, V |

所有工厂函数遵循相同模式：冻结 SAM 全部参数 → 替换 `image_encoder.blocks[i].attn.qkv` → 返回模型。

### 推理与工具函数

| 函数 | 功能 |
|------|------|
| `create_model_for_inference()` | 加载 LoRA 权重到模型，处理 `_orig_mod` 前缀 |
| `measure_inference_time()` | 基准测试推理速度 |

---

## loratask.py — HuggingFace PEFT 封装

基于 `peft` 库的 LoRA/AdaLoRA/LoHa/LoKr 封装，用于 SAM 的 HF 风格微调。

| 函数 | PEFT 方法 | 说明 |
|------|----------|------|
| `get_hf_lora_model()` | LoRA | 自动检测 target modules |
| `get_hf_dora_qv_model()` | DoRA | 先拆分 fused `qkv`，再仅对 `q_proj` / `v_proj` 注入 DoRA |
| `get_hf_lokr_qv_model()` | LoKr | 先拆分 fused `qkv`，再仅对 `q_proj` / `v_proj` 注入 LoKr |
| `get_hf_adalora_model()` | AdaLoRA | 自适应秩，训练中从 `init_r` 剪枝至 `target_r` |
| `get_hf_loha_model()` | LoHa | Hadamard 门控积 |
| `get_hf_lokr_model()` | LoKr | Kronecker 积 |

辅助函数：
- `prepare_sam_qkv_for_qv_peft(model, target_part)` — 将 fused `qkv` 包装为暴露 `q_proj/k_proj/v_proj` 的等价模块
- `get_sam_target_modules(model, target_part)` — 动态检测 SAM 中的 qkv 线性层名称
- `get_sam_qv_target_modules_for_peft(model, target_part)` — 仅枚举 strict q/v PEFT 的 `q_proj` / `v_proj` 目标
- `filter_target_modules(named_modules, target_substrings)` — 按子串过滤 `nn.Linear` 模块

---

## finetune_engine.py — SAM 微调训练引擎（核心）

### 模型路由

`create_model_from_type(args)` 根据 `ft_type` 字符串调用对应工厂函数：

| ft_type | 模型类 | DSC | 残差 | 门控 | MoE | LoRA+ |
|---------|--------|-----|------|------|-----|-------|
| `loradsc_qv` | LoRASam_DepWiseConv | ✓ | | | | |
| `lora_attn_qv` | LoRASam_DepWiseConv (无DSC) | | | | | |
| `loradsc_qv_residual` | LoRASam_DepWiseConv_Residual | ✓ | ✓ | | | |
| `loradsc_qv_gated` | LoRASam_DepWiseConv_Gated | ✓ | | ✓ | | |
| `loradsc_qv_global` | LoRASam_DepWiseConv (仅全局层) | ✓ | | | | |
| `loraplus_qv` | LoRASam_Plus | | | | | ✓ |
| `moelora_qv` | LoRA_Moe_DepwiseConv_Samqv | ✓ | | ✓ | ✓ | |
| `moeloraplus_qv` | LoRA_Moe_DepwiseConv_Samqv | ✓ | | ✓ | ✓ | ✓ |
| `loraga_qv` | SVD 初始化 LoRA | | | | | |
| `lorapro_qv` | LoRA + 梯度修正 | | | | | |
| `lora_encoder` | HF PEFT LoRA | | | | | |
| `dora_qv_encoder` | HF PEFT DoRA（strict q/v） | | | | | |
| `lokr_qv_encoder` | HF PEFT LoKr（strict q/v） | | | | | |
| `adalora_encoder` | HF PEFT AdaLoRA | | | | | |
| `sam_fully` | 全参数微调 | | | | | |
| `sam_decoder` | 仅解码器微调 | | | | | |

### Hugging Face DoRA（strict q/v）

- `dora_qv_encoder` 不会直接对 fused `qkv` 整层注入 adapter；它会先把 `vision_encoder.layers.*.attn.qkv` 包装成 `q_proj/k_proj/v_proj` 三个子层，再仅对 `q_proj` 和 `v_proj` 应用 PEFT DoRA。
- 该模式只支持 Hugging Face PEFT 的 `save_pretrained` / `PeftModel.from_pretrained` 路线，不能与 `save_custom_lora` 组合使用。
- 推荐快速验证命令：

```bash
python -m train.neu_finetune \
    --batch_size 4 \
    --learning_rate 2e-4 \
    --ft_type dora_qv_encoder \
    --num_epochs 10 \
    --save_hf_format \
    --swanlab \
    --pj_name neu_dora_qv_quicktest \
    --device_id 0
```

### Hugging Face LoKr（strict q/v）

- `lokr_qv_encoder` 与 `dora_qv_encoder` 一样，都会先拆分 fused `qkv`，再仅对 `q_proj` 和 `v_proj` 应用 PEFT adapter；区别是这里使用 `LoKrConfig(r=args.lora_rank, alpha=args.lora_alpha)`。
- 该模式同样只支持 Hugging Face PEFT 的 `save_pretrained` / `PeftModel.from_pretrained` 路线，不能与 `save_custom_lora` 组合使用。
- 推荐快速验证命令：

```bash
python -m train.neu_finetune \
    --batch_size 4 \
    --learning_rate 2e-4 \
    --ft_type lokr_qv_encoder \
    --lora_rank 16 \
    --lora_alpha 16 \
    --num_epochs 10 \
    --save_hf_format \
    --swanlab \
    --pj_name neu_lokr_qv_quicktest \
    --device_id 0
```

### 批处理函数

| 函数 | 适用场景 | 特殊处理 |
|------|---------|---------|
| `_process_batch()` | 通用（SD900/NEU/Magnetic等） | GT 下采样至 256×256 匹配预测 |
| `_process_batch_severstal()` | Severstal（256×1600） | 从 letterbox 预测裁剪有效区域，上采样回原始分辨率 |
| `_process_batch_sam_style()` | 多轮迭代提示训练 | 第1轮随机 prompt 类型，后续轮次用误差点修正 |
| `_process_batch_with_point_grid()` | 密集点网格推理 | 16×16 前景点网格 |

### 训练主循环 `run_finetune_engine()`

1. **torch.compile**：仅编译 `vision_encoder`（非全模型）
2. **LoRA+ 优化器**：从模型的 `get_loraplus_param_groups()` 构建差异化参数组
3. **学习率**：Cosine + Linear Warmup
4. **早停**：基于 val_dice 监控
5. **DDP 支持**：多 GPU 分布式训练
6. **MFU 追踪**：每 batch 估算模型算力利用率
7. **MoE 诊断**：每 epoch 采集 gate entropy/max_prob 并 log 到 SwanLab
8. **模型保存**：支持全量 / LoRA-only / HuggingFace 三种格式

### SAM compile 决策与参数规模

基于 `notebooks/compile_prompt_mask_decoder_check.ipynb` 的实测结果，当前仓库对 SAM 的 compile 策略做如下约束：

- 推荐：`torch.compile(model.vision_encoder, mode='default')`
- 不推荐：对 `prompt_encoder` 使用 `compile_static`、`compile_dynamic` 或 `compile_reduce_overhead_dynamic`
- 不推荐：对 `mask_decoder` 使用上述 compile 配置

原因总结：

- `vision_encoder` 是 SAM 的绝大多数参数和主要算力开销来源，编译它最可能带来训练提速
- `prompt_encoder` 在 notebook 中虽然可以 compile，但与 eager 输出不完全一致
- `mask_decoder` 在 notebook 的验证脚本中 eager 和 compile 路径都失败，错误为 `TypeError`，因此当前不作为 compile 目标

`sam_vit_base` 三大模块参数量如下：

| 模块 | 参数量 | 约占总参数比例 |
|------|--------|----------------|
| `vision_encoder` | `89,670,912` | `95.66%` |
| `prompt_encoder` | `6,476` | `0.0069%` |
| `mask_decoder` | `4,058,340` | `4.33%` |
| 合计 | `93,735,728` | `100%` |

因此，从工程收益上看：

- compile `vision_encoder`：高优先级
- compile `prompt_encoder`：低收益，不值得为它承担数值一致性风险
- compile `mask_decoder`：当前验证条件下不可取

### MoE 门控诊断

`collect_moe_gate_stats(model)` — 遍历所有 MoE 层，采集：
- `expert_{j}_mean`：各专家的平均选择概率
- `entropy`：路由熵（越高越均衡）
- `max_prob`：最大专家概率均值（越高越趋向坍塌）

---

## baseline_engine.py — Baseline 训练引擎

为传统分割模型（UNet/DeepLabV3+/SegFormer）提供训练循环。

| 函数 | 功能 |
|------|------|
| `create_bsl_model_from_type(args)` | 根据 `bse_model` 创建模型（`unet_res34` / `deeplabv3plus_effb3` / `segformer_b2` 等） |
| `train_one_epoch()` | 单 epoch 训练，返回 `(loss, dice, iou, hd95)` |
| `evaluate()` | 验证/测试评估 |
| `baseline_experiment()` | 完整训练循环（早停 + SwanLab + torch.compile + 最优模型保存） |
| `bsl_inference_engine()` | 加载权重并评估所有数据集划分 |

---

## helper_function.py — 通用辅助函数

| 函数 | 功能 |
|------|------|
| `setup_ddp()` | 检测并初始化 DDP 环境，返回 rank/device/world_size 等信息 |
| `cleanup_ddp()` | 销毁进程组 |
| `set_seed(seed, seed_offset)` | 设置 random/numpy/torch 随机种子 |
| `get_lr_scheduler(optimizer, warmup_steps, total_steps)` | Cosine + Linear Warmup 调度器 |
| `get_bounding_box(mask, perturb, perturb_range)` | 从二值掩码提取 bbox，训练时可加 ±20px 扰动 |

---

## mfu.py — 模型算力利用率估算

### SAMMFUEstimator

根据 SAM ViT 架构理论计算 FLOPs：

| SAM 变体 | embed_dim | depth | num_heads | 全局注意力层 |
|----------|-----------|-------|-----------|-------------|
| sam_base | 768 | 12 | 12 | [2, 5, 8, 11] |
| sam_large | 1024 | 24 | 16 | [5, 11, 17, 23] |
| sam_huge | 1280 | 32 | 16 | [7, 15, 23, 31] |

**FLOPs 计算**：
- 每块 MatMul：`24 × d² × T`（QKV + 输出投影 + MLP）
- 窗口注意力：`2 × T_w × T × d`
- 全局注意力：`2 × T² × d`
- 总 FLOPs = 3 × forward FLOPs（含反向传播）

**GPU 峰值算力** (BF16)：RTX 5090 = 838 TFLOPS, RTX 4090 = 330 TFLOPS

### MFUTracker
- 指数移动平均 (EMA) 跟踪每步 MFU
- `update(dt)` 更新，`mfu` 属性获取当前值

---

## swanlab_viz.py — SwanLab 实验可视化

CLI 工具，从 SwanLab 拉取实验数据并绘图。

```bash
python -m utils.swanlab_viz --projects sd900_ada --metrics train/loss val/dice
```

| 函数 | 功能 |
|------|------|
| `list_experiments()` | 列出项目下的实验（可按关键词过滤） |
| `fetch_metrics()` | 拉取指定实验的指标曲线 |
| `collect_all_data()` | 聚合多个项目的数据 |
| `print_summary_table()` | 打印 ASCII 对比表 |
| `plot_metrics()` | 生成 matplotlib 对比图 |

---

## utils.py — 通用工具与指标

### 损失函数

| 类 | 说明 |
|----|------|
| `DiceLoss` | Dice 系数损失 |
| `DiceBCELoss` | Dice + BCE 组合 |
| `Combine_DiceCrossEntropy_Loss` | 加权 Dice + CrossEntropy |
| `MultiClassDiceLoss` | 多类别逐通道 Dice |

### 指标计算

| 函数 | 说明 |
|------|------|
| `compute_dice_score(pred, target)` | sigmoid → 阈值 0.5 → Dice 系数 |
| `compute_iou_score(pred, target)` | sigmoid → 阈值 0.5 → IoU |

### 模型保存

| 函数 | 说明 |
|------|------|
| `save_lora_parameters(model, filename)` | 仅保存 `requires_grad=True` 的参数（处理 `_orig_mod` 前缀） |
| `save_model(...)` | 三种模式：HuggingFace / LoRA-only / 全量 checkpoint |

### 其他

| 函数 | 说明 |
|------|------|
| `print_trainable_parameters(model)` | 打印可训练参数量与占比 |
| `save_training_logs(...)` | 保存训练日志为 JSON（超参、时间戳、指标历史） |
| `get_loss_fn(loss_name)` | 按名称获取损失函数 |
