import argparse


# ─── 共享参数组 ───────────────────────────────────────────────

def _add_base_args(parser):
    """所有训练脚本共享的基础参数"""
    parser.add_argument('--batch_size', type=int, default=16, help='训练的batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='训练的学习率')
    parser.add_argument('--num_epochs', type=int, default=100, help='训练的总轮数 (epochs)')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')

    parser.add_argument('--patience', type=int, default=10, help='早停容忍的epoch数')
    parser.add_argument('--min_delta', type=float, default=0.0001, help='验证指标提升的最小阈值')
    parser.add_argument('--disable_early_stop', action='store_true', help='关闭早停以进行全周期训练')

    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device_id', type=int, default=0, help='gpu id')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers')

    parser.add_argument('--no_compile', action='store_true', help='禁用 torch.compile 以节省显存')
    parser.add_argument('--infer_mode', action="store_true", help='推理模式')
    parser.add_argument('--use_swanlab', '--swanlab', action='store_true', help='是否使用 swanlab 记录')
    parser.add_argument('--swanlab_project', '--pj_name', type=str, default='input your project name', help='swanlab项目名称')
    return parser


def _add_baseline_args(parser):
    """baseline 独有参数"""
    parser.add_argument('--bse_model', type=str, default='unet_res34', help='baseline model')
    parser.add_argument('--save_bse_model', '--save', action='store_true', help='是否保存baseline模型')
    return parser


def _add_finetune_args(parser):
    """finetune 独有参数"""
    # 覆盖基础默认值
    parser.set_defaults(batch_size=2, num_epochs=50)

    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='学习率预热 (warmup) 的比例')

    # LoRA 相关参数
    parser.add_argument('--lora_rank', type=int, default=16, help='LoRA 的秩 (rank)')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA 的 alpha 值')
    parser.add_argument('--lora_dropout', type=float, default=0.0, help='LoRA 的 dropout 比例')
    parser.add_argument('--use_loraplus_optim', action='store_true', help='是否启用 LoRA+ 优化器参数组')
    parser.add_argument('--lora_plus_lr_ratio', type=float, default=16.0, help='LoRA+ 中 B 组学习率相对 A 组的倍数')
    parser.add_argument('--moe_expert_type', type=str, default='conv', choices=['conv', 'linear', 'lora_conv'],
                        help='MoE专家类型: conv=DSC卷积(默认), linear=纯线性, lora_conv=线性+DSC串联')

    # 环境与设备
    parser.add_argument('--local-rank', '--local_rank', type=int, default=-1, help='torchrun DDP 自动传入，无需手动设置')

    # 微调与保存策略
    parser.add_argument('--ft_type', type=str, default="loradsc_qv", help='微调方法类型')
    parser.add_argument('--save_custom_lora', action="store_true", help='是否只保存自定义的LoRA参数')
    parser.add_argument('--save_hf_format', action="store_true", help='是否按照Hugging Face的格式保存模型')

    # 推理与数据模式
    parser.add_argument('--auto_seg', action="store_true", help='是否自动分割而不使用prompt')
    parser.add_argument('--sam_style_train', action="store_true", help='使用SAM原始多prompt多轮迭代训练策略')
    parser.add_argument('--zero_shot', action="store_true", help='是否进行zero-shot评估')

    # 交叉验证
    parser.add_argument('--use_kfold', action='store_true', help='是否启用 K 折交叉验证')
    parser.add_argument('--num_folds', type=int, default=5, help='K 折交叉验证中的折数')
    parser.add_argument('--fold_index', type=int, default=-1, help='指定运行某一折，-1 表示运行全部折')

    # sam模型大小选择
    parser.add_argument('--sam_type', type=str, default="sam_base", help='sam模型大小选择')

    # multimask 消融实验
    parser.add_argument('--multimask', action='store_true',
                        help='使用 multimask_output=True + best IoU selection (默认 False, 即 single mask)')
    return parser


def _add_severstal_args(parser):
    """Severstal 数据集特有参数"""
    parser.add_argument('--include_no_defect', action="store_true", help='是否在训练中引入无缺陷样本')
    parser.add_argument('--mini_dataset', action='store_true', help='是否使用一个小型子集进行快速调试')
    return parser


# ─── 对外接口（签名和返回值保持不变） ─────────────────────────

def get_bse_args():
    parser = argparse.ArgumentParser(description='基线训练模型的args选择')
    _add_base_args(parser)
    _add_baseline_args(parser)
    bse_args = parser.parse_args()
    return bse_args


def add_common_ft_args(parser):
    """向ArgumentParser对象中添加通用的微调参数"""
    _add_base_args(parser)
    _add_finetune_args(parser)
    return parser


def get_common_ft_args():
    """获取通用的微调模型参数"""
    parser = argparse.ArgumentParser(description='通用微调模型的参数配置')
    parser = add_common_ft_args(parser)
    args = parser.parse_args()
    return args


def get_severstal_ft_args():
    """获取针对Severstal数据集的微调模型参数"""
    parser = argparse.ArgumentParser(description='Severstal数据集微调模型的参数配置')
    parser = add_common_ft_args(parser)
    _add_severstal_args(parser)
    args = parser.parse_args()
    return args


def get_severstal_bsl_args():
    """获取针对Severstal数据集的baseline模型参数"""
    parser = argparse.ArgumentParser(description='Severstal基线训练模型的args选择')
    _add_base_args(parser)
    _add_baseline_args(parser)
    _add_severstal_args(parser)
    args = parser.parse_args()
    return args
