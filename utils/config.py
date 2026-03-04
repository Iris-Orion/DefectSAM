import argparse

def get_bse_args():
    parser = argparse.ArgumentParser(description='基线训练模型的args选择')
    parser.add_argument('--batch_size', type=int, default=24, help='训练的batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='训练的学习率')
    parser.add_argument('--num_epochs', type=int, default=100, help='epochs')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')

    parser.add_argument('--patience', type=int, default=10, help='早停容忍的epoch数')
    parser.add_argument('--min_delta', type=float, default=0.0001, help='验证指标提升的最小阈值')
    parser.add_argument('--disable_early_stop', action='store_true', help='关闭早停以进行全周期训练')

    parser.add_argument('--device_id', type=int, default=0, help='gpu id')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers')
    parser.add_argument('--bse_model', type=str, default='unet_res34', help='baseline model')

    parser.add_argument('--save_bse_model', action='store_true', help='是否保存baseline模型')
    parser.add_argument('--infer_mode', action="store_true", help='推理模式')

    parser.add_argument('--use_swanlab', action='store_true', help='是否使用 swanlab 记录')
    parser.add_argument('--swanlab_project', type=str, default='input your project name', help='swanlab项目名称')
    bse_args = parser.parse_args()  

    return bse_args


def add_common_ft_args(parser):
    """向ArgumentParser对象中添加通用的微调参数"""
    # 训练超参数
    parser.add_argument('--batch_size', type=int, default=2, help='训练的batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='训练的学习率')
    parser.add_argument('--num_epochs', type=int, default=100, help='训练的总轮数 (epochs)')
    
    # 早停与正则化
    parser.add_argument('--patience', type=int, default=10, help='早停(early stop)的耐心值')
    parser.add_argument('--min_delta', type=float, default=0.0001, help='验证指标被认为有提升的最小阈值')
    parser.add_argument('--disable_early_stop', action='store_true', help='关闭早停以进行全周期训练')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减 (weight_decay)')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='学习率预热 (warmup) 的比例')

    # LoRA 相关参数
    parser.add_argument('--lora_rank', type=int, default=16, help='LoRA 的秩 (rank)')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA 的 alpha 值')
    parser.add_argument('--lora_dropout', type=float, default=0.0, help='LoRA 的 dropout 比例')
    parser.add_argument('--use_loraplus_optim', action='store_true', help='是否启用 LoRA+ 优化器参数组')
    parser.add_argument('--lora_plus_lr_ratio', type=float, default=16.0, help='LoRA+ 中 B 组学习率相对 A 组的倍数')

    # 环境与设备
    parser.add_argument('--device_id', type=int, default=0, help='训练使用的设备ID (GPU)')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器 (DataLoader) 的工作进程数')

    # 微调与保存策略
    parser.add_argument('--ft_type', type=str, default="loradsc_qv", help='微调方法类型')
    parser.add_argument('--save_custom_lora', action="store_true", help='是否只保存自定义的LoRA参数')
    parser.add_argument('--save_hf_format', action="store_true", help='是否按照Hugging Face的格式保存模型')
    
    # 推理与数据模式
    parser.add_argument('--auto_seg', action="store_true", help='是否自动分割而不使用prompt')
    parser.add_argument('--infer_mode', action="store_true", help='是否在训练中引入无缺陷样本进行推理式评估')
    parser.add_argument('--zero_shot', action="store_true", help='是否进行zero-shot评估')

    parser.add_argument('--use_swanlab', action='store_true', help='是否使用 swanlab 记录')
    parser.add_argument('--swanlab_project', type=str, default='input your project name', help='swanlab项目名称')

    # sam模型大小选择
    parser.add_argument('--sam_type', type=str, default="sam_base", help='sam模型大小选择')
    
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
    
    # 添加Severstal特有的参数
    parser.add_argument('--include_no_defect', action="store_true", help='是否在训练中引入无缺陷样本')
    parser.add_argument('--mini_dataset', action='store_true', help='是否使用一个小型子集进行快速调试')
    
    # 如果需要，可以覆盖通用参数的默认值
    # parser.set_defaults(num_epochs=50, device_id=0)
    
    args = parser.parse_args()
    return args

def get_severstal_bsl_args():
    """获取针对Severstal数据集的baseline模型参数"""
    
    parser = argparse.ArgumentParser(description='Severstal基线训练模型的args选择')
    parser.add_argument('--batch_size', type=int, default=24, help='训练的batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='训练的学习率')
    parser.add_argument('--num_epochs', type=int, default=100, help='epochs')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')

    parser.add_argument('--patience', type=int, default=10, help='早停容忍的epoch数')
    parser.add_argument('--min_delta', type=float, default=0.0001, help='验证指标提升的最小阈值')
    parser.add_argument('--disable_early_stop', action='store_true', help='关闭早停以进行全周期训练')

    parser.add_argument('--device_id', type=int, default=2, help='gpu id')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers')
    parser.add_argument('--bse_model', type=str, default='unet_res34', help='baseline model')

    parser.add_argument('--save_bse_model', action='store_true', help='是否保存baseline模型')
    parser.add_argument('--infer_mode', action="store_true", help='推理模式')
    parser.add_argument('--use_swanlab', action='store_true', help='是否使用 swanlab 记录')
    parser.add_argument('--swanlab_project', type=str, default='input your project name', help='swanlab项目名称')

    # 添加Severstal特有的参数
    parser.add_argument('--include_no_defect', action="store_true", help='是否在训练中引入无缺陷样本')
    parser.add_argument('--mini_dataset', action='store_true', help='是否使用一个小型子集进行快速调试')
    
    # 如果需要，可以覆盖通用参数的默认值
    # parser.set_defaults(num_epochs=50, device_id=0)
    
    args = parser.parse_args()
    return args
