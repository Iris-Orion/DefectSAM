"""
SAM-family fine-tuning (single GPU) on the Magnetic-Tile defect subset. Reuses the whole pipeline and arguments of
train/magnetic_tile_finetune.py and patches three things at runtime without editing that file:
  1. Data: create_magnetic_dataset -> data.magnetic_tile_defect.create_magnetic_defect_dataset (drops MT_Free).
  2. Metrics: training-time Dice / IoU are computed per image (utils.per_image_metric_patch); best-epoch selection on val uses per-image Dice too.
  3. Output: weights and JSON go to new_weights/finetune/mag_defect_output/<ft_type>; task_name prefix becomes magnetic_defect_.
Usage (from the repo root, same arguments as the original script):
  python -m train.magnetic_tile_defect_finetune --ft_type loradsc_qv_residual_gated --batch_size 4 --learning_rate 2e-4 \
      --lora_rank 16 --lora_alpha 16 --use_loraplus_optim --lora_plus_lr_ratio 8 --disable_early_stop --seed 42 --device_id 0
"""
import runpy

import data.magnetic_tile_dataset as mag_data
import utils.finetune_engine as fe
from data.magnetic_tile_defect import create_magnetic_defect_dataset
from utils.per_image_metric_patch import install

install()
mag_data.create_magnetic_dataset = create_magnetic_defect_dataset
_run = fe.run_finetune_engine


def run_finetune_engine(*args, **kwargs):
    """Wrap the original engine: rewrite save_dir and task_name, pass everything else through."""
    kwargs["save_dir"] = kwargs["save_dir"].replace("/mag_output/", "/mag_defect_output/")
    hp = kwargs.get("hyperparameters", args[5] if len(args) > 5 else None)
    hp["task_name"] = hp["task_name"].replace("magnetic_", "magnetic_defect_", 1)
    return _run(*args, **kwargs)


fe.run_finetune_engine = run_finetune_engine

if __name__ == "__main__":
    runpy.run_module("train.magnetic_tile_finetune", run_name="__main__", alter_sys=True)
