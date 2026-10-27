"""
Classic segmentation baselines on the Magnetic-Tile defect subset. Reuses the whole pipeline and arguments of
baseline/magnetic_tile_baseline.py and patches three things at runtime without editing that file:
  1. Data: create_mag_dataset_baseline -> data.magnetic_tile_defect.create_mag_defect_dataset_baseline (drops MT_Free).
  2. Metrics: training-time Dice / IoU are computed per image (utils.per_image_metric_patch); best-model selection on val uses per-image Dice too.
  3. Output: weights and JSON go to new_weights/baseline/magnetic_tile_defect_output/<bse_model>; task_name prefix becomes mag_defect_.
Usage (from the repo root, same arguments as the original script):
  python -m baseline.magnetic_tile_defect_baseline --bse_model unet_res34 --batch_size 16 --learning_rate 1e-4 \
      --num_epochs 100 --disable_early_stop --save --seed 42 --device_id 0
"""
import runpy

import data.magnetic_tile_dataset as mag_data
import utils.baseline_engine as be
from data.magnetic_tile_defect import create_mag_defect_dataset_baseline
from utils.per_image_metric_patch import install

install()
mag_data.create_mag_dataset_baseline = create_mag_defect_dataset_baseline
_run = be.baseline_experiment


def baseline_experiment(*args, **kwargs):
    """Wrap the original engine: rewrite output_dir and task_name, pass everything else through."""
    hp = kwargs["hyperparameters"]
    hp["output_dir"] = hp["output_dir"].replace("magnetic_tile_output", "magnetic_tile_defect_output")
    hp["task_name"] = hp["task_name"].replace("mag_", "mag_defect_", 1)
    return _run(*args, **kwargs)


be.baseline_experiment = baseline_experiment

if __name__ == "__main__":
    runpy.run_module("baseline.magnetic_tile_baseline", run_name="__main__", alter_sys=True)
