import albumentations as A
import numpy as np
from torch.utils.data import Dataset

from data.common_ops import letterbox_image_np, letterbox_mask_np


class SegDatasetForFinetune(Dataset):
    """
    base class for dataset for finetune
    """

    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def letterbox_imagenp(self, image: np.ndarray, size):
        return letterbox_image_np(image, size)

    def letterbox_mask_1ch(self, mask: np.ndarray, size: tuple) -> np.ndarray:
        return letterbox_mask_np(mask, size)
