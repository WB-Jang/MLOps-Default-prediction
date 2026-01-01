"""Data augmentation utilities for tabular data."""
import torch


def tabular_augment(x_cat, x_num, mask_ratio=0.15):
    """
    Augment tabular data by randomly masking features.

    Args:
        x_cat: Categorical features tensor
        x_num: Numerical features tensor
        mask_ratio: Ratio of features to mask

    Returns:
        Augmented categorical and numerical features
    """
    # Categorical augmentation
    cat_mask = torch.rand(x_cat.shape, device=x_cat.device) < mask_ratio
    x_cat_aug = x_cat.clone()
    x_cat_aug[cat_mask] = 0

    # Numerical augmentation
    num_mask = torch.rand(x_num.shape, device=x_num.device) < mask_ratio
    x_num_aug = x_num.clone()
    x_num_aug[num_mask] = 0.0

    return x_cat_aug, x_num_aug
