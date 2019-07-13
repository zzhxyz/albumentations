from __future__ import absolute_import

import numpy as np
import torch

from ..core.transforms_interface import DualTransform


__all__ = ['ToTensor']


def array_to_tensor(image, 
    dtype=torch.float32, 
    ensure_channel_dim=True, 
    one_hot_channels=None, 
    binarize_threshold=None):
    assert len(image.shape) in {2,3}, 'Only images of shape HW or HWC are supported'
    assert not (one_hot_channels and binarize_threshold), 'one_hot_channels and binarize_threshold are mutually exclusive'

    if len(image.shape) == 2 and ensure_channel_dim:
        image = np.expand_dims(image, -1)

    if len(mask.shape) == 3:
        # HWC -> CHW
        mask = np.moveaxis(image, -1, 0)

    if one_hot_channels > 0:
        mask = np.eye(one_hot_channels)[mask]

    if binarize_threshold:
        mask = mask > binarize_threshold

    return torch.from_numpy(mask).type(dtype)


class ToTensor(DualTransform):
    """Convert image and mask to `torch.Tensor`.

    Args:
        image_dtype : torch data type of image tensor
        image_ensure_channel_dim (bool): ensures that image tensor will have dummy channel dimension 
            for single-channel input image.
        mask_dtype : torch data type of mask tensor
        mask_ensure_channel_dim (bool): ensures that mask tensor will have dummy channel dimension 
            for single-channel mask image.
        mask_one_hot_channels: Defines number of channels for one-hot encoding mask.
    """

    def __init__(self, 
            image_dtype=torch.float32, 
            image_ensure_channel_dim=True,
            mask_dtype=torch.float32,
            mask_ensure_channel_dim=True,
            mask_one_hot_channels=None):
        super(ToTensor, self).__init__(always_apply=True, p=1.)
        self.image_dtype = image_dtype
        self.image_ensure_channel_dim = image_ensure_channel_dim
        self.mask_dtype = mask_dtype
        self.mask_ensure_channel_dim = mask_ensure_channel_dim
        self.mask_one_hot_channels = mask_one_hot_channels


    def apply(self, image, **kwargs):
        return array_to_tensor(image, 
            dtype=self.image_dtype, 
            ensure_channel_dim=self.image_ensure_channel_dim)

    def apply_to_mask(self, image, **kwargs):
        return array_to_tensor(image,
            dtype=self.mask_dtype,
            ensure_channel_dim=self.mask_ensure_channel_dim,
            one_hot_channels=self.mask_one_hot_channels)

    def get_transform_init_args_names(self):
        return ('image_dtype', 
            'image_ensure_channel_dim', 
            'mask_dtype', 
            'mask_ensure_channel_dim', 
            'mask_one_hot_channels')
