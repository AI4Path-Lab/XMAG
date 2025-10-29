# Copyright (c) 2025 Ziyu Su
# Licensed under the PolyForm Noncommercial License 1.0.0
# See the LICENSE file or https://polyformproject.org/licenses/noncommercial/1.0.0/ for details.

import os
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from typing import Optional, List, Tuple, Dict, Callable

from .transform import PathologyTransforms

class PathologyPairDataset(Dataset):
    """
    Dataset class for paired pathology images at different mag.
    return: (high_mag_image, low_mag_image)
    """
    def __init__(
        self,
        data_dir: str,
        data_tag: str = "TCGA_DS_",
        mag_high: int = 20,
        mag_low: int = 5,
        patch_size: int = 224,
        transform = None,
        num_augmentations: int = 2  # Number of augmented views to generate
    ):
        self.data_dir = data_dir
        self.data_tag = data_tag
        self.mag_high = mag_high
        self.mag_low = mag_low
        self.patch_size = patch_size
        self.transform = transform
        self.num_augmentations = num_augmentations

        # Calculate the magnification ratio
        self.mag_ratio = mag_high / mag_low

        # Get the list of high and low mag images
        self.high_mag_patches = self._find_patches(mag_high)

        print(f"Found {len(self.high_mag_patches)} high mag patches and {len(self.high_mag_patches)*16} low mag patches.")

    def _find_patches(self, mag):
        """Find all patches for a given magnification."""
        patch_pattern = os.path.join(self.data_dir, f"{self.data_tag}*", f"{mag}x", "*/patches/*/*.png")
        return sorted(glob.glob(patch_pattern))

    def __len__(self):
        """Return the number of high mag patches."""
        return len(self.high_mag_patches)

    def __getitem__(self, idx):
        # Load high mag image
        high_img_path = self.high_mag_patches[idx]
        high_img = Image.open(high_img_path).convert('RGB')

        # Generate low mag image by resizing
        low_img = high_img.resize((self.patch_size, self.patch_size), Image.Resampling.LANCZOS)
        
        # Generate multiple augmentations
        high_views = []
        low_views = []
        
        for _ in range(self.num_augmentations):
            if self.transform:
                # Apply the same random transforms to both images
                high_aug, low_aug = self.transform(high_img.copy(), low_img.copy())
            else:
                high_aug = transforms.ToTensor()(high_img)
                low_aug = transforms.ToTensor()(low_img)
            
            high_views.append(high_aug)
            low_views.append(low_aug)
        
        return high_views, low_views

class PathologyDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for pathology distillation."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config['data']['batch_size']
        self.num_workers = config['data']['num_workers']
        self.pin_memory = config['data']['pin_memory']
        
        # Dataset paths
        self.train_path = config['data']['train_path']
        self.val_path = config['data']['val_path']
        self.data_tag = config['data'].get('data_tag', 'TCGA_DS_')
        
        # Magnification levels
        self.mag_high = config['data']['magnification_high']
        self.mag_low = config['data']['magnification_low']
        
        # Patch size
        self.patch_size_teacher = config['data']['patch_size_teacher']
        self.patch_size_student = config['data']['patch_size_student']
    
    def setup(self, stage=None):
        """Set up the datasets."""
        if stage == 'fit' or stage is None:
            # Set up transforms
            train_transform = PathologyTransforms.get_paired_transform(self.patch_size_teacher, self.patch_size_student, 
            is_train=True)
                
            val_transform = PathologyTransforms.get_paired_transform(self.patch_size_teacher, self.patch_size_student, 
            is_train=False)
            
            # Set up datasets
            self.train_dataset = PathologyPairDataset(
                data_dir=self.train_path,
                data_tag=self.data_tag,
                mag_high=self.mag_high,
                mag_low=self.mag_low,
                patch_size=self.patch_size_student, # this is the target size of low mag when resizing the high mag
                transform=train_transform
            )
            
            self.val_dataset = PathologyPairDataset(
                data_dir=self.val_path,
                data_tag=self.data_tag,
                mag_high=self.mag_high,
                mag_low=self.mag_low,
                patch_size=self.patch_size_student, # this is the target size of low mag when resizing the high mag
                transform=val_transform,
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )