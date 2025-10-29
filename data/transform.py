# Copyright (c) 2025 Ziyu Su
# Licensed under the PolyForm Noncommercial License 1.0.0
# See the LICENSE file or https://polyformproject.org/licenses/noncommercial/1.0.0/ for details.

# data/transforms.py
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
from typing import Tuple
from PIL import Image


class PathologyTransforms:
    """Transformations for pathology images that apply identical augmentations to paired images."""
    
    @staticmethod
    def get_paired_transform(patch_size_teacher: int, patch_size_student: int, is_train: bool = True):
        """
        Get transform function that applies identical augmentations to both images.
        """
        def paired_transform(img_high, img_low):
            # Ensure both images are the correct size
            assert img_high.size == (patch_size_teacher, patch_size_teacher)

            assert img_low.size == (patch_size_student, patch_size_student)
            
            if is_train:
                # Apply identical random transforms to both images
                
                # Random horizontal flip
                if torch.rand(1) < 0.5:
                    img_high = TF.hflip(img_high)
                    img_low = TF.hflip(img_low)
                
                # Random vertical flip
                if torch.rand(1) < 0.5:
                    img_high = TF.vflip(img_high)
                    img_low = TF.vflip(img_low)
                
                # Random color jitter
                if torch.rand(1) < 0.5:
                    brightness = 0.1 * torch.rand(1).item() + 0.95  # [0.95, 1.05]
                    contrast = 0.1 * torch.rand(1).item() + 0.95    # [0.95, 1.05]
                    saturation = 0.1 * torch.rand(1).item() + 0.95  # [0.95, 1.05]
                    hue = 0.05 * (torch.rand(1).item() * 2 - 1)     # [-0.05, 0.05]
                    
                    img_high = TF.adjust_brightness(img_high, brightness)
                    img_high = TF.adjust_contrast(img_high, contrast)
                    img_high = TF.adjust_saturation(img_high, saturation)
                    img_high = TF.adjust_hue(img_high, hue)
                    
                    img_low = TF.adjust_brightness(img_low, brightness)
                    img_low = TF.adjust_contrast(img_low, contrast)
                    img_low = TF.adjust_saturation(img_low, saturation)
                    img_low = TF.adjust_hue(img_low, hue)
            
            # Convert to tensor and normalize
            img_high = TF.to_tensor(img_high)
            img_low = TF.to_tensor(img_low)
            
            # img_high = TF.normalize(img_high, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # img_low = TF.normalize(img_low, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            
            return img_high, img_low
        
        return paired_transform