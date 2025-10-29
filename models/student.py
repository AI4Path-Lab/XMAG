# Copyright (c) 2025 Ziyu Su
# Licensed under the PolyForm Noncommercial License 1.0.0
# See the LICENSE file or https://polyformproject.org/licenses/noncommercial/1.0.0/ for details.

# models/student.py
import torch
import torch.nn as nn
from torchvision import transforms
# from timm import create_model
from transformers import AutoImageProcessor, AutoModel


class StudentModel(nn.Module):
    """Student Vision Transformer."""
    def __init__(self, model_name='dinov2_vitb14', embed_dim=768, pretrained=True):
        super().__init__()

        self.model = torch.hub.load('facebookresearch/dinov2', model_name)

        self.embed_dim = embed_dim

        self.transform = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        
    def forward(self, x):
        # Input shape: (B, 3, 224, 224)
        B, C, H, W = x.shape
        assert H == 224 and W == 224, "Input image size must be 224x224"

        x = self.transform(x)  # Normalize the patches

        outputs = self.model.forward_features(x)
        
        # Separate CLS token and patch tokens
        cls_token = outputs['x_norm_clstoken']
        patch_tokens = outputs['x_norm_patchtokens']
        
        return cls_token, patch_tokens